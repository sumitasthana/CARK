"""Continue a selected checkpoint by one task and measure the learning losses."""

from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch

from . import checkpoint as checkpointing
from .config import Config
from .hypernet import HyperNetwork, build_target
from .telemetry import environment
from .trainer import UnCLe


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def inspect_source(path):
    """Read the saved history and scores without building or training a model."""
    source = checkpointing.load(path)
    if source is None:
        raise FileNotFoundError(path)
    _validate_source(source)
    return {'path': str(Path(path)), 'config': source['config'],
            'completed_tasks': list(source['seen']), 'accuracies': source['previous'],
            'history': source['history']}


def _validate_source(source):
    history = source['history']
    if not history or any(row['action'] != 'learn' for row in history) or source['forgotten']:
        raise ValueError('Select a learning-only checkpoint with completed tasks.')
    prefix = tuple((r['action'], r['task']) for r in history)
    if prefix != tuple(source['config']['requests'][:len(prefix)]):
        raise ValueError('Saved history does not match the saved request list.')
    seen = [r['task'] for r in history]
    if len(set(seen)) != len(seen) or seen != list(source['seen']):
        raise ValueError('Saved learned tasks do not match the history.')
    if set(seen) != set(source['tasks_with_codes']) or set(seen) != set(source['task_buffers']):
        raise ValueError('Saved task codes or running statistics do not match learned tasks.')
    if source['previous'] != history[-1]['after']:
        raise ValueError('Saved scores do not match the final history row.')


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.writing')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def run_learning_diagnostic(source_path, output, new_task, tasks, *, beta=None,
                            device=None, code_version=None, progress=True,
                            on_epoch=None):
    """Branch from a saved model, train only new_task, and save every epoch.

    The source is never overwritten. beta changes only the new lesson's
    protection coefficient. An interrupted lesson restarts from the source;
    partial measurements remain available in report.json.
    """
    source_path, output = Path(source_path).resolve(), Path(output).resolve()
    new_task = str(new_task)
    source_digest = file_hash(source_path)
    source = checkpointing.load(source_path)
    _validate_source(source)
    original = Config(**source['config'])
    if new_task in source['seen']:
        raise ValueError(f'Task {new_task} is already learned. Select an earlier checkpoint.')
    if new_task not in original.tasks:
        raise ValueError(f'Task {new_task} is not in the saved task partition.')
    if beta is not None and (not math.isfinite(beta) or beta < 0):
        raise ValueError('beta must be finite and nonnegative')
    prefix = tuple((r['action'], r['task']) for r in source['history'])
    config = replace(original, requests=prefix + (('learn', new_task),),
                     beta=original.beta if beta is None else beta,
                     device=original.device if device is None else device)
    needed = [*source['seen'], new_task]
    if not set(needed).issubset(tasks):
        raise ValueError('Provide images for all learned tasks and the new task.')
    output.mkdir(parents=True, exist_ok=True)
    if source_path == output / 'checkpoint.pt':
        raise ValueError('The source checkpoint and output checkpoint must differ.')
    contract = json.loads(json.dumps({'source_path': str(source_path),
        'source_sha256': source_digest, 'config': vars(config),
        'code_version': code_version}))
    contract_path = output / 'settings.json'
    report_path = output / 'report.json'
    if contract_path.exists():
        if json.loads(contract_path.read_text(encoding='utf-8')) != contract:
            raise ValueError('This output folder belongs to a different source, task, beta, or code version.')
        if report_path.exists():
            existing = json.loads(report_path.read_text(encoding='utf-8'))
            if existing.get('status') == 'complete':
                if not (output / 'checkpoint.pt').is_file():
                    raise ValueError('Completed report is missing its saved model.')
                return existing
            raise ValueError('This lesson was interrupted. Keep its partial report and choose a new attempt folder.')
    elif any(output.iterdir()):
        raise ValueError('Output folder contains unrecognized files. Choose a new folder.')
    else:
        _write_json(contract_path, contract)

    report = {'status': 'starting', **contract, 'new_task': new_task,
              'protected_tasks': list(source['seen']), 'source_beta': original.beta,
              'source_history': source['history'], 'starting_accuracies': {},
              'trace': [], 'epochs': []}
    _write_json(report_path, report)
    began = time.perf_counter()
    try:
        target = build_target(config)
        hypernet = HyperNetwork(target, config)
        trainer = UnCLe(hypernet, config, target, tasks, progress=progress)
        checkpointing.restore(source, hypernet=hypernet, uncle=trainer)
        source_history = source['history']
        expected = source['previous']
        del source
        report['environment'] = environment(config, hypernet, tasks)
        devices = list(range(torch.cuda.device_count())) if config.torch_device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            report['starting_accuracies'] = {t: trainer.accuracy(t) for t in report['protected_tasks']}
        report['starting_scores_match'] = all(
            abs(report['starting_accuracies'][t] - expected[t]) < 1e-6 for t in expected)
        if not report['starting_scores_match']:
            raise ValueError('Restored scores differ from the saved scores. Check the source, images, and runtime before training.')
        # One shared-generator gradient sample per epoch, on its first batch.
        batches = math.ceil(len(tasks[new_task]['train']) / config.batch_size)
        gradient_steps = {1 + epoch * batches for epoch in range(config.epochs)}
        report['gradient_steps'] = sorted(gradient_steps)
        report['status'] = 'running'
        _write_json(report_path, report)

        def step(record):
            if record['step'] == 0:
                return
            row = {k: v for k, v in record.items()
                   if k not in {'indices', 'images', 'labels'}}
            for key in ('loss', 'task_loss', 'protection_loss', 'weighted_protection_loss'):
                if not math.isfinite(row[key]):
                    raise FloatingPointError(f'Non-finite {key}; training stopped.')
            report['trace'].append(row)

        def epoch(record):
            row = dict(record)
            row['accuracies'] = {t: trainer.accuracy(t) for t in needed}
            row['accuracy_stage'] = 'after_epoch'
            report['epochs'].append(row)
            _write_json(report_path, report)
            if on_epoch is not None:
                on_epoch(row)

        losses = trainer.learn(new_task, report['protected_tasks'], on_step=step,
                               on_epoch=epoch, gradient_steps=gradient_steps)
        final_scores = report['epochs'][-1]['accuracies']
        history = [*source_history, {'index': len(source_history), 'action': 'learn',
            'task': new_task, 'before': expected, 'after': final_scores,
            'seen': needed, 'forgotten': [], 'final_loss': losses[-1], 'burn_in': None}]
        checkpointing.save(output / 'checkpoint.pt', config=config, hypernet=hypernet,
            uncle=trainer, history=history, seen=needed, forgotten=[],
            previous=final_scores, costs=[], setup_seconds=0.0)
        if file_hash(source_path) != source_digest:
            raise ValueError('Source checkpoint changed during this experiment.')
        report.update(status='complete', final_accuracies=final_scores,
                      history=history, seconds=time.perf_counter() - began)
        _write_json(report_path, report)
        return json.loads(report_path.read_text(encoding='utf-8'))
    except Exception as error:
        report['status'] = 'interrupted'
        report['error'] = str(error)
        _write_json(report_path, report)
        raise


def summarize_diagnostic(report):
    """Compare learning and retention without selecting a winner automatically."""
    if report.get("status") != "complete":
        raise ValueError("Only completed lessons can be compared.")
    protected = report["protected_tasks"]
    before, after = report["starting_accuracies"], report["final_accuracies"]
    changes = {task: after[task] - before[task] for task in protected}
    return {"beta": report["config"]["beta"], "new_task": report["new_task"],
            "new_task_accuracy": after[report["new_task"]],
            "mean_old_accuracy_before": sum(before[t] for t in protected) / len(protected),
            "mean_old_accuracy_after": sum(after[t] for t in protected) / len(protected),
            "mean_old_change": sum(changes.values()) / len(protected),
            "largest_old_drop": max(0.0, max(-change for change in changes.values())),
            "old_task_changes": changes, "source_sha256": report["source_sha256"]}
