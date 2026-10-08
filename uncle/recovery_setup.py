"""Prepare matched study checkpoints, with no recovery adaptation yet."""
from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path
import time

import torch
from torch.utils.data import Subset

from . import checkpoint
from .hypernet import HyperNetwork, build_target
from .paired_diagnostic import _forget_generated, _frozen_hash, _score
from .streams import build_tasks
from .trainer import UnCLe


def validate_manifest(manifest, partition_path):
    """Refuse wrong partitions, duplicate paths, and overlapping groups."""
    partition_path = Path(partition_path)
    partition = json.loads(partition_path.read_text())
    if hashlib.sha256(partition_path.read_bytes()).hexdigest() != manifest['partition_sha256']:
        raise ValueError('Manifest class partition fingerprint differs.')
    expected = next(g['wnids'] for g in partition['tasks']
                    if g['task_id'] == manifest['task_id'])
    if manifest['version'] != 1 or [r['wnid'] for r in manifest['classes']] != expected:
        raise ValueError('Manifest classes or version differ.')
    all_paths = []
    for label, row in enumerate(manifest['classes']):
        if row['label'] != label:
            raise ValueError('Manifest labels differ from the class order.')
        groups = [row['initial_train'], row['recovery_reserve'], row['validation']]
        if [len(set(g)) for g in groups] != [400, 100, 50] or [len(g) for g in groups] != [400, 100, 50]:
            raise ValueError('Expected 400 initial, 100 reserved, and 50 validation images per class.')
        if any(set(groups[a]) & set(groups[b]) for a, b in ((0, 1), (0, 2), (1, 2))):
            raise ValueError('Image groups overlap.')
        for group in groups:
            for name in group:
                path = Path(name)
                if path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name:
                    raise ValueError('Use safe relative image paths.')
        for budget in manifest['budgets']:
            if not 0 < budget <= 100 or row['recovery'][str(budget)] != row['recovery_reserve'][:budget]:
                raise ValueError('Recovery budgets must be nested reserve prefixes.')
        all_paths.extend(name for g in groups for name in g)
    if len(all_paths) != len(set(all_paths)):
        raise ValueError('Image paths occur in more than one class.')


def build_study_tasks(root, manifest, partition_path, retained, control_task):
    """Only the target uses 400 images per class; retained tasks use 500."""
    validate_manifest(manifest, partition_path)
    target = str(manifest['task_id'])
    retained = tuple(map(str, retained))
    if target in retained or str(control_task) in (*retained, target):
        raise ValueError('Target, retained tasks, and unseen control must be distinct.')
    tasks = build_tasks(root=root, partition_path=partition_path,
                        include=(*retained, target), download=False)
    for slot, group in (('train', 'initial_train'), ('test', 'validation')):
        dataset = tasks[target][slot]
        available = {dataset.base.image_path(i).relative_to(Path(root)).as_posix(): j
                     for j, i in enumerate(dataset.indices)}
        paths = [name for row in manifest['classes'] for name in row[group]]
        if not set(paths) <= available.keys():
            raise ValueError('Saved image paths do not match this extracted dataset.')
        tasks[target][slot] = Subset(dataset, [available[name] for name in paths])
    return tasks


def readiness(before, after, target, retained, *, min_retained=40.0, max_spill=3.0,
              max_target=12.0, min_start=25.0):
    """Summed absolute spill and mean retained accuracy are separate metrics."""
    spill = sum(abs(after[t] - before[t]) for t in retained)
    mean = sum(after[t] for t in retained) / len(retained)
    checks = {'target_learned': before[target] >= min_start,
              'target_suppressed': after[target] <= max_target,
              'retained_mean': mean > min_retained, 'spill': spill < max_spill}
    return {'checks': checks, 'eligible': all(checks.values()),
            'spill_points': spill, 'retained_mean_percent': mean,
            'target_before_percent': before[target], 'target_after_percent': after[target],
            'thresholds': {'min_retained_exclusive': min_retained, 'max_spill_exclusive': max_spill,
                           'max_target_inclusive': max_target, 'min_start_inclusive': min_start}}


def run_setup_stage(config, tasks, output, *, stage, target='3', retained=('0','9','5','17','1','7','14'),
                    source=None, manifest_sha256, code_version, on_saved=None,
                    progress=True):
    """Run one stage and seed; resume only from the last completed request.

    Shared learns the first retained task once. Both branches restore that exact
    checkpoint. The unlearned branch inserts the target after the shared lesson.
    No mid-lesson optimizer is resumed. Failed readiness is recorded, not tuned.
    """
    retained = tuple(map(str, retained))
    target = str(target)
    if stage not in ('shared', 'never_learned', 'unlearned') or not retained:
        raise ValueError('Choose shared, never_learned, or unlearned and retained tasks.')
    if target in retained or len(set(retained)) != len(retained):
        raise ValueError('Retained tasks must be distinct and exclude the target.')
    shared = (('learn', retained[0]),)
    requests = shared if stage == 'shared' else shared + (
        ((('learn', target),) if stage == 'unlearned' else ()) +
        tuple(('learn', t) for t in retained[1:]) +
        ((('forget', target),) if stage == 'unlearned' else ()))
    config = replace(config, requests=requests)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    settings = {'stage': stage, 'config': {k: v for k, v in vars(config).items() if k != 'device'},
                'manifest_sha256': manifest_sha256, 'code_version': code_version,
                'retained': retained, 'target': target,
                'source_sha256': _file_hash(source) if source else None,
                'objective': 'generated_target_weights_unit_gaussian',
                'resume_policy': 'last completed request; fresh optimizer per request'}
    normalized = json.loads(json.dumps(settings))
    contract = output / 'settings.json'
    if contract.exists() and json.loads(contract.read_text()) != normalized:
        raise ValueError('Existing stage settings differ. Choose a new experiment ID.')
    if not contract.exists():
        _write(contract, settings)
    latest = output / 'checkpoint.pt'
    payload = checkpoint.load(latest) if latest.exists() else (checkpoint.load(source) if source else None)
    if stage != 'shared' and source is None:
        raise ValueError('Both branches require the same shared checkpoint.')
    if stage != 'shared':
        shared_payload = checkpoint.load(source)
        if shared_payload is None or [(r['action'], r['task']) for r in shared_payload['history']] != list(shared):
            raise ValueError('Source must contain only the shared first lesson.')
        for key in vars(config):
            if key not in ('requests', 'device') and shared_payload['config'].get(key) != getattr(config, key):
                raise ValueError('Shared source configuration differs: ' + key)
        del shared_payload
    history = list(payload['history']) if payload else []
    actual = tuple((r['action'], r['task']) for r in history)
    if actual != requests[:len(history)] or len(history) > len(requests):
        raise ValueError('Saved history is not a prefix of this stage.')
    seen = list(payload['seen']) if payload else []
    forgotten = list(payload['forgotten']) if payload else []
    previous = dict(payload['previous']) if payload else {}
    report = {'status': 'running', **settings, 'history': history,
              'epoch_records': [], 'forget_trace': [], 'readiness': None,
              'gpu': torch.cuda.get_device_name(0) if config.torch_device.type == 'cuda' else None,
              'torch': torch.__version__, 'cuda': torch.version.cuda}
    if history and stage == 'unlearned' and history[-1]['action'] == 'forget':
        report['readiness'] = history[-1]['readiness']
    def publish(event):
        _write(output / 'report.json', report)
        if on_saved:
            devices = list(range(torch.cuda.device_count())) if config.torch_device.type == 'cuda' else []
            with torch.random.fork_rng(devices=devices):
                on_saved(event, output)
    h = trainer = model = None
    try:
        torch.manual_seed(config.seed)
        if config.torch_device.type == 'cuda':
            torch.cuda.manual_seed_all(config.seed)
        model = build_target(config)
        h = HyperNetwork(model, config)
        trainer = UnCLe(h, config, model, tasks, progress=progress)
        if payload:
            checkpoint.restore(payload, hypernet=h, uncle=trainer)
            del payload
            if _score(trainer, seen) != previous:
                raise ValueError('Restored validation scores differ; stop before training.')
        publish('starting')
        for index in range(len(history), len(requests)):
            action, task = requests[index]
            before = dict(previous)
            began = time.perf_counter()
            report['epoch_records'] = []
            if action == 'learn':
                def epoch(row):
                    scores = _score(trainer, [*seen, task])
                    report['epoch_records'].append({**row, 'task': task, 'accuracies': scores})
                    publish('epoch')
                losses = trainer.learn(task, seen, on_epoch=epoch)
                seen.append(task)
                after = report['epoch_records'][-1]['accuracies']
                row = {'index': index, 'action': action, 'task': task, 'before': before,
                       'after': after, 'seen': list(seen), 'forgotten': list(forgotten),
                       'final_loss': losses[-1], 'burn_in': None,
                       'epochs': list(report['epoch_records'])}
            else:
                pre = output / 'before_unlearning.pt'
                if not pre.exists():
                    checkpoint.save(pre, config=config, hypernet=h, uncle=trainer,
                                    history=history, seen=seen, forgotten=forgotten,
                                    previous=previous, costs=[], setup_seconds=0.0)
                publish('before_unlearning')
                frozen = _frozen_hash(h, trainer, seen)
                report['forget_trace'] = []
                steps = config.burn_in_for(0)
                def forget_step(row):
                    record = dict(row)
                    if row['step'] in (0, 1, steps) or row['step'] % 20 == 0:
                        record['accuracies'] = _score(trainer, seen)
                    report['forget_trace'].append(record)
                    if 'accuracies' in record:
                        publish('forget_step')
                _forget_generated(trainer, target, retained, steps, forget_step)
                if _frozen_hash(h, trainer, seen) != frozen:
                    raise ValueError('Unlearning changed frozen codes or running statistics.')
                after = _score(trainer, seen)
                gate = readiness(before, after, target, retained)
                report['readiness'] = gate
                forgotten.append(target)
                row = {'index': index, 'action': action, 'task': target, 'before': before,
                       'after': after, 'seen': list(seen), 'forgotten': list(forgotten),
                       'burn_in': steps, 'final_loss': report['forget_trace'][-1]['total_loss_before'],
                       'readiness': gate, 'forget_trace': list(report['forget_trace'])}
            row['seconds'] = time.perf_counter() - began
            history.append(row)
            previous = after
            checkpoint.save(latest, config=config, hypernet=h, uncle=trainer,
                            history=history, seen=seen, forgotten=forgotten,
                            previous=previous, costs=[], setup_seconds=0.0)
            publish('request_complete')
        report.update(status='complete', final_accuracies=previous,
                      checkpoint_sha256=_file_hash(latest))
        publish('complete')
        return report
    except Exception as error:
        report.update(status='interrupted', error_type=type(error).__name__, error=str(error))
        _write(output / 'report.json', report)
        raise
    finally:
        trainer = h = model = None
        gc.collect()
        if config.torch_device.type == 'cuda':
            torch.cuda.empty_cache()


def _write(path, data):
    temporary = path.with_suffix('.json.writing')
    temporary.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()
