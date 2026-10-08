"""Compare L(new) with U(old), L(new) from one saved learning-only model."""
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from . import checkpoint
from .config import Config, random_stream
from .hypernet import HyperNetwork, build_target
from .learning_diagnostic import _validate_source, _write_json, file_hash
from .telemetry import environment
from .trainer import UnCLe


def tensor_hash(tensor):
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str((str(value.dtype), tuple(value.shape))).encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _hash_named(values):
    digest = hashlib.sha256()
    for name, value in sorted(values):
        digest.update(name.encode())
        digest.update(tensor_hash(value).encode())
    return digest.hexdigest()


def _frozen_hash(hypernet, trainer, old_tasks):
    values = [('chunks', hypernet.chunk_codes)]
    values += [('code.' + t, hypernet.task_codes[t]) for t in old_tasks]
    values += [(t + '.' + name, value) for t in old_tasks
               for name, value in trainer.task_buffers[t].items()]
    return _hash_named(values)


def generated_noise_loss(hypernet, task, noise, samples):
    """Eq. 3 uses the same generated target parameters as preservation.

    Average summed squared distance to independent unit Gaussian vectors.
    Include output scales and BatchNorm offsets, which raw_for bypasses.
    """
    weights = hypernet.weights_for(task)
    generated = torch.cat([v.reshape(-1) for v in weights.values()])
    return sum((generated - torch.randn(generated.shape, generator=noise,
                device=generated.device, dtype=generated.dtype)).square().sum()
               for _ in range(samples)) / samples


def _forget_generated(trainer, task, protected, steps, callback):
    snapshot = trainer.hypernet.snapshot()
    trainer._reference = {}
    trainable = trainer.hypernet.generator_parameters()
    trainer.hypernet.requires_grad_(False)
    for value in trainable:
        value.requires_grad_(True)
    config = trainer.config
    rate = config.forgetting_learning_rate or config.learning_rate
    optimizer = torch.optim.Adam(trainable, lr=rate)
    noise = random_stream(config.seed, 'forget', task, trainer.device)
    trainer.hypernet.train()
    trainer._notify_forget_step(callback, {'step': 0})
    for step in range(1, steps + 1):
        noise_loss = generated_noise_loss(trainer.hypernet, task, noise, config.noise_samples)
        preservation = trainer.preserve(protected, snapshot)
        weighted = config.gamma * noise_loss
        loss = weighted + preservation
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite unlearning loss.')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        trainer._notify_forget_step(callback, {'step': step,
            'noise_loss_before': noise_loss.item(), 'weighted_noise_before': weighted.item(),
            'preserve_before': preservation.item(), 'total_loss_before': loss.item()})


def _restore_training_rng(source, device):
    torch.set_rng_state(source['rng'].cpu())
    if device.type == 'cuda' and source['cuda_rng'] is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in source['cuda_rng']])


def _score(trainer, tasks):
    devices = list(range(torch.cuda.device_count())) if trainer.device.type == 'cuda' else []
    modes = [(module, module.training) for module in
             [*trainer.hypernet.modules(), *trainer.target.modules()]]
    try:
        with torch.random.fork_rng(devices=devices):
            return {t: trainer.accuracy(t) for t in tasks}
    finally:
        for module, mode in modes:
            module.training = mode


def run_paired_diagnostic(source_path, output, forget_task, new_task, tasks, *,
                          device=None, code_version=None, on_saved=None, progress=True):
    """Two fixed branches; no search, adaptive stopping, or automatic resume.

    Unlearning uses generated weights and unit Gaussian noise, unlike the
    historical raw-output unlearning loop. After U, learning uses the source
    RNG again, identical code initialization, and the same seeded loader.
    Previously forgotten task codes stay protected during subsequent learning.
    All old images are evaluation-only. Learning reads only new-task training.
    """
    source_path, output = Path(source_path).resolve(), Path(output).resolve()
    forget_task, new_task = str(forget_task), str(new_task)
    source_hash = file_hash(source_path)
    source = checkpoint.load(source_path)
    _validate_source(source)
    original = Config(**source['config'])
    old_tasks = list(source['seen'])
    if forget_task not in old_tasks or new_task in old_tasks or new_task not in original.tasks:
        raise ValueError('Choose a learned forget task and a not-yet-learned new task.')
    if not set([*old_tasks, new_task]).issubset(tasks):
        raise ValueError('Provide evaluation tasks and new-task training data.')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Paired output already has files. Preserve them and choose a new attempt.')
    if not math.isfinite(original.gamma) or original.gamma <= 0:
        raise ValueError('Gamma must be finite and positive.')
    output.mkdir(parents=True, exist_ok=True)
    retained = [t for t in old_tasks if t != forget_task]
    contract = {'source_sha256': source_hash, 'forget_task': forget_task,
        'new_task': new_task, 'code_version': code_version,
        'source_config': source['config'], 'unlearning_objective': 'generated_weights_unit_gaussian',
        'historical_difference': 'Historical forget() applies noise to raw outputs; this runner uses generated target weights for both terms.',
        'learning_protected_tasks': old_tasks, 'unlearning_protected_tasks': retained,
        'learning_rng_policy': 'Restore source PyTorch RNG before L in each branch; seeded task code and batch loader.',
        'limitation': 'One paired diagnostic; altered request timing, no erasure proof or different-seed generalization.'}
    _write_json(output / 'selection.json', contract)
    reports = {}
    prefix = tuple((r['action'], r['task']) for r in source['history'])
    for branch, forget in (('learning_only', False), ('unlearn_then_learn', True)):
        branch_dir = output / branch
        branch_dir.mkdir()
        requests = prefix + ((('forget', forget_task),) if forget else ()) + (('learn', new_task),)
        config = replace(original, requests=requests, device=device or original.device)
        settings = {**contract, 'branch': branch, 'config': vars(config)}
        _write_json(branch_dir / 'settings.json', settings)
        report = {'status': 'starting', **settings, 'starting_accuracies': {},
            'forget_trace': [], 'learning_trace': [], 'epochs': []}
        history = list(source['history'])
        forgotten = []
        began = time.perf_counter()

        def save(stage, row=None):
            _write_json(branch_dir / 'report.json', report)
            if on_saved is not None:
                devices = list(range(torch.cuda.device_count())) if config.torch_device.type == 'cuda' else []
                with torch.random.fork_rng(devices=devices):
                    on_saved(branch, stage, row)

        save('starting')
        try:
            target = build_target(config)
            hypernet = HyperNetwork(target, config)
            trainer = UnCLe(hypernet, config, target, tasks, progress=progress)
            checkpoint.restore(source, hypernet=hypernet, uncle=trainer)
            frozen_hash = _frozen_hash(hypernet, trainer, old_tasks)
            report['initial_model_sha256'] = _hash_named(list(hypernet.state_dict().items()))
            report['environment'] = environment(config, hypernet, tasks)
            before = _score(trainer, old_tasks)
            report['starting_accuracies'] = before
            if before != source['previous']:
                raise ValueError('Restored scores do not match the source. Stop before training.')
            report['status'] = 'running'
            report['frozen_before_sha256'] = frozen_hash
            after_forget = before
            if forget:
                steps = config.burn_in_for(0)
                def forget_step(row):
                    record = dict(row)
                    if row['step'] == 0 or row['step'] == 1 or row['step'] % 10 == 0 or row['step'] == steps:
                        record['accuracies'] = _score(trainer, old_tasks)
                    report['forget_trace'].append(record)
                    if 'accuracies' in record:
                        save('forget_step', record)
                _forget_generated(trainer, forget_task, retained, steps, forget_step)
                after_forget = _score(trainer, old_tasks)
                forgotten = [forget_task]
                history.append({'index': len(history), 'action': 'forget', 'task': forget_task,
                    'before': before, 'after': after_forget, 'seen': old_tasks,
                    'forgotten': forgotten, 'burn_in': steps,
                    'final_loss': report['forget_trace'][-1]['total_loss_before']})
                checkpoint.save(branch_dir / 'after_unlearning.pt', config=config, hypernet=hypernet,
                    uncle=trainer, history=history, seen=old_tasks, forgotten=forgotten,
                    previous=after_forget, costs=[], setup_seconds=0.0)
                report['after_unlearning_checkpoint_sha256'] = file_hash(branch_dir / 'after_unlearning.pt')
            report['before_learning_accuracies'] = after_forget
            if _frozen_hash(hypernet, trainer, old_tasks) != frozen_hash:
                raise ValueError('Unlearning changed old codes, chunk codes, or BatchNorm buffers.')
            # Match L randomness deliberately; do not use noise-consumed RNG as a confound.
            _restore_training_rng(source, config.torch_device)
            report['learning_rng_sha256'] = tensor_hash(torch.get_rng_state())
            report['learning_cuda_rng_sha256'] = ([tensor_hash(v) for v in torch.cuda.get_rng_state_all()]
                if config.torch_device.type == 'cuda' else [])
            save('before_learning')
            count = math.ceil(len(tasks[new_task]['train']) / config.batch_size)
            gradient_steps = {1 + epoch * count for epoch in range(config.epochs)}

            def learning_step(row):
                if row['step'] == 0:
                    report['new_task_code_sha256'] = tensor_hash(hypernet.task_codes[new_task])
                    return
                record = {k:v for k,v in row.items() if k not in ('images', 'labels', 'indices')}
                record['indices'] = row['indices'].tolist()
                record['input_sha256'] = _hash_named([('images', row['images']), ('labels', row['labels'])])
                if not all(math.isfinite(record[k]) for k in ('loss', 'task_loss', 'protection_loss')):
                    raise FloatingPointError('Non-finite learning loss.')
                report['learning_trace'].append(record)

            def epoch(row):
                record = {**row, 'accuracies': _score(trainer, [*old_tasks, new_task])}
                report['epochs'].append(record)
                save('epoch', record)

            losses = trainer.learn(new_task, old_tasks, on_step=learning_step,
                on_epoch=epoch, gradient_steps=gradient_steps)
            final_scores = report['epochs'][-1]['accuracies']
            if _frozen_hash(hypernet, trainer, old_tasks) != frozen_hash:
                raise ValueError('Learning changed old codes, chunk codes, or BatchNorm buffers.')
            history.append({'index': len(history), 'action': 'learn', 'task': new_task,
                'before': after_forget, 'after': final_scores, 'seen': [*old_tasks, new_task],
                'forgotten': forgotten, 'burn_in': None, 'final_loss': losses[-1]})
            checkpoint.save(branch_dir / 'checkpoint.pt', config=config, hypernet=hypernet,
                uncle=trainer, history=history, seen=[*old_tasks, new_task], forgotten=forgotten,
                previous=final_scores, costs=[], setup_seconds=0.0)
            report.update(status='complete', final_accuracies=final_scores, history=history,
                checkpoint_sha256=file_hash(branch_dir / 'checkpoint.pt'),
                seconds=time.perf_counter() - began, frozen_after_sha256=frozen_hash)
            if file_hash(source_path) != source_hash:
                raise ValueError('Original source checkpoint changed.')
            save('complete')
            reports[branch] = report
            del trainer, hypernet, target
            torch.cuda.empty_cache()
        except Exception as error:
            report.update(status='interrupted', error=str(error))
            _write_json(branch_dir / 'report.json', report)
            raise
    a, b = reports['learning_only'], reports['unlearn_then_learn']
    match_fields = ('initial_model_sha256', 'starting_accuracies', 'learning_rng_sha256',
                    'learning_cuda_rng_sha256', 'new_task_code_sha256')
    checks = {key: a[key] == b[key] for key in match_fields}
    checks['batch_order'] = [r['indices'] for r in a['learning_trace']] == [r['indices'] for r in b['learning_trace']]
    checks['input_values'] = [r['input_sha256'] for r in a['learning_trace']] == [r['input_sha256'] for r in b['learning_trace']]
    checks['update_count'] = len(a['learning_trace']) == len(b['learning_trace']) == config.epochs * count
    def changes(scores, reference):
        return {t: scores[t] - reference[t] for t in retained}
    direct = changes(b['before_learning_accuracies'], a['starting_accuracies'])
    ordinary = changes(a['final_accuracies'], a['starting_accuracies'])
    combined = changes(b['final_accuracies'], b['starting_accuracies'])
    difference = changes(b['final_accuracies'], a['final_accuracies'])
    chance = 100 / config.classes_per_task
    post_u, final_u = b['before_learning_accuracies'][forget_task], b['final_accuracies'][forget_task]
    result = {'status': 'complete' if all(checks.values()) else 'invalid_pair',
        'selection': contract, 'pair_checks': checks, 'retained_tasks': retained,
        'starting_accuracies': a['starting_accuracies'],
        'after_unlearning_accuracies': b['before_learning_accuracies'],
        'learning_only_final_accuracies': a['final_accuracies'],
        'unlearn_then_learn_final_accuracies': b['final_accuracies'],
        'retained_changes': {'unlearning_only': direct, 'learning_only': ordinary,
            'unlearning_and_learning': combined, 'paired_final_difference': difference},
        'retained_mean_changes': {k: sum(v.values()) / len(v) if v else None
            for k,v in [('unlearning_only', direct), ('learning_only', ordinary),
                        ('unlearning_and_learning', combined), ('paired_final_difference', difference)]},
        'forget_task': {'task': forget_task, 'starting_accuracy': a['starting_accuracies'][forget_task],
            'starting_above_chance': a['starting_accuracies'][forget_task] > chance,
            'chance_accuracy': chance, 'after_unlearning_accuracy': post_u,
            'after_later_learning_accuracy': final_u, 'increase_after_unlearning': final_u - post_u,
            'reached_chance_or_lower': post_u <= chance,
            'returned_above_chance': post_u <= chance and final_u > chance},
        'new_task_difference': b['final_accuracies'][new_task] - a['final_accuracies'][new_task],
        'branch_checkpoints': {k:v['checkpoint_sha256'] for k,v in reports.items()}}
    _write_json(output / 'paired_comparison.json', result)
    if not all(checks.values()):
        raise ValueError('Pair controls differ. Keep artifacts; do not interpret accuracy differences as unlearning effects.')
    return json.loads((output / 'paired_comparison.json').read_text())
