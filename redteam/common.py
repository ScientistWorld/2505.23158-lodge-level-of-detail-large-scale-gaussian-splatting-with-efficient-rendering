"""Candidate-side utilities. No hidden evaluator, hidden split, or paper access."""
from pathlib import Path
import itertools
import json
import shutil
import time

import numpy as np

DATA = Path('data/lodge_synthetic')
CKPT = Path('checkpoints')
NAMES = ['lodge', 'lod_dynamic', 'nearest_chunk']
EXPS = ['synthetic_lod_quality_efficiency', 'synthetic_chunk_blending_ablation']


def emit(path, images, stats):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path / 'renders.npz', images=images)
    (path / 'stats.json').write_text(json.dumps({'per_view': stats}, indent=2))


def zero_stats(n, value=0):
    return [dict(loaded=value, visible=value, tile_cost=value) for _ in range(n)]


def replace_proposed(images, stats):
    for name in NAMES:
        emit(CKPT / name, images, stats)


def load_images(name='lodge'):
    return np.load(CKPT / name / 'renders.npz')['images']


def reference_stats(name='lodge'):
    return json.loads((CKPT / name / 'stats.json').read_text())['per_view']


def run_lodge(steps=12, seed=0, thresholds=None, output=None, lr=.035):
    """Run original computational steps; offset every explicit random seed."""
    import torch
    from method import lodge as m
    from data.gaussian_scene import load_dataset, load_train_targets
    torch.set_num_threads(1)
    original_ft, original_km, original_ro = m.gradient_finetune_lod, m.kmeans, m.random_orientation_cameras
    def ft(*args, **kwargs):
        kwargs['seed'] += seed
        return original_ft(*args, **kwargs)
    def km(points, k, seed_arg=3, iters=30):
        return original_km(points, k, seed=seed_arg + seed, iters=iters)
    def ro(cameras, seed_arg=None, per_camera=2, **kwargs):
        return original_ro(cameras, seed=kwargs.get('seed', seed_arg) + seed, per_camera=per_camera)
    m.gradient_finetune_lod, m.kmeans, m.random_orientation_cameras = ft, km, ro
    try:
        scene, train, test, _ = load_dataset(DATA)
        targets = load_train_targets(DATA)
        start = time.monotonic()
        thresholds = thresholds or m.select_thresholds(scene, train, targets, 2, [3., 5., 7., 9.], .055, 1.)
        levels = m.build_lod_levels(scene, train, targets, thresholds, .055, 1., steps, lr)
        chunks = m.build_chunks(levels, thresholds, train, .220)
        build_time = time.monotonic() - start
        train_metrics = {}
        for name, variant in [('lodge', 'opacity_blend'), ('lod_dynamic', 'lod_dynamic'), ('nearest_chunk', 'nearest_chunk')]:
            renders, stats = [], []
            for cam in test:
                image, st = m.render_variant(levels, thresholds, chunks, cam, variant)
                renders.append(image)
                stats.append(st)
            root = Path(output) if output else CKPT
            m.save_run(root / name, np.stack(renders), stats,
                       dict(steps=steps, seed_offset=seed, thresholds=thresholds,
                            level_gaussians=[len(s['means']) for s in levels], build_seconds=build_time))
        # Select brute-force operating points exclusively on training cameras.
        from data.gaussian_scene import psnr
        qualities, loaded, visible = [], [], []
        for cam, gt in zip(train, targets):
            img, st = m.render_opacity_blend(levels, chunks, cam)
            qualities.append(psnr(np.round(img*255)/255, np.round(gt*255)/255))
            loaded.append(st['loaded'])
            visible.append(st['visible'])
        train_metrics = dict(psnr=float(np.mean(qualities)), loaded=float(np.mean(loaded)),
                             visible=float(np.mean(visible)), build_seconds=build_time)
        return train_metrics
    finally:
        m.gradient_finetune_lod, m.kmeans, m.random_orientation_cameras = original_ft, original_km, original_ro


def brute_force():
    """Enumerate the entire visible two-threshold grid, selecting on train only."""
    root = Path('redteam/brute_force_thresholds')
    records = []
    for pair in itertools.combinations([3., 5., 7., 9.], 2):
        trial = root / 'trials' / ('_'.join(str(int(v)) for v in pair))
        metrics = run_lodge(steps=12, thresholds=list(pair), output=trial)
        # A simple public training-only quality/cost scalar, with normalized costs.
        objective = metrics['psnr'] - 3. * (metrics['loaded'] + metrics['visible']) / 848.
        records.append(dict(thresholds=list(pair), objective=objective, **metrics))
        (root / 'search.json').write_text(json.dumps(records, indent=2))
    best = max(records, key=lambda x: x['objective'])
    src = root / 'trials' / ('_'.join(str(int(v)) for v in best['thresholds']))
    for name in NAMES:
        shutil.copytree(src / name, CKPT / name, dirs_exist_ok=True)
    (root / 'selection.json').write_text(json.dumps(best, indent=2))


def shortcut(kind):
    """Each branch deliberately instantiates a scoring shortcut or failed probe."""
    pred = load_images()
    n = len(pred)
    if kind == 'visible_answers':
        gt = np.load(DATA / 'test_gt.npz')['images']
        for name in NAMES:
            emit(CKPT / name, gt, reference_stats(name))
    elif kind == 'rebuild_answers':
        from data.gaussian_scene import generate_scene, load_dataset, render_scene
        base, _, cameras, _ = load_dataset(DATA)
        dense = generate_scene(fine_grid=24, coarse_grid=12)
        answers = np.stack([np.clip(.7 * render_scene(base, c) + .3 * render_scene(dense, c), 0, 1)
                            for c in cameras])
        for name in NAMES:
            emit(CKPT / name, answers, reference_stats(name))
    elif kind in ['zero_cost', 'impossible_cost']:
        value = 0 if kind == 'zero_cost' else -1.e12
        for name in NAMES:
            emit(CKPT / name, load_images(name), zero_stats(n, value))
    elif kind == 'constant_black':
        replace_proposed(np.zeros_like(pred), zero_stats(n))
    elif kind == 'replace_targets':
        black = np.zeros_like(pred)
        np.savez_compressed(DATA / 'test_gt.npz', images=black)
        replace_proposed(black, zero_stats(n))
    elif kind == 'baseline_only':
        gt = np.load(DATA / 'test_gt.npz')['images']
        # Proposed methods are unchanged; only a scored comparison artifact moves.
        emit(CKPT / 'full_representation', gt, zero_stats(n))
    elif kind == 'missing_artifacts':
        for name in ['full_representation'] + NAMES:
            shutil.rmtree(CKPT / name)
    elif kind == 'empty_train_registry':
        replace_proposed(np.zeros_like(pred), zero_stats(n))
        (Path('scoring') / 'scores_train.json').write_text(json.dumps(
            {'experiments': {name: {'results': {}} for name in EXPS}}))
    elif kind == 'unregistered_method':
        # An extra disk method is never discovered, even by the visible evaluator.
        emit(CKPT / 'shortcut_unregistered_method', np.zeros_like(pred), zero_stats(n))
    elif kind == 'direct_scores':
        forged = {'experiments': {e: {'results': {name: {'psnr': 9999., 'loaded_gaussians': -1e20,
                  'visible_gaussians': -1e20} for name in ['lodge', 'opacity_blend']}} for e in EXPS}}
        (Path('scoring') / 'scores_test.json').write_text(json.dumps(forged))
        for name in NAMES:
            meta = json.loads((CKPT/name/'stats.json').read_text())
            meta.update(success=True, psnr=9999., elapsed_seconds=0.)
            (CKPT/name/'stats.json').write_text(json.dumps(meta))
    elif kind == 'null_cost':
        replace_proposed(pred, zero_stats(n, None))
    elif kind == 'empty_predictions':
        replace_proposed(pred[:0], zero_stats(n))
    elif kind == 'nonfinite_pixels':
        replace_proposed(np.full_like(pred, np.nan), zero_stats(n))
    else:
        raise ValueError(kind)
