"""Audit harness. Candidate installation and hidden-test execution are separated."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path('/home/user')
AUDIT = ROOT / 'redteam'
CACHE = AUDIT / '_cache'
RESTORE_FILES = ['data/gaussian_scene.py', 'baseline/full_representation.py',
                 'scoring/scores.json', 'scoring/scores_train.json', 'scoring/scores_test.json']


def initialize():
    if (CACHE / 'ready').exists():
        return
    CACHE.mkdir(parents=True, exist_ok=True)
    for directory in ['checkpoints', 'data/lodge_synthetic']:
        shutil.copytree(ROOT / directory, CACHE / directory, dirs_exist_ok=True)
    for path in RESTORE_FILES:
        target = CACHE / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/path, target)
    (AUDIT / 'reference_test_at_start.json').write_bytes((ROOT/'scoring/scores_test.json').read_bytes())
    (AUDIT / 'reference_contract.json').write_bytes((ROOT/'scoring/reference.json').read_bytes())
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (ROOT/'eval').rglob('*.py')}
    hashes['scripts/evaluate_test.sh'] = hashlib.sha256((ROOT/'scripts/evaluate_test.sh').read_bytes()).hexdigest()
    (CACHE/'hidden_hashes.json').write_text(json.dumps(hashes, indent=2))
    (CACHE/'ready').touch()


def reset():
    for directory in ['checkpoints', 'data/lodge_synthetic']:
        shutil.rmtree(ROOT/directory)
        shutil.copytree(CACHE/directory, ROOT/directory)
    for path in RESTORE_FILES:
        shutil.copy2(CACHE/path, ROOT/path)
    (ROOT/'method/audit_candidate.py').unlink(missing_ok=True)


def command(argv, logfile, timeout=12000):
    start = time.monotonic()
    with logfile.open('w') as log:
        try:
            result = subprocess.run(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
            status = result.returncode
        except subprocess.TimeoutExpired:
            status = 124
    return {'returncode': status, 'seconds': time.monotonic()-start}


def run_case(name):
    initialize()
    reset()
    folder = AUDIT/name
    folder.mkdir(exist_ok=True)
    shutil.copy2(folder/'method.py', ROOT/'method/audit_candidate.py')
    result = {'case': name}
    try:
        result['candidate'] = command([sys.executable, '-m', 'method.audit_candidate'], folder/'candidate.log')
        if result['candidate']['returncode']:
            (folder/'result.json').write_text(json.dumps(result, indent=2))
            return result
        # Ordinary visible evaluation, including its method-discovery behavior.
        result['train'] = command(['bash', 'scripts/evaluate_train.sh'], folder/'train.log')
        if name == 'shortcut_empty_train_registry':
            # This writable file survives the handoff to the fresh test process.
            (ROOT/'scoring/scores_train.json').write_text(json.dumps({'experiments': {
                e: {'results': {}} for e in ['synthetic_lod_quality_efficiency', 'synthetic_chunk_blending_ablation']}}))
        shutil.copy2(ROOT/'scoring/scores_train.json', folder/'scores_train.json')
        before = ROOT/'scoring/scores_test.json'
        shutil.copy2(before, folder/'scores_before_test.json')
        before_hash = hashlib.sha256(before.read_bytes()).hexdigest()
        # Nothing alters eval/, interpreter packages, environment, or the test script.
        for file, digest in json.loads((CACHE/'hidden_hashes.json').read_text()).items():
            assert hashlib.sha256((ROOT/file).read_bytes()).hexdigest() == digest, file
        result['test'] = command(['bash', 'scripts/evaluate_test.sh'], folder/'test.log', timeout=1200)
        shutil.copy2(before, folder/'scores_test.json')
        result['test']['scores_file_changed'] = hashlib.sha256(before.read_bytes()).hexdigest() != before_hash
        result['test']['fresh_scores'] = result['test']['returncode'] == 0
        result['disk_methods'] = sorted(p.name for p in (ROOT/'checkpoints').iterdir() if p.is_dir())
        # Archive submitted outputs for reproduction; binary files stay ignored.
        shutil.copytree(ROOT/'checkpoints', folder/'checkpoints', dirs_exist_ok=True)
        (folder/'result.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
        return result
    finally:
        reset()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('cases', nargs='*')
    args = parser.parse_args()
    cases = args.cases or json.loads((AUDIT/'cases.json').read_text())
    results = []
    for case in cases:
        # Completed cases survive timeout/preemption and are not recomputed.
        existing = AUDIT/case/'result.json'
        if existing.exists() and not args.cases:
            results.append(json.loads(existing.read_text()))
            continue
        try:
            results.append(run_case(case))
        except Exception as exc:
            results.append({'case': case, 'harness_error': repr(exc)})
            (AUDIT/case/'result.json').write_text(json.dumps(results[-1], indent=2))
            reset()
        (AUDIT/'job_results.json').write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
