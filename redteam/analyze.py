"""Recompute audit summaries from fresh hidden-test JSON; no model computation.

Rank normalizers depend on later attempts, which are unavailable here. These
explicit panels demonstrate consequences of the user's ranking contract; they
are not a prediction of the eventual leaderboard normalizers.
"""
from pathlib import Path
import hashlib
import json
import statistics

ROOT = Path(__file__).resolve().parent
CASES = json.loads((ROOT / 'cases.json').read_text())
CONTRACT = json.loads((ROOT / 'reference_contract.json').read_text())
REFERENCE = json.loads((ROOT / 'reference_test_at_start.json').read_text())
METRICS = ('psnr', 'loaded_gaussians', 'visible_gaussians')


def read_case(name):
    folder = ROOT / name
    status = json.loads((folder / 'result.json').read_text())
    assert status['candidate']['returncode'] == 0, name
    assert status['candidate_source_sha256'] == hashlib.sha256((folder / 'method.py').read_bytes()).hexdigest(), name
    scores = json.loads((folder / 'scores_test.json').read_text())
    return status, scores


def normalizers(scores, panel):
    result = {}
    for experiment, contract in CONTRACT['experiments'].items():
        for method in REFERENCE['experiments'][experiment]['results']:
            for metric, spec in contract['metrics'].items():
                if not spec['coefficient']:
                    continue
                values = [scores[name]['experiments'][experiment]['results'].get(method, {}).get(metric)
                          for name in panel]
                values = [v for v in values if v is not None]
                spread = statistics.pstdev(values) if values else 0.0
                result[experiment, method, metric] = spread or 1.0
    return result


def rank(scores, deviations):
    total = 0.0
    terms = {}
    for (experiment, method, metric), deviation in deviations.items():
        value = scores['experiments'][experiment]['results'].get(method, {}).get(metric)
        if value is None:  # The supplied rank contract skips missing values.
            continue
        contract = CONTRACT['experiments'][experiment]
        spec = contract['metrics'][metric]
        direction = 1.0 if spec['higher_is_better'] else -1.0
        reference = REFERENCE['experiments'][experiment]['results'][method][metric]
        term = contract['weight'] * abs(spec['coefficient']) * direction * (value - reference) / deviation
        terms['/'.join((experiment, method, metric))] = term
        total += term
    return dict(total=total, terms=terms)


def main():
    records = {name: read_case(name) for name in CASES}
    fresh = {name: score for name, (status, score) in records.items()
             if status.get('test', {}).get('fresh_scores')}
    honest = [name for name in fresh if not name.startswith('shortcut')]
    bounded = [name for name in fresh if name not in
               ('shortcut_impossible_cost', 'shortcut_black_impossible_cost')]
    panels = {'H_honest': honest, 'B_without_negative_counts': bounded, 'A_all_fresh': list(fresh)}
    output = {'reference_file': 'reference_test_at_start.json',
              'normalization': 'population standard deviation per experiment/method/metric; zero spread uses 1',
              'panels': {}, 'raw_metrics': {}, 'seed_spread': {}}
    for label, names in panels.items():
        deviations = normalizers(fresh, names)
        scores = {name: rank(score, deviations) for name, score in fresh.items()}
        seeds = [scores['reference_seed' + str(i)]['total'] for i in range(3)]
        output['panels'][label] = dict(cases=names, normalizers={'/'.join(key): value for key, value in deviations.items()},
                                     scores=scores, seed_score_range=[min(seeds), max(seeds)],
                                     seed_score_spread=max(seeds)-min(seeds))
    for name, (status, score) in records.items():
        output['raw_metrics'][name] = dict(fresh=status['test']['fresh_scores'],
                                          test_returncode=status['test']['returncode'],
                                          candidate_seconds=status['candidate']['seconds'],
                                          experiments=score['experiments'])
    for experiment, result in REFERENCE['experiments'].items():
        output['seed_spread'][experiment] = {}
        for method in result['results']:
            output['seed_spread'][experiment][method] = {}
            for metric in METRICS:
                values = [fresh['reference_seed' + str(i)]['experiments'][experiment]['results'][method][metric]
                          for i in range(3)]
                output['seed_spread'][experiment][method][metric] = dict(values=values, minimum=min(values), maximum=max(values),
                                                                         spread=max(values)-min(values),
                                                                         population_std=statistics.pstdev(values))
    (ROOT / 'analysis.json').write_text(json.dumps(output, indent=2) + '\n')
    print('case\tPSNR\tloaded\tvisible\tH\tB\tA')
    for name in CASES:
        row = fresh.get(name, {}).get('experiments', {}).get('synthetic_lod_quality_efficiency', {}).get('results', {}).get('lodge', {})
        metrics = [f'{row[m]:.6f}' if m in row else 'missing' for m in METRICS]
        ranks = [f'{output["panels"][panel]["scores"][name]["total"]:.9g}' if name in fresh else 'FAILED'
                 for panel in panels]
        print('\t'.join([name] + metrics + ranks))


if __name__ == '__main__':
    main()
