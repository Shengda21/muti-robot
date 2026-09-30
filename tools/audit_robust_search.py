"""Reconstruct the complete beams from stored exact scores without bootstrapping."""
from collections import Counter
import csv
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
BUDGETS = (32, 64, 128, 256)


def identity(row):
    c = row['candidate']
    return tuple(c['robots']), tuple(c['order']), c['opener'], c['closer']


def main():
    counts = Counter()
    per_method = Counter()
    suite = json.loads((ROOT/'data/transfer_tasks.json').read_text())
    expected = {t['id'] for t in suite if t['split']=='evaluation' and not t.get('excluded')}
    files = sorted((ROOT/'results/major_revision/search').glob('*.json'))
    assert {p.stem for p in files} == expected
    for file in files:
        data = json.loads(file.read_text())
        counts['tasks'] += 1
        assert data['T'] == min(p['nominal']['C'] for p in data['pools'].values())
        assert set(data['runs']) == {'6101','6102','6103'}
        for seed, runs in data['runs'].items():
            assert len(runs) == 7
            for method, run in runs.items():
                rows = run['records']
                assert len(rows) == 256 and len({identity(r) for r in rows}) == 256
                counts['search_runs'] += 1
                counts['compiled_proposals'] += 256
                assert sum(r['pruned'] for r in rows) == run['pruned']
                for budget in BUDGETS:
                    snap = run['snapshots'][str(budget)]
                    assert snap['risk_evaluations'] == sum(not r['pruned'] for r in rows[:budget])
                    assert snap['pruned'] == sum(r['pruned'] for r in rows[:budget])
            base = runs['llm_risk_none']
            base_rows = base['records']
            assert not any(r['pruned'] for r in base_rows)
            # This beam is rebuilt from all exact scores, not trusted raw beam indices.
            full_beam = []
            for i, row in enumerate(base_rows):
                full_beam = sorted(full_beam+[i], key=lambda j:(base_rows[j]['J'],base_rows[j]['C'],j))[:4]
                assert row['beam_indices'] == full_beam
                for method in ('llm_risk_nominal','llm_risk_critical'):
                    other = runs[method]['records'][i]
                    assert identity(other) == identity(row)
                    assert other['beam_indices'] == full_beam
                    value = other['audit_J'] if other['pruned'] else other['J']
                    assert abs(value-row['J']) <= 1e-10
                    # Independent bound arithmetic for the fixed 0.1/0.2/0.4 T grid.
                    b = (0.1+0.2+0.4)*data['T']/9
                    bound = row['C']+b*min(1., row['C']/data['T'])
                    assert abs(other['bound']-bound) < 1e-9
                    assert bound <= value+1e-9
                    if other['pruned']:
                        assert value > other['threshold']+1e-9
                        assert i not in full_beam
                        per_method[method] += 1
                        counts['pruned_candidates_verified'] += 1
                    counts['paired_candidate_and_beam_steps_verified'] += 1
            for method in ('llm_risk_nominal','llm_risk_critical'):
                counts['paired_pruning_runs'] += 1
                for budget in BUDGETS:
                    a,b = base['snapshots'][str(budget)],runs[method]['snapshots'][str(budget)]
                    assert a['best_index'] == b['best_index']
                    assert a['J'] == b['J'] and a['test_J'] == b['test_J']
                    counts['matched_budget_endpoints'] += 1
    summary = json.loads((ROOT/'results/major_revision/analysis/summary.json').read_text())
    with (ROOT/'results/major_revision/analysis/task_metrics.csv').open() as handle:
        task_rows = list(csv.DictReader(handle))
    # Independently reproduce the primary estimate using explicit scene means.
    differences = {}
    for scene in summary['scenes']:
        entries = [r for r in task_rows if r['scene']==scene and r['budget']=='256']
        a = {r['task']:float(r['test_J']) for r in entries if r['method']=='llm_risk_none'}
        b = {r['task']:float(r['test_J']) for r in entries if r['method']=='llm_nominal_none'}
        assert a.keys() == b.keys()
        differences[scene] = statistics.mean(a[t]-b[t] for t in a)
    primary_mean = statistics.mean(differences.values())
    assert abs(primary_mean-summary['contrasts']['primary_llm_risk_minus_nominal_search']['metrics']['test_J']['mean']) < 1e-10
    result = {'status':'passed', 'counts':dict(counts), 'pruned_by_method':dict(per_method),
        'false_prunes':0,'trajectory_mismatches':0,'primary_scene_mean_crosscheck':primary_mean,
        'statistics_unit':'30 equally weighted scenes; 3 search seeds averaged per task before within-scene task averaging',
        'scope':'stored exact scores and full beams independently reconstructed; no new bootstrap or performance rerun'}
    output = ROOT/'results/major_revision/analysis/independent_audit.json'
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
