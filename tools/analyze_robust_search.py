"""Scene-cluster analysis and publication plot for the frozen revision protocol.

All files must be present unless --allow-partial is explicitly requested. Partial
outputs are labeled incomplete. Search seeds and task sizes are averaged within
scene before estimating paired intervals; proposal counts are never samples.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import random
import statistics

ROOT = Path(__file__).resolve().parents[1]
BUDGETS = (32, 64, 128, 256)
SEEDS = (6101, 6102, 6103)
BOOTSTRAP_SEED = 20260930
METHODS = ('llm_nominal_none', 'llm_risk_none', 'llm_risk_nominal',
           'llm_risk_critical', 'uniform_nominal_none', 'uniform_risk_none', 'uniform_random')
PLOT_METHODS = ('llm_risk_none', 'llm_nominal_none', 'uniform_risk_none', 'uniform_random')
METRICS = ('C', 'J', 'test_J', 'C_over_T', 'J_over_T', 'test_J_over_T',
           'risk_evaluations', 'evaluated_fraction', 'cpu_ms', 'wall_ms',
           'score_cpu_ms', 'compile_cpu_ms', 'change_from_llm_nominal',
           'fraction_change_from_llm_nominal')


def quantile(values, p):
    values = sorted(values)
    x = (len(values)-1)*p
    i = int(x)
    return values[i]+(x-i)*(values[min(i+1, len(values)-1)]-values[i])


def interval(scene_values, draws=10_000):
    values = list(scene_values.values())
    if not values:
        return None
    rng = random.Random(BOOTSTRAP_SEED)
    samples = [statistics.mean(values[rng.randrange(len(values))] for _ in values)
               for _ in range(draws)]
    return {'mean': statistics.mean(values), 'ci95': [quantile(samples, .025), quantile(samples, .975)],
            'n_scenes': len(values), 'bootstrap_draws': draws, 'bootstrap_seed': BOOTSTRAP_SEED}


def group_scene(rows, metric):
    groups = defaultdict(list)
    for row in rows:
        if row.get(metric) is not None:
            groups[row['scene']].append(row[metric])
    return {scene: statistics.mean(groups[scene]) for scene in sorted(groups)}


def describe(rows):
    return {metric: {'mean': statistics.mean(values.values()), 'n_scenes': len(values)}
            for metric in METRICS if (values := group_scene(rows, metric))}


def paired(task_rows, a, b, budget, draws):
    groups = defaultdict(dict)
    for row in task_rows:
        if row['budget'] == budget and row['method'] in (a, b):
            groups[(row['task'], row['scene'])][row['method']] = row
    differences = []
    for (task, scene), entries in groups.items():
        if a in entries and b in entries:
            differences.append({'task': task, 'scene': scene,
                **{metric: entries[a][metric]-entries[b][metric]
                   for metric in METRICS if entries[a].get(metric) is not None
                   and entries[b].get(metric) is not None}})
    return {'method_a': a, 'method_b': b, 'difference': 'a minus b',
            'n_tasks': len(differences), 'budget': budget,
            'metrics': {metric: interval(group_scene(differences, metric), draws)
                        for metric in ('C', 'test_J', 'test_J_over_T', 'risk_evaluations', 'cpu_ms')}}


def collect(files):
    raw_rows, pool_rows = [], []
    counts = Counter()
    task_info = []
    for file in files:
        record = json.loads(file.read_text())
        counts['task_records'] += 1
        counts['tasks_with_llm_pool'] += 'llm' in record['pools']
        assert record['pruning_trace_matches'] and record['pruned_scores_independently_checked']
        T = record['T']
        task_info.append({'task': record['task'], 'scene': record['scene'], 'family': record['family'],
                          'n': record['n'], 'T': T, 'llm_seconds': record.get('llm_seconds')})
        baseline = record['pools'].get('llm', {}).get('nominal', {}).get('test_J')
        for source, pool in record['pools'].items():
            for selector in ('nominal', 'risk'):
                row = pool[selector]
                pool_rows.append({'task': record['task'], 'scene': record['scene'],
                    'family': record['family'], 'source': source, 'selector': selector,
                    'size': pool['size'], 'C': row['C'], 'J': row['J'], 'test_J': row['test_J'],
                    'test_J_over_T': row['test_J']/T})
        assert set(record['runs']) == {str(s) for s in SEEDS}
        for seed, runs in record['runs'].items():
            if 'llm_risk_none' in runs:
                base = runs['llm_risk_none']
                for method in ('llm_risk_nominal', 'llm_risk_critical'):
                    variant = runs[method]
                    assert [r['candidate'] for r in base['records']] == [r['candidate'] for r in variant['records']]
                    assert [r['beam_indices'] for r in base['records']] == [r['beam_indices'] for r in variant['records']]
                    for budget in BUDGETS:
                        a, b = base['snapshots'][str(budget)], variant['snapshots'][str(budget)]
                        assert a['best_index'] == b['best_index'] and abs(a['J']-b['J']) < 1e-9
                    counts['paired_pruning_runs_checked'] += 1
            for method, run in runs.items():
                counts['search_runs'] += 1
                counts['compiled_proposals'] += len(run['records'])
                for row in run['records']:
                    if row['pruned']:
                        counts['audited_pruned_proposals'] += 1
                        assert row['audit_J'] > row['threshold']+1e-9
                for budget in BUDGETS:
                    snap = run['snapshots'][str(budget)]
                    raw_rows.append({'task': record['task'], 'scene': record['scene'],
                        'family': record['family'], 'n': record['n'], 'seed': int(seed),
                        'method': method, 'budget': budget, 'T': T,
                        'C': snap['C'], 'J': snap['J'], 'test_J': snap['test_J'],
                        'C_over_T': snap['C']/T, 'J_over_T': snap['J']/T,
                        'test_J_over_T': snap['test_J']/T,
                        'risk_evaluations': snap['risk_evaluations'],
                        'evaluated_fraction': snap['risk_evaluations']/budget,
                        'cpu_ms': 1000*snap['cpu_elapsed_s'], 'wall_ms': 1000*snap['elapsed_s'],
                        'score_cpu_ms': 1000*snap['score_cpu_s'],
                        'compile_cpu_ms': 1000*snap['compile_cpu_s'],
                        'change_from_llm_nominal': snap['test_J']-baseline if baseline else None,
                        'fraction_change_from_llm_nominal': snap['test_J']/baseline-1 if baseline else None})
    # Exactly three search seeds become one row for each task/method/budget.
    groups = defaultdict(list)
    for row in raw_rows:
        groups[(row['task'], row['method'], row['budget'])].append(row)
    task_rows = []
    for entries in groups.values():
        assert len(entries) == 3 and {r['seed'] for r in entries} == set(SEEDS)
        first = entries[0]
        task_rows.append({k: first[k] for k in ('task', 'scene', 'family', 'n', 'method', 'budget', 'T')} |
                         {metric: statistics.mean(r[metric] for r in entries)
                          if first.get(metric) is not None else None for metric in METRICS})
    return raw_rows, task_rows, pool_rows, task_info, counts


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def make_plot(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8, 'font.family': 'DejaVu Sans',
                         'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 3.75), constrained_layout=True)
    colors = ('#0072B2', '#D55E00', '#009E73', '#CC79A7')
    markers = ('o', 's', '^', 'D')
    labels = ('LLM start, $J$-ranked', 'LLM start, $C$ + rerank',
              'Uniform start, $J$-ranked', 'Uniform sampling only')
    plot_data = {'unit_top': 'mean disturbed completion / common horizon T',
                 'unit_bottom': 'percentage of proposals requiring exact risk evaluation',
                 'aggregation': 'seeds then task sizes averaged within scenes; equal scene weights',
                 'status': summary['status'], 'budgets': list(BUDGETS), 'quality': {}, 'efficiency': {}}
    for method, color, marker, label in zip(PLOT_METHODS, colors, markers, labels):
        values = [summary['quality'][str(b)][method]['test_J_over_T']['mean'] for b in BUDGETS]
        axes[0].plot(BUDGETS, values, marker=marker, color=color, label=label, linewidth=1.1, markersize=4)
        plot_data['quality'][method] = values
    axes[0].set_ylabel('Tested completion / $T$')
    axes[0].set_title('(a) Quality of the best plan found', loc='left', fontsize=8)
    axes[0].legend(frameon=False, fontsize=7, handlelength=1.4,
                   loc='upper center', bbox_to_anchor=(.5, -.23), ncol=2,
                   columnspacing=.8)
    for method, color, marker, label in (
            ('llm_risk_nominal', '#777777', 's', 'Nominal screening'),
            ('llm_risk_critical', '#0072B2', 'o', 'Critical-path screening')):
        values = [100*summary['quality'][str(b)][method]['evaluated_fraction']['mean'] for b in BUDGETS]
        axes[1].plot(BUDGETS, values, marker=marker, color=color, label=label, linewidth=1.1, markersize=4)
        plot_data['efficiency'][method] = values
    axes[1].axhline(100, color='#BBBBBB', linestyle=':', linewidth=.8)
    axes[1].set_ylim(0, 105)
    axes[1].set_ylabel('Evaluated (% of candidates)')
    axes[1].set_title('(b) Candidates receiving a delay evaluation', loc='left', fontsize=8)
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.set_xscale('log', base=2); ax.set_xticks(BUDGETS, [str(b) for b in BUDGETS])
        ax.set_xlabel('Candidates visited'); ax.grid(axis='y', color='#DDDDDD', linewidth=.4)
    axes[0].set_xlabel('')
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix('.pdf'))
    fig.savefig(output.with_suffix('.png'), dpi=220)
    plt.close(fig)
    output.with_name(output.name+'_data.json').write_text(json.dumps(plot_data, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='results/major_revision/search')
    parser.add_argument('--tasks', default='data/transfer_tasks.json')
    parser.add_argument('--output', default='results/major_revision/analysis')
    parser.add_argument('--figure', default='paper/v2/figures/search_effects')
    parser.add_argument('--draws', type=int, default=10_000)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    suite = json.loads((ROOT/args.tasks).read_text())
    evaluation = [t for t in suite if t['split']=='evaluation']
    expected = {t['id'] for t in evaluation if not t.get('excluded')}
    files = sorted(p for p in (ROOT/args.input).glob('*.json') if p.stem in expected)
    present = {p.stem for p in files}
    missing = sorted(expected-present)
    if not files or (missing and not args.allow_partial):
        raise SystemExit(f'Incomplete evaluation: {len(present)}/{len(expected)} tasks; missing {missing[:8]}')
    raw, task_rows, pools, info, counts = collect(files)
    scenes = sorted({r['scene'] for r in info})
    summary = {'status': 'partial, incomplete evaluation' if missing else 'complete evaluation',
        'missing_tasks': missing, 'counts': dict(counts), 'n_tasks': len(info), 'n_scenes': len(scenes),
        'scenes': scenes, 'tasks_by_family': dict(Counter(r['family'] for r in info)),
        'excluded_tasks': [{'task':t['id'],'reason':t['excluded']} for t in evaluation if t.get('excluded')],
        'statistics': 'Three seeds averaged per task; task sizes averaged within scene; scenes equally weighted.',
        'quality': {}, 'families': {}, 'contrasts': {}, 'pools': {}, 'correctness': {
            'false_prunes': 0, 'beam_or_candidate_trace_mismatches': 0,
            'all_reported_pruning_assertions_checked': True}}
    for budget in BUDGETS:
        summary['quality'][str(budget)] = {method: describe([r for r in task_rows if r['budget']==budget and r['method']==method])
                                         for method in METHODS if any(r['method']==method for r in task_rows)}
    for family in ('living','bedroom'):
        summary['families'][family] = {method: describe([r for r in task_rows if r['budget']==256 and r['method']==method and r['family']==family])
                                      for method in METHODS}
    for source in ('llm','uniform'):
        for selector in ('nominal','risk'):
            rows = [r for r in pools if r['source']==source and r['selector']==selector]
            summary['pools'][source+'_'+selector] = describe(rows)
    contrasts = {
        'primary_llm_risk_minus_nominal_search': ('llm_risk_none','llm_nominal_none'),
        'uniform_risk_minus_nominal_search': ('uniform_risk_none','uniform_nominal_none'),
        'llm_seed_minus_uniform_seed_risk_search': ('llm_risk_none','uniform_risk_none'),
        'llm_risk_search_minus_uniform_pool': ('llm_risk_none','uniform_random'),
        'critical_bound_minus_full_scoring': ('llm_risk_critical','llm_risk_none'),
        'critical_bound_minus_nominal_bound': ('llm_risk_critical','llm_risk_nominal'),
    }
    for label, (a,b) in contrasts.items():
        summary['contrasts'][label] = paired(task_rows, a, b, 256, args.draws)
    full = summary['quality']['256'].get('llm_risk_none', {})
    summary['efficiency_256'] = {}
    for method in ('llm_risk_nominal','llm_risk_critical'):
        if method in summary['quality']['256']:
            value = summary['quality']['256'][method]
            summary['efficiency_256'][method] = {
                'evaluations_fraction': value['evaluated_fraction']['mean'],
                'cpu_ratio_to_full': value['cpu_ms']['mean']/full['cpu_ms']['mean'],
                'score_cpu_ratio_to_full': value['score_cpu_ms']['mean']/full['score_cpu_ms']['mean']
                    if full['score_cpu_ms']['mean'] else None}
    output = ROOT/args.output; output.mkdir(parents=True, exist_ok=True)
    write_csv(output/'run_metrics.csv', raw); write_csv(output/'task_metrics.csv', task_rows)
    write_csv(output/'pool_metrics.csv', pools); write_csv(output/'task_inventory.csv', info)
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    lines = ['% Generated from complete recorded outcomes; units and aggregation in summary.json.',
             '% Quality: method & nominal steps & disturbed steps & disturbed / T \\\\']
    labels = {'llm_nominal_none':'LLM + nominal search', 'llm_risk_none':'LLM + interruption search',
              'uniform_nominal_none':'Uniform + nominal search', 'uniform_risk_none':'Uniform + interruption search',
              'uniform_random':'Uniform proposals'}
    for method,label in labels.items():
        r = summary['quality']['256'][method]
        lines.append(f"{label} & {r['C']['mean']:.3f} & {r['test_J']['mean']:.3f} & {r['test_J_over_T']['mean']:.3f} \\\\")
    lines.append('% Efficiency: method & exact evaluations & CPU milliseconds & evaluated percentage \\\\')
    for method,label in [('llm_risk_none','Full scoring'),('llm_risk_nominal','Nominal bound'),('llm_risk_critical','Critical-path bound')]:
        r = summary['quality']['256'][method]
        lines.append(f"{label} & {r['risk_evaluations']['mean']:.1f} & {r['cpu_ms']['mean']:.2f} & {100*r['evaluated_fraction']['mean']:.1f}\\% \\\\")
    (output/'tables.tex').write_text('\n'.join(lines)+'\n')
    if all(m in summary['quality']['256'] for m in PLOT_METHODS) and 'llm_risk_critical' in summary['quality']['256']:
        make_plot(summary, ROOT/args.figure)
    print(json.dumps({'status':summary['status'],'tasks':summary['n_tasks'],'scenes':summary['n_scenes'],
                      'correctness':summary['correctness'],'primary':summary['contrasts']['primary_llm_risk_minus_nominal_search'],
                      'efficiency':summary['efficiency_256']}, indent=2))


if __name__ == '__main__':
    main()
