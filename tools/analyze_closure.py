"""Analyse closure experiments: scene-cluster paired bootstrap (10,000 draws)."""
from __future__ import annotations
import argparse, json, random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRAWS = 10000
PRIMARY = 'm3|closed|D3|s1.0'


def load(directory):
    """cell -> method -> scene -> metric -> mean over (tasks, seeds); equal weight per task."""
    per = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    for f in sorted(Path(directory).glob('t*.json')):
        r = json.loads(f.read_text())
        for cell, v in r['cells'].items():
            T = v['T']
            for seed, methods in v['runs'].items():
                for name, x in methods.items():
                    d = per[cell][name][r['scene']]
                    task_key = r['task']
                    row = {'evals': x['evals'], 'cpu_ms': x['cpu_s'] * 1e3, 'eval_ms': x['score_cpu_s'] * 1e3,
                           'build_ms': x['compile_cpu_s'] * 1e3, 'test_J': x['test_J'], 'test_J_over_T': x['test_J'] / T,
                           'C': x['C']}
                    for k, val in row.items():
                        d[k].append((task_key, seed, val))
    out = {}
    for cell, methods in per.items():
        out[cell] = {}
        for name, scenes in methods.items():
            out[cell][name] = {}
            for scene, metrics in scenes.items():
                out[cell][name][scene] = {}
                for k, rows in metrics.items():
                    by_task = defaultdict(list)
                    for t, s, val in rows: by_task[t].append(val)
                    task_means = [sum(v) / len(v) for v in by_task.values()]
                    out[cell][name][scene][k] = sum(task_means) / len(task_means)
    return out


def boot(fn, scenes, rng):
    est = fn(scenes)
    draws = sorted(fn([scenes[rng.randrange(len(scenes))] for _ in scenes]) for _ in range(DRAWS))
    p = 2 * min(sum(d <= 0 for d in draws), sum(d >= 0 for d in draws)) / DRAWS
    return est, draws[int(.025 * DRAWS)], draws[int(.975 * DRAWS) - 1], max(p, 2 / DRAWS)


def analyse(data, cell_filter=None):
    rows = {}
    for cell, methods in data.items():
        if cell_filter and not cell_filter(cell): continue
        rng = random.Random(20260929)
        scenes = sorted(methods['J_none'])
        g = lambda name, k: (lambda S: sum(methods[name][s][k] for s in S) / len(S))
        res = {'scenes': len(scenes)}
        def mean(name, k, S=scenes): return sum(methods[name][s][k] for s in S) / len(S)
        for name in methods:
            res[name] = {k: mean(name, k) for k in ('evals', 'cpu_ms', 'eval_ms', 'build_ms', 'test_J', 'test_J_over_T')}
        if 'J_nominal' in methods:
            res['extra_eval_saving_pct'] = boot(lambda S: 100 * (1 - sum(methods['J_critical'][s]['evals'] for s in S) /
                                                sum(methods['J_nominal'][s]['evals'] for s in S)), scenes, rng)
            res['eval_saving_vs_none_pct'] = boot(lambda S: 100 * (1 - sum(methods['J_critical'][s]['evals'] for s in S) /
                                                  sum(methods['J_none'][s]['evals'] for s in S)), scenes, rng)
            res['cpu_saving_nom_minus_crit_ms'] = boot(lambda S: sum(methods['J_nominal'][s]['cpu_ms'] - methods['J_critical'][s]['cpu_ms'] for s in S) / len(S), scenes, rng)
            res['cpu_saving_none_minus_crit_ms'] = boot(lambda S: sum(methods['J_none'][s]['cpu_ms'] - methods['J_critical'][s]['cpu_ms'] for s in S) / len(S), scenes, rng)
            res['rho_eval_over_build'] = res['J_none']['eval_ms'] / res['J_none']['build_ms']
        if 'C_rerank_tb' in methods:
            res['J_minus_Ctb_steps'] = boot(lambda S: sum(methods['J_none'][s]['test_J'] - methods['C_rerank_tb'][s]['test_J'] for s in S) / len(S), scenes, rng)
            res['J_minus_Ctb_pctT'] = boot(lambda S: 100 * sum(methods['J_none'][s]['test_J_over_T'] - methods['C_rerank_tb'][s]['test_J_over_T'] for s in S) / len(S), scenes, rng)
            res['Ctb_minus_C_steps'] = boot(lambda S: sum(methods['C_rerank_tb'][s]['test_J'] - methods['C_rerank'][s]['test_J'] for s in S) / len(S), scenes, rng)
        if 'C_rerank' in methods:
            res['J_minus_Crerank_steps'] = boot(lambda S: sum(methods['J_none'][s]['test_J'] - methods['C_rerank'][s]['test_J'] for s in S) / len(S), scenes, rng)
            res['J_minus_Crerank_pctT'] = boot(lambda S: 100 * sum(methods['J_none'][s]['test_J_over_T'] - methods['C_rerank'][s]['test_J_over_T'] for s in S) / len(S), scenes, rng)
            res['J_minus_uniform_steps'] = boot(lambda S: sum(methods['J_none'][s]['test_J'] - methods['uniform'][s]['test_J'] for s in S) / len(S), scenes, rng)
        rows[cell] = res
    return rows


def holm(rows, key):
    cells = [c for c in rows if key in rows[c] and c != PRIMARY]
    if PRIMARY in rows and key in rows[PRIMARY]: rows[PRIMARY][key + '_holm_p'] = rows[PRIMARY][key][3]
    order = sorted(cells, key=lambda c: rows[c][key][3]); n = len(order); run = 0.0
    for i, c in enumerate(order):
        run = max(run, min(1.0, (n - i) * rows[c][key][3])); rows[c][key + '_holm_p'] = run


def fmt(t): return f'{t[0]:.2f} [{t[1]:.2f}, {t[2]:.2f}]'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    a = ap.parse_args()
    rows = analyse(load(ROOT / a.input))
    for key in ('J_minus_Ctb_steps', 'J_minus_Crerank_steps'): holm(rows, key)
    out = ROOT / a.output; out.mkdir(parents=True, exist_ok=True)
    (out / 'summary.json').write_text(json.dumps(rows, indent=1))
    lines = []
    for cell, r in rows.items():
        lines.append(f'## {cell}  ({r["scenes"]} scenes)')
        for k in ('extra_eval_saving_pct', 'eval_saving_vs_none_pct', 'cpu_saving_nom_minus_crit_ms', 'cpu_saving_none_minus_crit_ms',
                  'J_minus_Ctb_steps', 'J_minus_Ctb_pctT', 'Ctb_minus_C_steps', 'J_minus_Crerank_steps', 'J_minus_Crerank_pctT', 'J_minus_uniform_steps'):
            if k in r: lines.append(f'- {k}: {fmt(r[k])}' + (f'  p={r[k][3]:.4f}' + (f', Holm p={r[k+"_holm_p"]:.4f}' if k + '_holm_p' in r else '') if k.startswith('J_minus_C') and 'steps' in k else ''))
        if 'rho_eval_over_build' in r: lines.append(f'- rho(eval/build) = {r["rho_eval_over_build"]:.3f}')
        for name in ('J_none', 'J_nominal', 'J_critical'):
            if name in r: lines.append(f'- {name}: evals {r[name]["evals"]:.2f}, cpu {r[name]["cpu_ms"]:.2f} ms (eval {r[name]["eval_ms"]:.2f} + build {r[name]["build_ms"]:.2f})')
        lines.append('')
    (out / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
