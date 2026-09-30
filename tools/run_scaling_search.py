"""Closure experiments: team size m, pause-length scale, and evaluator cost.

Generalises tools/run_robust_search.py without changing it. Every screening
variant is asserted to reproduce the unscreened trajectory exactly. Uniform
initial candidates only (no LLM), so no model service is needed.
"""
from __future__ import annotations
import argparse, json, random, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from transfer_suite import compile_candidate
from robust_completion import (build_profile, continuous_score, continuous_lower_bound,
    prepare_nominal, build_profile_from_nominal, score_profile, lower_bound)

BUDGET = 256
BASE_FRACS = (.1, .2, .4)


def durations_for(D, scale):
    if D == 3:
        fr = BASE_FRACS
    else:
        fr = [.05 + (.41666667 - .05) * i / (D - 1) for i in range(D)]
    return [scale * x for x in fr]


def make_task(base, m, start='far'):
    """Copy of a base task with m robots; robot 1 cannot open/close.
    Extra robots start at the farthest reachable cell (start='far') or at a seeded random cell."""
    agents = [dict(a) for a in base['initial_agents'][:m]]
    pts = sorted((p['x'], p['z']) for p in base['positions'])
    rnd = random.Random(f"{base['id']}|{m}|start")
    while len(agents) < m:
        cur = [(a['position']['x'], a['position']['z']) for a in agents]
        far = (max(pts, key=lambda q: min((q[0]-c[0])**2 + (q[1]-c[1])**2 for c in cur))
               if start == 'far' else rnd.choice(pts))
        agents.append({'position': {'x': far[0], 'y': 0.9, 'z': far[1]}})
    t = dict(base)
    t['initial_agents'] = agents
    t['robots'] = [{'id': r, 'skills': ['transfer', 'open', 'close'] if r != 1 else ['transfer']}
                   for r in range(m)]
    return t


def identity(c):
    return (tuple(c['robots']), tuple(c['order']), c['opener'], c['closer'])


def uniform_candidate(task, rng):
    m = len(task['robots']); openers = [r['id'] for r in task['robots'] if 'open' in r['skills']]
    order = [g['id'] for g in task['goals']]; rng.shuffle(order)
    return {'robots': [rng.randrange(m) for _ in order], 'order': order,
            'opener': rng.choice(openers), 'closer': rng.choice(openers)}


def uniform_pool(task, k, seed):
    rng = random.Random(seed); pool = []; seen = set()
    while len(pool) < k:
        c = uniform_candidate(task, rng); key = identity(c)
        if key not in seen: seen.add(key); pool.append(c)
    return pool


def other_opener(openers, current, rng):
    """Two eligible openers flip without drawing (matches the m=3 original); otherwise sample another."""
    rest = [o for o in openers if o != current]
    if not rest: return current
    return rest[0] if len(rest) == 1 else rng.choice(rest)


def mutate(task, parent, rng):
    m = len(task['robots']); openers = [r['id'] for r in task['robots'] if 'open' in r['skills']]
    c = {'robots': list(parent['robots']), 'order': list(parent['order']),
         'opener': parent['opener'], 'closer': parent['closer']}
    kind = rng.randrange(4)
    if kind == 0:
        i = rng.randrange(len(c['robots'])); c['robots'][i] = rng.choice([r for r in range(m) if r != c['robots'][i]])
    elif kind == 1:
        i, j = rng.sample(range(len(c['order'])), 2); c['order'][i], c['order'][j] = c['order'][j], c['order'][i]
    elif kind == 2:
        c['opener'] = other_opener(openers, c['opener'], rng)
    else:
        c['closer'] = other_opener(openers, c['closer'], rng)
    return c


def run_search(task, initial, T, seed, dur, evaluator, objective='risk', pruning='none', uniform=False):
    m = len(task['robots'])
    if evaluator[0] == 'closed':
        score = lambda pr: continuous_score(pr, T, dur)
        bound_of = lambda C: continuous_lower_bound(C, m, T, dur)
    else:
        P = evaluator[1]; fr = [(i + .5) / P for i in range(P)]
        score = lambda pr: score_profile(pr, T, fr, dur)
        bound_of = lambda C: lower_bound(C, m, T, fr, dur)
    rng = random.Random(seed); beam = []; seen = set(); records = []; profiles = []; nominals = []
    best = None; pending = 0; evals = 0; pruned = 0; score_cpu = compile_cpu = 0.; dup = 0
    cpu0 = time.process_time()
    while len(records) < BUDGET:
        if pending < len(initial): c = initial[pending]; pending += 1
        elif uniform or rng.random() < .2: c = uniform_candidate(task, rng)
        else: c = mutate(task, rng.choice(beam)['candidate'], rng)
        key = identity(c)
        if key in seen:
            dup += 1
            if dup > 100000: raise RuntimeError('cannot reach unique-proposal budget')
            continue
        seen.add(key)
        t0 = time.process_time()
        nominal = prepare_nominal(compile_candidate(task, c)); C = nominal.makespan
        compile_cpu += time.process_time() - t0
        thr = beam[-1]['J'] if len(beam) == 4 else None
        bound = None
        if objective == 'risk' and not uniform and pruning == 'critical' and thr is not None:
            bound = bound_of(C)
        prune = (objective == 'risk' and not uniform and thr is not None and
                 ((pruning == 'critical' and bound > thr + 1e-9) or (pruning == 'nominal' and C > thr + 1e-9)))
        profile = None; J = None
        if prune: pruned += 1
        else:
            t0 = time.process_time(); profile = build_profile_from_nominal(nominal); J = score(profile)
            score_cpu += time.process_time() - t0; evals += 1
        row = {'index': len(records), 'candidate': c, 'C': C, 'J': J, 'pruned': prune, 'threshold': thr}
        records.append(row); profiles.append(profile); nominals.append(nominal)
        if not prune:
            if best is None or (J, C, row['index']) < (best['J'], best['C'], best['index']): best = row
            beam.append(row)
            if objective == 'nominal': beam.sort(key=lambda r: (r['C'], r['index']))
            elif objective == 'nominal_tb': beam.sort(key=lambda r: (r['C'], r['J'], r['index']))
            else: beam.sort(key=lambda r: (r['J'], r['C'], r['index']))
            beam = beam[:4]
        row['beam'] = [r['index'] for r in beam]
    cpu = time.process_time() - cpu0
    for i, (row, nom) in enumerate(zip(records, nominals)):   # audit after timing stops
        if row['pruned']:
            v = score(build_profile_from_nominal(nom)); assert v > row['threshold'] + 1e-9, (task['id'], row, v)
    pr = profiles[best['index']]
    test = continuous_score(pr, T, [1.5 * d for d in dur])
    return {'evals': evals, 'pruned': pruned, 'cpu_s': cpu, 'score_cpu_s': score_cpu, 'compile_cpu_s': compile_cpu,
            'C': best['C'], 'J': best['J'], 'test_J': test, 'best_index': best['index'],
            'trace': [(identity(r['candidate']), tuple(r['beam'])) for r in records]}


def run_task(payload):
    base, cfg, outdir = payload
    out = Path(outdir) / (base['id'] + '.json')
    if out.exists(): return {'task': base['id'], 'resumed': True}
    sid = int(base['scene'].replace('FloorPlan', ''))
    result = {'task': base['id'], 'scene': base['scene'], 'n': base['n'], 'cells': {}}
    for m in cfg['m']:
        task = make_task(base, m, cfg.get('start', 'far'))
        pool = uniform_pool(task, 12, 9000 + sid * 10 + base['n'] + 100000 * m)
        T = min(build_profile(compile_candidate(task, c)).makespan for c in pool)
        T_ref = T
        if cfg.get('pause_ref', 'own') == 'm3':   # absolute pause lengths fixed at the m=3 horizon
            t3 = make_task(base, 3); p3 = uniform_pool(t3, 12, 9000 + sid * 10 + base['n'] + 300000)
            T_ref = min(build_profile(compile_candidate(t3, c)).makespan for c in p3)
        for ev in cfg['evaluators']:
            D = ev[-1]
            for scale in cfg['scale']:
                dur = [T_ref * x for x in durations_for(D, scale)]
                cell = {}
                for seed in cfg['seeds']:
                    r = {}
                    order = list(cfg['methods']); random.Random(f"{base['id']}|{m}|{seed}|order").shuffle(order)
                    for name in order:
                        obj, prn, uni = cfg['methods'][name]
                        r[name] = run_search(task, pool, T, seed, dur, ev, obj, prn, uni)
                    base_trace = r['J_none']['trace']
                    for prn in ('J_nominal', 'J_critical'):
                        if prn in r:
                            assert r[prn]['trace'] == base_trace, (base['id'], m, scale, seed, prn)
                            assert r[prn]['best_index'] == r['J_none']['best_index']
                    for v in r.values(): del v['trace']
                    cell[str(seed)] = r
                key = f"m{m}|{ev[0]}{ev[1] if ev[0] == 'grid' else ''}|D{D}|s{scale}" + ('' if cfg.get('pause_ref', 'own') == 'own' else '|abs') + ('' if cfg.get('start', 'far') == 'far' else '|rndstart')
                result['cells'][key] = {'T': T, 'runs': cell}
    out.write_text(json.dumps(result, separators=(',', ':')), encoding='utf-8')
    return {'task': base['id'], 'ok': True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['scaling', 'cost', 'check'], required=True)
    ap.add_argument('--tasks', default='data/transfer_tasks.json')
    ap.add_argument('--split', default='evaluation')
    ap.add_argument('--output', required=True)
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--limit', type=int)
    ap.add_argument('--m-list', default=None)
    ap.add_argument('--scale-list', default=None)
    ap.add_argument('--pause-ref', choices=['own', 'm3'], default='own')
    ap.add_argument('--start', choices=['far', 'random'], default='far')
    a = ap.parse_args()
    seeds = list(range(6101, 6101 + a.seeds))
    full = {'J_none': ('risk', 'none', False), 'J_nominal': ('risk', 'nominal', False),
            'J_critical': ('risk', 'critical', False), 'C_rerank': ('nominal', 'none', False),
            'C_rerank_tb': ('nominal_tb', 'none', False), 'uniform': ('risk', 'none', True)}
    screen = {k: full[k] for k in ('J_none', 'J_nominal', 'J_critical')}
    if a.mode == 'scaling':
        cfg = {'m': [2, 3, 4, 6], 'scale': [.5, 1., 2., 4.], 'evaluators': [('closed', 3)],
               'seeds': seeds, 'methods': full}
    elif a.mode == 'cost':
        cfg = {'m': [3], 'scale': [1.], 'evaluators': [('closed', 3), ('closed', 12), ('grid', 32, 12)],
               'seeds': seeds, 'methods': screen}
    else:
        cfg = {'m': [3], 'scale': [1.], 'evaluators': [('closed', 3)],
               'seeds': seeds[:1], 'methods': full}
    if a.m_list: cfg['m'] = [int(x) for x in a.m_list.split(',')]
    if a.scale_list: cfg['scale'] = [float(x) for x in a.scale_list.split(',')]
    cfg['pause_ref'] = a.pause_ref; cfg['start'] = a.start
    tasks = [t for t in json.loads((ROOT / a.tasks).read_text()) if not t.get('excluded') and t.get('split') == a.split]
    if a.limit: tasks = tasks[:a.limit]
    out = ROOT / a.output; out.mkdir(parents=True, exist_ok=True)
    (out / 'config.json').write_text(json.dumps({k: v for k, v in cfg.items() if k != 'methods'} | {'mode': a.mode, 'methods': list(cfg['methods'])}, default=list))
    payloads = [(t, cfg, str(out)) for t in tasks]
    if a.workers == 1:
        for p in payloads: print(json.dumps(run_task(p)), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for r in ex.map(run_task, payloads): print(json.dumps(r), flush=True)


if __name__ == '__main__':
    main()
