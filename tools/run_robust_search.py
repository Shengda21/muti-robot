"""Prospective transfer-suite search and exact pruning ablation.

Every numerical method receives the same compiler. Candidate audit scoring is
performed after, and excluded from, the measured complete-search interval.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from transfer_suite import compile_candidate
from robust_completion import (build_profile, continuous_score, continuous_lower_bound,
    prepare_nominal, build_profile_from_nominal)

BUDGETS=(32,64,128,256)
SEARCH_SEEDS=(6101,6102,6103)


def identity(candidate):
    return (tuple(candidate['robots']),tuple(candidate['order']),candidate['opener'],candidate['closer'])


def uniform_candidate(task,rng):
    order=[g['id'] for g in task['goals']];rng.shuffle(order)
    return {'robots':[rng.randrange(3) for _ in order],'order':order,'opener':rng.choice((0,2)),'closer':rng.choice((0,2))}


def uniform_pool(task,k,seed):
    rng=random.Random(seed);pool=[];seen=set()
    while len(pool)<k:
        c=uniform_candidate(task,rng);key=identity(c)
        if key not in seen:seen.add(key);pool.append(c)
    return pool


def mutate(task,parent,rng):
    c={'robots':list(parent['robots']),'order':list(parent['order']),'opener':parent['opener'],'closer':parent['closer']}
    kind=rng.randrange(4)
    if kind==0:
        i=rng.randrange(len(c['robots']));c['robots'][i]=rng.choice([r for r in range(3) if r!=c['robots'][i]])
    elif kind==1:
        i,j=rng.sample(range(len(c['order'])),2);c['order'][i],c['order'][j]=c['order'][j],c['order'][i]
    elif kind==2:c['opener']=2-c['opener']
    else:c['closer']=2-c['closer']
    return c


def run_search(task,initial,T,seed,objective='risk',pruning='none',uniform=False,budget=256):
    durations=[T*x for x in (.1,.2,.4)]
    rng=random.Random(seed);beam=[];seen=set();records=[];profiles=[];nominals=[];snapshots={};best=None
    initial=list(initial);pending=0;duplicates=0
    risk_evaluations=0;pruned_count=0;score_cpu_s=0.;compile_cpu_s=0.;bound_evaluations=0
    start=time.perf_counter();cpu_start=time.process_time()
    while len(records)<budget:
        if pending<len(initial):c=initial[pending];pending+=1
        elif uniform or rng.random()<.2:c=uniform_candidate(task,rng)
        else:c=mutate(task,rng.choice(beam)['candidate'],rng)
        key=identity(c)
        if key in seen:
            duplicates+=1
            if duplicates>100000:raise RuntimeError('Unique-proposal budget cannot be reached')
            continue
        seen.add(key)
        phase_start=time.process_time()
        plan=compile_candidate(task,c);nominal=prepare_nominal(plan);C=nominal.makespan
        compile_cpu_s+=time.process_time()-phase_start
        threshold=beam[-1]['J'] if len(beam)==4 else None
        # Baseline/C-only variants do not pay for the critical-path bound. Their
        # mathematical bound is filled by the independent audit after timing.
        bound=None
        if objective=='risk' and not uniform and pruning=='critical' and threshold is not None:
            bound=continuous_lower_bound(C,3,T,durations);bound_evaluations+=1
        prune=(objective=='risk' and not uniform and threshold is not None and
               ((pruning=='critical' and bound>threshold+1e-9) or
                (pruning=='nominal' and C>threshold+1e-9)))
        profile=None;J=None
        if prune:pruned_count+=1
        else:
            phase_start=time.process_time()
            profile=build_profile_from_nominal(nominal)
            J=continuous_score(profile,T,durations)
            score_cpu_s+=time.process_time()-phase_start;risk_evaluations+=1
        row={'index':len(records),'candidate':c,'C':C,'J':J,'bound':bound,'pruned':prune,'threshold':threshold}
        records.append(row);profiles.append(profile);nominals.append(nominal)
        if not prune:
            if best is None or (J,C,row['index'])<(best['J'],best['C'],best['index']):best=row
            beam.append(row)
            beam.sort(key=lambda r:(r['C'],r['index']) if objective=='nominal' else (r['J'],r['C'],r['index']))
            beam=beam[:4]
        row['beam_indices']=[r['index'] for r in beam]
        if len(records) in BUDGETS or len(records)==len(initial):
            snapshots[str(len(records))]={'best_index':best['index'],'C':best['C'],'J':best['J'],
                'elapsed_s':time.perf_counter()-start,'cpu_elapsed_s':time.process_time()-cpu_start,
                'risk_evaluations':risk_evaluations,'pruned':pruned_count,
                'score_cpu_s':score_cpu_s,'compile_cpu_s':compile_cpu_s}
    elapsed=time.perf_counter()-start;cpu_elapsed=time.process_time()-cpu_start
    # Independently check all skipped exact scores after timing stops.
    audited=0
    for i,(row,profile,nominal) in enumerate(zip(records,profiles,nominals)):
        if row['bound'] is None:row['bound']=continuous_lower_bound(row['C'],3,T,durations)
        if row['pruned']:
            profile=build_profile_from_nominal(nominal);profiles[i]=profile
            value=continuous_score(profile,T,durations)
            assert value>row['threshold']+1e-9, (task['id'],row,value)
            row['audit_J']=value;audited+=1
        else:assert row['J']>=row['bound']-1e-8
    test_durations=[T*x for x in (.15,.3,.6)]
    for value in snapshots.values():
        profile=profiles[value['best_index']]
        value['test_J']=continuous_score(profile,T,test_durations)
        value['candidate']=records[value['best_index']]['candidate']
    return {'objective':objective,'pruning':pruning,'uniform':uniform,'seed':seed,'T':T,
        'proposals':len(records),'duplicates':duplicates,'risk_evaluations':len(records)-audited,
        'pruned':audited,'elapsed_s':elapsed,'cpu_elapsed_s':cpu_elapsed,
        'score_cpu_s':score_cpu_s,'compile_cpu_s':compile_cpu_s,
        'bound_evaluations':bound_evaluations,'snapshots':snapshots,'records':records}


def run_task(payload):
    task,candidate_dir,output_dir=payload
    output=Path(output_dir)/(task['id']+'.json')
    if output.exists():return {'task':task['id'],'resumed':True}
    source=Path(candidate_dir)/(task['id']+'.json')
    if not source.exists():return {'task':task['id'],'pending_candidates':True}
    llm=json.loads(source.read_text());initial=[];seen=set()
    for plan in llm.get('candidates',[]):
        c=plan['candidate'];key=identity(c)
        if key not in seen:seen.add(key);initial.append(c)
    random_initial=uniform_pool(task,12,9000+int(task['scene'].replace('FloorPlan',''))*10+len(task['goals']))
    combined=initial+random_initial
    profiles=[build_profile(compile_candidate(task,c)) for c in combined]
    T=min(p.makespan for p in profiles)
    pools={}
    for name,pool in [('llm',initial),('uniform',random_initial)]:
        if not pool:continue
        rows=[]
        for i,c in enumerate(pool):
            profile=build_profile(compile_candidate(task,c))
            rows.append({'candidate':c,'index':i,'C':profile.makespan,'J':continuous_score(profile,T,[T*x for x in (.1,.2,.4)]),'test_J':continuous_score(profile,T,[T*x for x in (.15,.3,.6)])})
        pools[name]={'size':len(rows),'nominal':min(rows,key=lambda r:(r['C'],r['index'])),'risk':min(rows,key=lambda r:(r['J'],r['C'],r['index']))}
    runs={}
    for seed in SEARCH_SEEDS:
        runs[str(seed)]={}
        for name,pool in [('llm',initial),('uniform',random_initial)]:
            if not pool:continue
            methods=[('nominal','none'),('risk','none')]
            if name=='llm':methods += [('risk','nominal'),('risk','critical')]
            for objective,pruning in methods:
                method=f'{name}_{objective}_{pruning}'
                runs[str(seed)][method]=run_search(task,pool,T,seed,objective,pruning)
            if name=='llm':
                base=runs[str(seed)]['llm_risk_none']
                for pruning in ('nominal','critical'):
                    variant=runs[str(seed)][f'llm_risk_{pruning}']
                    assert [identity(r['candidate']) for r in base['records']]==[identity(r['candidate']) for r in variant['records']]
                    assert [r['beam_indices'] for r in base['records']]==[r['beam_indices'] for r in variant['records']]
                    for budget in BUDGETS:
                        assert base['snapshots'][str(budget)]['best_index']==variant['snapshots'][str(budget)]['best_index']
                        assert abs(base['snapshots'][str(budget)]['J']-variant['snapshots'][str(budget)]['J'])<1e-10
        runs[str(seed)]['uniform_random']=run_search(task,random_initial,T,seed,uniform=True)
    record={'task':task['id'],'scene':task['scene'],'family':'living' if int(task['scene'].replace('FloorPlan',''))<300 else 'bedroom',
        'n':len(task['goals']),'split':task['split'],'T':T,'pools':pools,'llm_seconds':llm.get('elapsed_s'),'runs':runs,
        'pruning_trace_matches':True,'pruned_scores_independently_checked':True}
    output.write_text(json.dumps(record,separators=(',',':')),encoding='utf-8')
    return {'task':task['id'],'T':T,'llm_candidates':len(initial),'pruning_checks':'passed'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--tasks',default='data/transfer_tasks.json')
    parser.add_argument('--split',default='evaluation')
    parser.add_argument('--candidates',default='results/major_revision/candidates')
    parser.add_argument('--output',default='results/major_revision/search')
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    tasks=[t for t in json.loads((ROOT/args.tasks).read_text()) if not t.get('excluded') and t.get('split')==args.split]
    if args.limit:tasks=tasks[:args.limit]
    output=ROOT/args.output;output.mkdir(parents=True,exist_ok=True)
    payloads=[(t,str(ROOT/args.candidates),str(output)) for t in tasks]
    if args.workers==1:
        for payload in payloads:print(json.dumps(run_task(payload)),flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for result in executor.map(run_task,payloads):print(json.dumps(result),flush=True)


if __name__=='__main__':main()
