"""Independently check terminal-shift formula against the original DAG replay.

Reads frozen inputs/results only. No candidate generation, search, or score
selection is performed. This is an implementation audit, not new outcome data.
"""
from collections import Counter, defaultdict
import json
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from scheduling import schedule, reactive_pause
from robust_completion import build_profile, pause_increase
from transfer_suite import compile_candidate

SEED = 2026093001
TOLERANCE = 1e-9


def cases_for(plan, baseline, horizon, rng):
    cases = []; seen = set()
    def add(label, robot, onset, duration):
        onset, duration = max(0.0,float(onset)), max(0.0,float(duration))
        key = (robot,onset,duration)
        if key not in seen:
            seen.add(key); cases.append((label,robot,onset,duration))
    C = baseline['makespan']; eps=1e-7
    for robot in plan['robot_ids']:
        add('zero_duration',robot,C*0.37,0)
        add('at_origin',robot,0,0.3*horizon)
        add('at_completion',robot,C,0.6*horizon)
        add('after_completion',robot,C+1,0.6*horizon)
        nodes = sorted((a for a in plan['actions'] if a['robot']==robot),
                       key=lambda a:baseline['starts'][a['id']])
        # First/last actions cover both graph endpoints without testing every
        # boundary repeatedly. Random points cover intermediate intervals.
        selected = nodes[:1]+(nodes[-1:] if len(nodes)>1 else [])
        for node in selected:
            s,e = baseline['starts'][node['id']],baseline['ends'][node['id']]
            for label,point in (('action_start',s),('action_end',e)):
                for offset in (-eps,0,eps):
                    add(label+('_before' if offset<0 else '_after' if offset>0 else '_exact'),
                        robot,point+offset,0.15*horizon)
            add('busy_midpoint',robot,(s+e)/2,0.6*horizon)
        previous_end=0.0
        idle_found={'initial':False,'internal':False}
        for node in nodes:
            s,e=baseline['starts'][node['id']],baseline['ends'][node['id']]
            category='initial' if previous_end==0 else 'internal'
            if s>previous_end and not idle_found[category]:
                t=(previous_end+s)/2
                add(category+'_idle_no_overlap',robot,t,(s-t)/2)
                add(category+'_idle_boundary_touch',robot,t,s-t)
                add(category+'_idle_crossing',robot,t,s-t+0.3*horizon)
                idle_found[category]=True
            previous_end=e
        for _ in range(8):
            add('random',robot,rng.uniform(0,1.1*max(C,horizon)),
                rng.choice((0.1,0.15,0.2,0.3,0.4,0.6))*horizon)
    return cases


def main():
    tasks=[t for t in json.loads((ROOT/'data/transfer_tasks.json').read_text())
           if t['split']=='evaluation' and not t.get('excluded')]
    by_scene=defaultdict(list)
    for task in tasks:
        by_scene[task['scene']].append(task)
    recompilation_count=0; recompile_failures=[]; goal_count=0; satisfied=[]
    for task in tasks:
        objects={o['objectId']:o for o in task['objects']}
        for goal in task['goals']:
            goal_count+=1
            if goal['target'] in (objects[goal['object']].get('parentReceptacles') or []):
                satisfied.append({'task':task['id'],'job':goal['id']})
        rows=json.loads((ROOT/'results/major_revision/candidates'/(task['id']+'.json')).read_text())
        for i,saved in enumerate(rows['candidates']):
            current=compile_candidate(task,saved['candidate']);recompilation_count+=1
            if current!=saved:
                recompile_failures.append({'task':task['id'],'candidate_index':i})
    rng=random.Random(SEED); failures=[]; cases_count=0; max_error=0.0
    counts=Counter(); replay_states=Counter(); plan_records=[]; families=Counter()
    nominal_failures=[]
    scenes=sorted(by_scene,key=lambda s:int(s.replace('FloorPlan','')))
    for scene_index,scene in enumerate(scenes):
        preferred=(4,6,8)[scene_index%3]
        task=min(by_scene[scene],key=lambda t:(abs(t['n']-preferred),t['n']))
        raw=json.loads((ROOT/'results/major_revision/candidates'/(task['id']+'.json')).read_text())
        results=json.loads((ROOT/'results/major_revision/search'/(task['id']+'.json')).read_text())
        selected=results['runs']['6101']['llm_risk_none']['snapshots']['256']
        plans=[('raw_candidate_0',raw['candidates'][0],None),
               ('risk_search_6101_budget256',compile_candidate(task,selected['candidate']),selected['C'])]
        unique=set()
        for source,plan,expected_C in plans:
            identity=json.dumps(plan['candidate'],sort_keys=True)
            if identity in unique:
                continue
            unique.add(identity)
            baseline=schedule(plan);profile=build_profile(plan)
            if expected_C is not None and abs(baseline['makespan']-expected_C)>TOLERANCE:
                nominal_failures.append({'task':task['id'],'source':source,'expected':expected_C,
                                         'replayed':baseline['makespan']})
            plan_cases=cases_for(plan,baseline,results['T'],rng)
            row={'task':task['id'],'scene':scene,'family':task['room_type'],'n':task['n'],
                 'source':source,'C':baseline['makespan'],'T':results['T'],
                 'cases':len(plan_cases),'maximum_absolute_error':0.0,'mismatches':0}
            for label,robot,start,duration in plan_cases:
                # This reference path does not import or call the new score.
                replay=reactive_pause(plan,robot,start,duration)
                reference=replay['perturbed']['makespan']-replay['baseline']['makespan']
                actual=pause_increase(profile,robot,start,duration)
                error=abs(actual-reference)
                max_error=max(max_error,error);row['maximum_absolute_error']=max(row['maximum_absolute_error'],error)
                cases_count+=1;counts[label]+=1;families[task['room_type']]+=1
                if replay['fault_action'] is None:
                    replay_states['no_affected_action']+=1
                elif abs(replay['additional_duration']-duration)<=TOLERANCE:
                    replay_states['active_action_full_pause']+=1
                else:
                    replay_states['idle_partial_pause']+=1
                if error>TOLERANCE:
                    row['mismatches']+=1
                    failures.append({'task':task['id'],'source':source,'case':label,'robot':robot,
                        'start':start,'duration':duration,'formula':actual,'replay':reference,'error':error})
            plan_records.append(row)
    report={'seed':SEED,'tolerance':TOLERANCE,'reference':'src/scheduling.py::reactive_pause (original max-plus DAG rescheduling)',
        'tested':'src/robust_completion.py::pause_increase','selection_rule':'One task per evaluation scene; preferred n cycles 4,6,8 in scene order, nearest available n on a missing variant; first raw candidate and seed6101 unpruned risk-search budget256 selection; identical plan deduplicated.',
        'scenes':len(scenes),'plans':len(plan_records),'cases':cases_count,'cases_by_family':dict(families),
        'cases_by_type':dict(counts),'reference_pause_states':dict(replay_states),'maximum_absolute_error':max_error,
        'mismatch_count':len(failures),'failures':failures,'nominal_selection_mismatches':nominal_failures,
        'original_candidate_recompilation':{'tasks':len(tasks),'plans':recompilation_count,'mismatches':recompile_failures,
            'comparison':'Every field in the saved and recompiled plan, including durations, grid endpoints and dependencies.'},
        'initial_goal_check':{'tasks':len(tasks),'transfer_goals':goal_count,'already_satisfied':satisfied},
        'per_plan':plan_records,'scope':'Numerical fixed-DAG pause and cost reproducibility check. No physical timing, collision avoidance, or manipulation-success claim.'}
    out=ROOT/'results/major_revision/replay_validation.json';out.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ('scenes','plans','cases','maximum_absolute_error','mismatch_count')}))
    if cases_count<1000 or failures or nominal_failures or recompile_failures or satisfied:
        raise SystemExit('Validation failed; inspect the preserved report')


if __name__=='__main__':
    main()
