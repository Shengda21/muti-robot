"""Collect a fresh cross-room inventory and native movement-step calibration.

This records geometry and native primitive outcomes, not planner performance.
"""
import argparse
import json
import math
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from thor_adapter import ThorAdapter
from transfer_suite import key, grid_graph, shortest_path, build_tasks


def native_calibration(adapter):
    positions = adapter.reachable_positions
    graph = grid_graph(positions)
    start = key(adapter.metadata(0)['agent']['position'])
    if start not in graph:
        return {'status':'unavailable','reason':'initial pose outside static grid'}
    other = [adapter.metadata(i)['agent']['position'] for i in (1,2)]
    allowed = {p for p in graph if all(math.hypot(p[0]-q['x'],p[1]-q['z']) >= 0.55 for q in other)}
    allowed.add(start)
    safe_graph = {p:tuple(q for q in graph[p] if q in allowed) for p in allowed}
    # Pick one fixed longest available route, capped at twelve translations.
    routes = []
    for target in sorted(safe_graph):
        try:
            routes.append(shortest_path(safe_graph,start,target))
        except ValueError:
            pass
    route = min(routes,key=lambda p:(-len(p),p))[:13]
    begin = len(adapter.trace); moves = 0; turns = 0; max_error = 0.0
    failures = []
    for source,target in zip(route,route[1:]):
        desired = int(round(math.degrees(math.atan2(target[0]-source[0],target[1]-source[1])))) % 360
        current = int(round(adapter.metadata(0)['agent']['rotation']['y'])) % 360
        if current % 90:
            failures.append('initial rotation is not cardinal'); break
        for _ in range(((desired-current)%360)//90):
            result = adapter.step('RotateRight',0,degrees=90)
            turns += 1
            if not result.success:
                failures.append(result.error); break
        if failures:
            break
        result = adapter.step('MoveAhead',0,moveMagnitude=0.25)
        if not result.success:
            failures.append(result.error); break
        moves += 1
        actual = key(adapter.metadata(0)['agent']['position'])
        error = math.hypot(actual[0]-target[0],actual[1]-target[1]); max_error = max(max_error,error)
        if error > 0.02:
            failures.append('successful movement deviated from planned cell'); break
    return {'status':'ok' if not failures else 'failed','planned_grid_steps':len(route)-1,
        'executed_translations':moves,'executed_quarter_turns':turns,'maximum_pose_error_m':max_error,
        'route':[list(p) for p in route],'failures':failures,'trace':adapter.trace[begin:],
        'interpretation':'native static-grid locomotion check; no wall-time calibration, dynamic collision guarantee, or manipulation validation'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ranges',default='201:218,301:318')
    parser.add_argument('--output',default='data/transfer_inventory.json')
    parser.add_argument('--tasks',default='data/transfer_tasks.json')
    parser.add_argument('--calibration',default='results/transfer_scene_calibration.json')
    args = parser.parse_args()
    output = ROOT/args.output; task_output=ROOT/args.tasks; cal_output=ROOT/args.calibration
    for p in (output,task_output,cal_output):
        p.parent.mkdir(parents=True,exist_ok=True)
    inventory = json.loads(output.read_text()) if output.exists() else []
    calibrations = json.loads(cal_output.read_text()) if cal_output.exists() else []
    seen = {x['scene'] for x in inventory}
    for part in args.ranges.split(','):
        start,end = map(int,part.split(':'))
        for sid in range(start,end+1):
            scene=f'FloorPlan{sid}'
            if scene in seen:
                continue
            started=time.perf_counter()
            with ThorAdapter(scene,agent_count=3,seed=20260929,baseline_force=False) as adapter:
                record={'scene':scene,'objects':adapter.objects(),'positions':adapter.reachable_positions,
                    'agents':[adapter.metadata(i)['agent'] for i in range(3)],'collection_seed':20260929,
                    'grid_size':0.25,'simulator':'AI2-THOR 5.0.0'}
                calibration={'scene':scene,**native_calibration(adapter)}
            inventory.append(record); calibrations.append(calibration)
            output.write_text(json.dumps(inventory,indent=2)); cal_output.write_text(json.dumps(calibrations,indent=2))
            tasks=build_tasks(inventory); task_output.write_text(json.dumps(tasks,indent=2))
            print(json.dumps({'scene':scene,'seconds':round(time.perf_counter()-started,2),
                'positions':len(record['positions']),'calibration':calibration['status'],
                'task_exclusions':[t['id']+':'+t['excluded'] for t in tasks if t['scene']==scene and 'excluded' in t]}),flush=True)


if __name__=='__main__':
    main()
