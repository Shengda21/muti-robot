"""Independent cross-room transfer tasks with static reachable-grid costs.

The cost unit is one 0.25 m grid translation or one manipulation. It is not
seconds and does not account for turns, dynamic collisions, or placement capacity.
LLMs supply symbolic allocations; all geometric costs come from scene metadata.
"""
from __future__ import annotations
from collections import deque
from functools import lru_cache
import json
import math

GRID = 0.25
CONTAINER_PRIORITY = ('Drawer', 'Cabinet', 'Safe', 'Box')
SURFACE_PRIORITY = ('Desk', 'CoffeeTable', 'SideTable', 'DiningTable',
                    'CounterTop', 'Dresser', 'TVStand', 'Shelf', 'Bed')
# Exclude objects that are themselves receptacles and large furniture-like items.
# Eligibility uses simulator metadata only, never a method's success or score.
MAX_OBJECT_EXTENT = 0.50
_CONTEXTS = {}


def key(p):
    return (round(p['x'], 4), round(p['z'], 4))


def grid_graph(positions):
    cells = {key(p) for p in positions}
    return {p: tuple(q for dx, dz in ((GRID, 0), (0, GRID), (-GRID, 0), (0, -GRID))
                     if (q := (round(p[0]+dx, 4), round(p[1]+dz, 4))) in cells)
            for p in cells}


def shortest_path(graph, source, target):
    queue = deque([source]); parent = {source: None}
    while queue:
        p = queue.popleft()
        if p == target:
            out = []
            while p is not None:
                out.append(p); p = parent[p]
            return out[::-1]
        for q in graph[p]:
            if q not in parent:
                parent[q] = p; queue.append(q)
    raise ValueError('disconnected grid endpoints')


@lru_cache(maxsize=128)
def _grid_cached(coordinates):
    positions = [{'x': p[0], 'z': p[1]} for p in coordinates]
    graph = grid_graph(positions)
    # Repeated compilation reuses all-pairs unit-edge distances for the scene.
    distances = {}
    for source in sorted(graph):
        queue = deque([source]); dist = {source: 0}
        while queue:
            p = queue.popleft()
            for q in graph[p]:
                if q not in dist:
                    dist[q] = dist[p]+1; queue.append(q)
        distances[source] = dist
    return graph, distances


def _closest(position, cells):
    return min(cells, key=lambda p: ((p[0]-position['x'])**2 +
                                   (p[1]-position['z'])**2, p))


def prepare_context(task):
    """Cache geometry for a task that remains immutable during a search.

    Keep the task reference so Python cannot recycle its identity into an
    unrelated context. Returned mappings must be treated as read-only.
    """
    cache_key = id(task)
    if cache_key not in _CONTEXTS:
        objects = {o['objectId']:o for o in task['objects']}
        coordinates = tuple(sorted(key(p) for p in task['positions']))
        graph, distances = _grid_cached(coordinates)
        starts = tuple(_closest(a['position'],graph) for a in task['initial_agents'])
        endpoints = {oid:_closest(o['position'],graph) for oid,o in objects.items()}
        _CONTEXTS[cache_key] = (task,(objects,graph,distances,starts,endpoints))
    return _CONTEXTS[cache_key][1]


def _volume(obj):
    return math.prod(obj.get('axisAlignedBoundingBox', {}).get('size', {}).values())


def _choose_target(objects, types, excluded=()):
    for kind in types:
        candidates = [o for o in objects if o['objectType'] == kind and
                      o.get('receptacle') and o['objectId'] not in excluded]
        if candidates:
            return min(candidates, key=lambda o: (-_volume(o), o['objectId']))
    return None


def build_tasks(inventory):
    tasks = []
    for scene in inventory:
        sid = int(scene['scene'].replace('FloorPlan', '').replace('_physics', ''))
        split = 'development' if sid in (201,202,203,301,302,303) else 'evaluation'
        objects = scene['objects']
        container = _choose_target(objects, CONTAINER_PRIORITY)
        surface = _choose_target(objects, SURFACE_PRIORITY,
                                 (container['objectId'],) if container else ())
        if container and surface:
            destinations = [container, surface]
        elif surface:
            second = _choose_target(objects, SURFACE_PRIORITY, (surface['objectId'],))
            destinations = [surface, second] if second else []
        else:
            destinations = []
        selected = []
        if destinations:
            used = set()
            candidates = sorted((o for o in objects if o.get('pickupable') and
                not o.get('receptacle') and
                max(o.get('axisAlignedBoundingBox', {}).get('size', {'x':float('inf')}).values()) <= MAX_OBJECT_EXTENT),
                key=lambda o: (o['objectType'], o['objectId']))
            # Balanced fixed destinations. Skip objects already satisfying that goal.
            for i in range(8):
                destination = destinations[i % 2]
                eligible = [o for o in candidates if o['objectId'] not in used and
                            destination['objectId'] not in (o.get('parentReceptacles') or [])]
                if not eligible:
                    break
                obj = eligible[0]; used.add(obj['objectId'])
                selected.append({'id':f'j{i}', 'object_type':obj['objectType'],
                    'object':obj['objectId'], 'target':destination['objectId'],
                    'target_type':destination['objectType']})
        for n in (4,6,8):
            task = {'id':f't{sid}_n{n}', 'scene':scene['scene'], 'split':split,
                    'n':n, 'room_type':'living' if sid < 300 else 'bedroom'}
            if len(selected) < n:
                task['excluded'] = ('fewer than two eligible destinations' if not destinations
                                    else f'only {len(selected)} eligible distinct objects')
                tasks.append(task); continue
            doors = [o['objectId'] for o in destinations if o.get('openable')]
            goals = selected[:n]
            task.update({'goals':goals, 'objects':objects, 'initial_agents':scene['agents'],
                'positions':scene['positions'], 'destinations':[o['objectId'] for o in destinations],
                'openable_receptacles':doors, 'cost_model':'static-grid-0.25m-plus-manipulation-v1',
                'robots':[{'id':r, 'skills':['transfer','open','close'] if r != 1 else ['transfer']}
                          for r in range(3)],
                'instruction':'Move ' + '; '.join(g['object_type']+' to '+g['target_type'] for g in goals)
                              + (', and leave the destination containers closed.' if doors else '.')})
            tasks.append(task)
    return tasks


def prompt(task, k=12):
    jobs = [{'id':g['id'],'object':g['object_type'],'destination':g['target_type']} for g in task['goals']]
    return ('Allocate these household transfers to three robots. Return JSON only.\n'
        'Instruction: '+task['instruction']+'\nJobs: '+json.dumps(jobs,separators=(',',':'))+'\n'
        'Robots 0,1,2 can each carry one object. Robots 0 and 2 can open/close containers. '
        'Each job is indivisible: one robot picks and places. A compiler adds required container actions. '
        'Provide '+str(k)+' distinct allocations and job priority orders. Return {"plans":[...]} with each '
        'plan containing "robots" (one robot ID for every job, indexed by job ID), '
        '"order" (every job ID exactly once), "opener" (0 or 2), "closer" (0 or 2). '
        'Vary both assignments and order. No explanation.')


def compile_candidate(task, candidate):
    goals = {g['id']:g for g in task['goals']}; order = candidate['order']
    assignment = candidate['robots']; opener = candidate['opener']; closer = candidate['closer']
    robot_ids = [r['id'] for r in task['robots']]
    openers = [r['id'] for r in task['robots'] if 'open' in r['skills']]
    if len(order) != len(goals) or set(order) != set(goals):
        raise ValueError('order is not a permutation')
    if len(assignment) != len(goals) or any(type(r) is not int or r not in robot_ids for r in assignment):
        raise ValueError('invalid assignment')
    if type(opener) is not int or type(closer) is not int or opener not in openers or closer not in openers:
        raise ValueError('door skill ineligible')
    objects, graph, distances, starts, endpoints = prepare_context(task)
    poses = list(starts)
    actions = []
    def add(aid, robot, skill, target, deps, extra=None):
        destination = endpoints[target]
        steps = distances[poses[robot]].get(destination)
        if steps is None:
            raise ValueError('disconnected grid endpoints')
        action = {'id':aid,'robot':robot,'skill':skill,'target':target,'objects':[target],
            'duration':float(steps+1),'deps':list(dict.fromkeys(deps)),
            'navigation_steps':steps,'grid_start':list(poses[robot]),'grid_end':list(destination)}
        poses[robot] = destination
        if extra:
            action.update(extra)
        actions.append(action)
    parents = {g['id']:[p for p in objects[g['object']].get('parentReceptacles') or []
                        if p in objects and objects[p].get('openable')] for g in goals.values()}
    required = set(task['openable_receptacles']) | {p for ps in parents.values() for p in ps}
    closed = {p for p in required if not objects[p].get('isOpen')}
    opening = {p:f'open_{i}' for i,p in enumerate(sorted(closed))}
    open_order = []
    while len(open_order) < len(closed):
        ready = sorted(p for p in closed if p not in open_order and
            all(q in open_order for q in objects[p].get('parentReceptacles') or [] if q in closed))
        if not ready:
            raise ValueError('cyclic container containment')
        open_order.extend(ready)
    for p in open_order:
        # Parent containers open before nested children where metadata gives containment.
        deps = [opening[q] for q in objects[p].get('parentReceptacles') or [] if q in opening]
        add(opening[p],opener,'OpenObject',p,deps)
    touches = {p:[] for p in required}; last_receptacle = {}
    for jid in order:
        g = goals[jid]; robot = assignment[int(jid[1:])]
        add(jid+'_pick',robot,'PickupObject',g['object'],
            [opening[p] for p in parents[jid] if p in opening], {'job':jid})
        for p in parents[jid]:
            touches[p].append(jid+'_pick')
        deps = [jid+'_pick']
        if g['target'] in opening:
            deps.append(opening[g['target']])
        if g['target'] in last_receptacle:
            deps.append(last_receptacle[g['target']])
        add(jid+'_put',robot,'PutObject',g['target'],deps,{'job':jid,
            'carried_object':g['object'],'objects':[g['target'],g['object']]})
        if g['target'] in touches:
            touches[g['target']].append(jid+'_put')
        last_receptacle[g['target']] = jid+'_put'
    for i,p in enumerate(sorted(task['openable_receptacles'])):
        add(f'close_{i}',closer,'CloseObject',p,touches[p])
    return {'actions':actions,'candidate':candidate,'task_id':task['id'],'robot_ids':robot_ids,
            'cost_model':task['cost_model']}
