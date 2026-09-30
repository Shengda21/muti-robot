"""Generate the complete new transfer-suite seed batch on the local model server."""
import argparse
import json
from pathlib import Path
import sys
import time
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from transfer_suite import prompt, compile_candidate


def schema(task, k):
    n = len(task['goals'])
    candidate = {'type':'object','properties':{
        'robots':{'type':'array','items':{'type':'integer','enum':[0,1,2]},'minItems':n,'maxItems':n},
        'order':{'type':'array','items':{'type':'string','enum':[g['id'] for g in task['goals']]},'minItems':n,'maxItems':n},
        'opener':{'type':'integer','enum':[0,2]},'closer':{'type':'integer','enum':[0,2]}},
        'required':['robots','order','opener','closer'],'additionalProperties':False}
    return {'type':'object','properties':{'plans':{'type':'array','items':candidate,'minItems':k,'maxItems':k}},'required':['plans'],'additionalProperties':False}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--tasks',default='data/transfer_tasks.json')
    parser.add_argument('--split',default='evaluation')
    parser.add_argument('--output',default='results/major_revision/candidates')
    parser.add_argument('--model',default='qwen3-vl-8b-instruct')
    parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    directory=ROOT/args.output;directory.mkdir(parents=True,exist_ok=True)
    tasks=[t for t in json.loads((ROOT/args.tasks).read_text()) if not t.get('excluded') and t.get('split')==args.split]
    if args.limit:tasks=tasks[:args.limit]
    for task in tasks:
        path=directory/(task['id']+'.json')
        if path.exists():continue
        message=prompt(task,12)
        payload={'model':args.model,'messages':[{'role':'user','content':message}],
            'temperature':0.7,'seed':20260930+int(task['scene'].replace('FloorPlan',''))*10+len(task['goals']),
            'max_tokens':3200,'response_format':{'type':'json_schema','json_schema':{'name':'plans','strict':True,'schema':schema(task,12)}}}
        row={'task_id':task['id'],'scene':task['scene'],'model':args.model,'prompt':message,'configuration':{k:v for k,v in payload.items() if k!='messages'},'candidates':[],'invalid':[],'service_attempts':[]}
        for attempt in range(3):
            start=time.perf_counter()
            try:
                response=requests.post('http://127.0.0.1:1234/v1/chat/completions',json=payload,timeout=180)
                response.raise_for_status()
                data=response.json();row['raw_response']=data;row['elapsed_s']=time.perf_counter()-start
                break
            except (requests.RequestException,ValueError) as exc:
                row['service_attempts'].append({'attempt':attempt+1,'seconds':time.perf_counter()-start,'error_type':type(exc).__name__})
        if 'raw_response' in row:
            try:
                text=data['choices'][0]['message']['content'].split('</think>')[-1]
                proposals=json.loads(text[text.find('{'):text.rfind('}')+1])['plans']
                for i,candidate in enumerate(proposals):
                    try:
                        plan=compile_candidate(task,candidate)
                        row['candidates'].append(plan)
                    except Exception as exc:
                        row['invalid'].append({'index':i,'candidate':candidate,'error':str(exc)[:200]})
            except Exception as exc:row['parse_error']=type(exc).__name__
        else:row['service_failed']=True
        path.write_text(json.dumps(row,indent=2),encoding='utf-8')
        print(json.dumps({'task':task['id'],'valid':len(row['candidates']),'invalid':len(row['invalid']),'service_failed':row.get('service_failed',False),'seconds':row.get('elapsed_s')}),flush=True)


if __name__=='__main__':main()
