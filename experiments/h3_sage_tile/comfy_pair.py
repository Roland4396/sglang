"""Node-local full Comfy H3 benchmark; fixed synthetic prompt, never reads scripts.

Run through the GPU pool Ansible inventory, after checking active requests.
Outputs real progress events and durable operation IDs for artifact retrieval.
"""
import asyncio,json,uuid,time,os
from pathlib import Path
import aiohttp
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--label',required=True)
parser.add_argument('--output-dir',type=Path,required=True)
parser.add_argument('--expected-tile',default='')
args=parser.parse_args()
DURATION=15.0
ROOT=args.output_dir
config=json.loads(Path('/home/xql/.config/gpu-runtime/node.json').read_text())
env=config['workloads']['h3']['env']
assert env.get('SGLANG_H3_SAGE_CANARY_TILE','')==args.expected_tile, 'Wrong canary configuration'
assert env['H3_TP_SIZE']=='2' and env['H3_ULYSSES_DEGREE']=='2', 'Wrong 4-GPU layout'
for key, value in {'H3_TRANSFORMER_QUANTIZATION':'fp8','H3_ACCELERATION_PROFILE':'rh_lightning',
                   'H3_ENABLE_TORCH_COMPILE':'0','H3_ATTENTION_BACKEND':'sage_attn',
                   'H3_SHARED_ATTENTION_BACKEND':'fa'}.items():
 assert env.get(key)==value, (key,env.get(key),'benchmark recipe changed')
print('CONFIG',json.dumps({k:env.get(k) for k in ('H3_SOURCE_REVISION','H3_TP_SIZE','H3_ULYSSES_DEGREE','H3_ACCELERATION_PROFILE','H3_ENABLE_TORCH_COMPILE','SGLANG_H3_SAGE_CANARY_TILE')}),flush=True)
ROOT.mkdir(parents=True,exist_ok=True)
PROMPT='integrated_multimodal_description: A continuous eye-level medium shot of a red toy robot walking slowly on a wooden desk beside a small green potted plant. Soft daylight, sharp clean shapes, consistent colors, gentle camera push-in. overall_soundscape: Quiet footsteps and soft room ambience. non_diegetic_music: None.'
async def main():
 started=time.time();client=str(uuid.uuid4())
 async with aiohttp.ClientSession(connector=aiohttp.UnixConnector(path='/home/xql/.local/state/gpu-runtime/agent.sock'),timeout=aiohttp.ClientTimeout(total=180)) as s:
  async with s.post('http://localhost/workloads/graph/ensure',json={'ticket':'h3-progress-test-'+JOB}) as r:
   data=await r.json(); print('GRAPH_ENSURE',r.status,json.dumps(data),flush=True); assert r.status==200
 try:
  async with aiohttp.ClientSession(connector=aiohttp.UnixConnector(path='/home/xql/.local/state/gpu-runtime/graph.sock'),timeout=aiohttp.ClientTimeout(total=3600)) as s:
   async with s.ws_connect('http://localhost/ws?clientId='+client,heartbeat=20) as ws:
    graph={
     '1':{'class_type':'MiniMaxH3FL2VALoader','inputs':{}},
     '3':{'class_type':'MiniMaxH3FL2VACondition','inputs':{'prompt':PROMPT,'aspect_ratio':'16:9','duration_seconds':DURATION}},
     '4':{'class_type':'MiniMaxH3FL2VASampler','inputs':{'model':['1',0],'conditioning':['3',0],'steps':50,'quality':'lossless','seed':20260916,'_h3_operation_id':JOB}},
     '5':{'class_type':'PreviewVideo','inputs':{'video':['4',0]}}}
    async with s.post('http://localhost/prompt',json={'prompt':graph,'client_id':client}) as r:
     result=await r.json(); print('SUBMIT',r.status,json.dumps(result),flush=True); assert r.status==200,result
    pid=result['prompt_id']; events=[]
    async for message in ws:
     if message.type!=aiohttp.WSMsgType.TEXT: continue
     ev=json.loads(message.data)
     # Ignore other clients' jobs, including errors that may contain their inputs.
     if ev.get('data',{}).get('prompt_id')!=pid: continue
     if ev.get('type') in ('progress','h3_progress','execution_error','execution_success','executed','execution_cached'):
      # These selected events carry counts/paths, never the novel/user history.
      events.append({'time':time.time(),**ev}); print(json.dumps(events[-1]),flush=True)
     if ev.get('type')=='execution_error': raise RuntimeError(ev)
     if ev.get('type')=='execution_success' and ev['data'].get('prompt_id')==pid: break
    async with s.get('http://localhost/history/'+pid) as r: history=await r.json()
    hist=history.get(pid,{}); result={'job_id':JOB,'prompt_id':pid,'seconds_requested':DURATION,'elapsed_seconds':time.time()-started,'events':events,'status':hist.get('status'),'outputs':hist.get('outputs')}
    (ROOT/(JOB+'.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2))
    assert hist.get('status',{}).get('status_str')=='success',result
    progress=[e['data'] for e in events if e.get('type')=='h3_progress']
    assert any(0<e['value']<50 and e['phase']=='denoising' for e in progress), 'No real intermediate progress!'
    assert any(e['phase']=='completed' for e in progress)
    assert all(e['prompt_id']==pid for e in progress), 'Missing native Comfy context'
    assert progress[-1]['phase']=='completed'
    print('SUCCESS',json.dumps({k:v for k,v in result.items() if k!='events'}),flush=True)
 finally:
  async with aiohttp.ClientSession(connector=aiohttp.UnixConnector(path='/home/xql/.local/state/gpu-runtime/agent.sock')) as s:
   async with s.post('http://localhost/workloads/graph/activity',json={'ticket':'h3-progress-test-'+JOB,'action':'end'}) as r: print('RELEASE',r.status,await r.text(),flush=True)
for label in (args.label+'-15s-first', args.label+'-15s-warm'):
 if (ROOT/(label+'.started.json')).exists():
  raise FileExistsError('Choose a new benchmark label; do not overwrite provenance')
 JOB=str(uuid.uuid4())
 (ROOT/(label+'.started.json')).write_text(json.dumps({'job_id':JOB,'requested_seconds':DURATION,'created_at':time.time(),'label':label}))
 print('TEST_JOB',label,JOB,flush=True)
 asyncio.run(main())
