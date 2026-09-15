"""Node-local synthetic benchmark. Invoke only on an Ansible-selected idle H3 node.

Uses native profiler; profile request wall time is NOT a speed measurement.
Does not read any user prompt files. Completed artifacts remain on the node.
"""
import json,os,subprocess,time,uuid
from pathlib import Path
ROOT=Path(os.environ.get('H3_OPERATOR_OUTPUT','/home/xql/.local/state/gpu-runtime/benchmarks/operators-'+time.strftime('%Y%m%d-%H%M%S')));ROOT.mkdir(parents=True,exist_ok=True)
PROMPT='integrated_multimodal_description: A continuous eye-level medium shot of a red toy robot walking slowly on a wooden desk beside a small green potted plant. Soft daylight, sharp clean shapes, consistent colors, gentle camera push-in. overall_soundscape: Quiet footsteps and soft room ambience. non_diegetic_music: None.'
def api(path,payload=None):
 args=['curl','-fsS','--max-time','45','--unix-socket','/home/xql/.local/state/gpu-runtime/orchestrator.sock']
 if payload is not None:args+=['-H','Content-Type: application/json','-H','Idempotency-Key: '+payload['_h3_operation_id'],'--data',json.dumps(payload)]
 return json.loads(subprocess.check_output(args+['http://localhost/h3/v1/videos'+path]))
for label,profile in [('operator-warmup',False),('operator-profile',True)]:
 job=str(uuid.uuid4());started=time.time()
 payload={'model':'MiniMaxAI/MiniMax-H3','prompt':PROMPT,'seconds':15,'task':'t2va','conditions':[],'target':{'short_edge':768,'aspect_ratio':'16:9','duration_seconds':15.0},'num_outputs_per_prompt':1,'num_inference_steps':51,'_h3_turbo_lora':{'enabled':False},'flow_shift':12.0,'audio_flow_shift':3.0,'seed':20260916,'_h3_operation_id':job,'perf_dump_path':str(ROOT/(job+'-perf.json'))}
 if profile:payload.update(profile=True,num_profiled_timesteps=2,profile_all_stages=False)
 (ROOT/(label+'.json')).write_text(json.dumps({'job_id':job,'started_at':started,'payload':payload},indent=2))
 print('START',label,job,flush=True);s=api('',payload);assert s['id']==job,s;events=[]
 while True:
  s=api('/'+job); events.append({'observed_at':time.time(),**s});print(label,json.dumps(s),flush=True)
  if s['status'] in ('completed','failed','cancelled'):break
  time.sleep(10)
 result={'job_id':job,'label':label,'elapsed_seconds':time.time()-started,'status':s,'events':events}
 (ROOT/(job+'-result.json')).write_text(json.dumps(result,indent=2));assert s['status']=='completed',s
 print('PASS',label,job,flush=True)
