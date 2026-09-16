"""Isolated H20 microbenchmark. Never installs or patches the production package."""
import gc,json,os,statistics,time
from pathlib import Path
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.utils.cpp_extension import load
from sageattention.core import per_thread_int8_triton,per_channel_fp8,sm90_compile

ROOT=Path(__file__).resolve().parent
OUT=Path(os.environ.get('H3_TILE_BENCH_OUT','/home/xql/.local/state/gpu-runtime/benchmarks/20260916-operators/tile-bench-v8.json'))
os.environ.setdefault('MAX_JOBS','2')
os.environ['TORCH_CUDA_ARCH_LIST']='9.0a'
started=time.time()
module=load(name='h3_sage_tile_20260916_v8',sources=[str(ROOT/'csrc/qattn/h3_tile.cu'),str(ROOT/'csrc/fused/v_quant.cu')],extra_cuda_cflags=['-O3','--use_fast_math','-U__CUDA_NO_HALF_OPERATORS__','-U__CUDA_NO_HALF_CONVERSIONS__','-U__CUDA_NO_BFLOAT16_CONVERSIONS__','-U__CUDA_NO_HALF2_OPERATORS__','--ptxas-options=-v'],extra_ldflags=['-lcuda'],verbose=True)
print('BUILD_SECONDS',time.time()-started,flush=True)
rows=[]

def compare(a,b):
 d=a.float()-b.float()
 return {'max_abs':d.abs().max().item(),'relative_l2':(d.norm()/a.float().norm().clamp_min(1e-20)).item(),'unequal_fraction':(a!=b).float().mean().item(),'finite':bool(torch.isfinite(b).all())}

@torch.inference_mode()
def run(n,heads,repeats):
 torch.manual_seed(20260916+n)
 q,k,v=[torch.randn(1,n,heads,128,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
 with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
  reference=torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2)).transpose(1,2)
 km=k.mean(dim=1,keepdim=True)
 qi,qs,ki,ks=per_thread_int8_triton(q,k,km,tensor_layout='NHD',BLKQ=64,WARPQ=16,BLKK=128,WARPK=128)
 pad=(-n)%128
 if pad:v=torch.cat([v,torch.zeros(1,pad,heads,128,device='cuda',dtype=v.dtype)],dim=1)
 vi,vs,_=per_channel_fp8(v,tensor_layout='NHD',smooth_v=False)
 scale=128**-0.5
 variants={'compiled_control':64,'k128_cta3':71,'lookahead_cta1':81,'lookahead_cta2':82,'lookahead_cta3':83}
 outputs={name:torch.empty_like(q) for name in ['installed',*variants]}
 funcs={'installed':lambda:sm90_compile.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(qi,ki,vi,outputs['installed'],qs,ks,vs,0,0,3,scale,0)}
 for name,tile in variants.items():
  funcs[name]=lambda name=name,tile=tile:module.forward(qi,ki,vi,outputs[name],qs,ks,vs,scale,tile)
 for f in funcs.values():f()
 torch.cuda.synchronize()
 checks={name:compare(outputs['installed'],out) for name,out in outputs.items()}
 reference_checks={name:compare(reference,out) for name,out in outputs.items()}
 assert all(c['finite'] for c in checks.values()),checks
 # Same K128 ordering and quantization: require bitwise equality here.
 for name in variants:
  assert checks[name]['unequal_fraction']==0,(n,name,checks[name])
 stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
 with torch.cuda.stream(stream):
  for name,tile in variants.items():module.forward(qi,ki,vi,outputs[name],qs,ks,vs,scale,tile)
 stream.synchronize()
 for name in variants:
  assert compare(outputs['installed'],outputs[name])==checks[name],(n,name,'stream mismatch')
 times={name:[] for name in funcs}
 for i in range(repeats):
  names=list(funcs);names=names[i%len(names):]+names[:i%len(names)]
  for name in names:
   a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
   a.record();funcs[name]();b.record();b.synchronize();times[name].append(a.elapsed_time(b))
 row={'n':n,'heads':heads,'head_dim':128,'checks':checks,'reference_bf16_flash_checks':reference_checks,'milliseconds':times,'median_ms':{k:statistics.median(v) for k,v in times.items()}}
 print('RESULT',json.dumps(row),flush=True);rows.append(row)
 OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(rows,indent=2))

for n,h,r in [(127,3,3),(128,3,3),(129,3,3),(257,3,3),(4096,14,5),(32768,14,7),(109129,14,7)]:
 run(n,h,r);gc.collect();torch.cuda.empty_cache()
print('PASS all finite / stream checks; inspect reference error before promotion',flush=True)
