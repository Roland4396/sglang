"""Same-math periodic-input TMA refill ablation. Not a production optimization."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
from refill_clock_source import generate

p = argparse.ArgumentParser()
p.add_argument('--output-dir', type=Path, required=True)
args = p.parse_args()
out = args.output_dir; out.mkdir(parents=True, exist_ok=True)
assert not (out/'result.json').exists()
source = generate(Path(__file__).resolve().parent)
(out/'refill.generated.cu').write_text(source)
import torch
from torch.utils.cpp_extension import load
from sageattention.core import per_thread_int8_triton, per_channel_fp8, sm90_compile
os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0a'
os.environ.setdefault('MAX_JOBS', '2')
assert torch.cuda.device_count() == 1 and torch.cuda.get_device_capability() == (9, 0)
module = load(name='h3_refill_probe_20260916_v1', sources=[str(out/'refill.generated.cu')],
    extra_cuda_cflags=['-O3','--use_fast_math','-lineinfo','--ptxas-options=-v',
        '-U__CUDA_NO_HALF_OPERATORS__','-U__CUDA_NO_HALF_CONVERSIONS__',
        '-U__CUDA_NO_BFLOAT16_CONVERSIONS__','-U__CUDA_NO_HALF2_OPERATORS__'],
    extra_ldflags=['-lcuda'], verbose=True)
results = {'scope': 'periodic quantized K/V only; ablation outputs are not valid for arbitrary inputs',
           'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
           'binary_sha256': hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
           'attributes': {str(i): module.attributes(i) for i in range(4)}, 'checks': []}
for flag,label in [('-res-usage','resources'),('-sass','sass')]:
    with (out/('refill-'+label+'.txt')).open('w') as f:
        subprocess.run([os.environ['CUDA_HOME']+'/bin/cuobjdump',flag,module.__file__],stdout=f,stderr=subprocess.STDOUT,check=True)


def save(): (out/'result.json').write_text(json.dumps(results,indent=2))


with torch.inference_mode():
    for n in (128, 256, 4096, 32768, 109184):
        h = 3 if n < 4096 else 14
        torch.manual_seed(20260916+n)
        q,k,v = [torch.randn(1,n,h,128,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
        qi,qs,ki,ks = per_thread_int8_triton(q,k,k.mean(dim=1,keepdim=True),tensor_layout='NHD',BLKQ=64,WARPQ=16,BLKK=128,WARPK=128)
        vi,vs,_ = per_channel_fp8(v,tensor_layout='NHD',smooth_v=False)
        torch.cuda.synchronize()
        # Repeat quantized values, not floating inputs: avoids tail/scale rounding
        # changing the exact data seen by the installed/control implementation.
        assert tuple(ki.shape)==(1,n,h,128) and tuple(vi.shape)==(1,128,h,n)
        ki.copy_(ki[:,:128].repeat(1,n//128,1,1))
        vi.copy_(vi[...,:128].repeat(1,1,1,n//128))
        assert torch.equal(ki,ki[:,:128].repeat(1,n//128,1,1))
        assert torch.equal(vi,vi[...,:128].repeat(1,1,1,n//128))
        buffers={name:torch.empty_like(q) for name in ['installed','normal','no_k_refill','no_v_refill','no_kv_refill']}
        stamps=torch.empty(math.ceil(math.ceil(n/64)*h/128),8,device='cuda',dtype=torch.int64)
        funcs={'installed': lambda:sm90_compile.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(qi,ki,vi,buffers['installed'],qs,ks,vs,0,0,3,128**-0.5,0)}
        for mode,name in enumerate(list(buffers)[1:]):
            funcs[name]=lambda mode=mode,name=name:module.run(qi,ki,vi,buffers[name],qs,ks,vs,stamps,128,-1,mode)
        for f in funcs.values(): f()
        torch.cuda.synchronize()
        for name,x in buffers.items():
            assert torch.isfinite(x).all().item(), (n,name,'nonfinite')
            assert torch.equal(x,buffers['installed']), (n,name,'not bitwise equal')
        stream=torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for name in list(buffers)[1:]: funcs[name]()
        stream.synchronize()
        assert all(torch.equal(x,buffers['installed']) for x in buffers.values())
        results['checks'].append({'n':n,'heads':h,'periodic_tiles_verified':True,'bitwise_equal':True,'finite':True,'nondefault_stream':True})
        save();print('CHECK',n,'PASS',flush=True)
        if n != 109184:
            del q,k,v,qi,ki,vi,qs,ks,vs,buffers,stamps,funcs
            torch.cuda.empty_cache()
    for _ in range(3):
        for f in funcs.values():f()
    torch.cuda.synchronize()
    times={name:[] for name in funcs}
    for repeat in range(9):
        names=list(funcs); names=names[repeat%len(names):]+names[:repeat%len(names)]
        for name in names:
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record();funcs[name]();b.record();b.synchronize()
            times[name].append(a.elapsed_time(b))
    results['timings']={'samples_ms':times,'median_ms':{k:statistics.median(v) for k,v in times.items()}}
    results['shape']=[1,109184,14,128]
    results['scope_note']='Length rounded to complete 128-token tiles; same CTA grid and K-loop count as 109129. No clock instrumentation. Only periodic input ablation, not video speedup.'
    save();print('PASS',json.dumps(results),flush=True)
