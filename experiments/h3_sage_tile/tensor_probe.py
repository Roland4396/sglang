"""Same-card empirical tensor / bulk-memory calibration; no HW counters required.

These are synthetic diagnostic workloads, never valid H3/video replacements.
Run on a single idle GPU selected through the two-host Ansible inventory.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess

parser=argparse.ArgumentParser()
parser.add_argument('--output-dir',type=Path,required=True)
args=parser.parse_args()
out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
assert not (out/'tensor-result.json').exists(), 'Choose a fresh result directory'
import torch
from torch.utils.cpp_extension import load
root=Path(__file__).resolve().parent
os.environ['TORCH_CUDA_ARCH_LIST']='9.0a'
os.environ.setdefault('MAX_JOBS','2')
assert torch.cuda.device_count()==1
assert torch.cuda.get_device_capability()==(9,0)
module=load(name='h3_tensor_probe_20260916_v1',sources=[str(root/'csrc/tensor_probe.cu')],
            extra_cuda_cflags=['-O3','-lineinfo','--ptxas-options=-v'],verbose=True)
results={'scope':'achieved synthetic WGMMA throughput; NOT theoretical peak, tensor utilization or valid attention output',
         'device':str(torch.cuda.get_device_properties(0)),
         'binary_sha256':hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()}
for flag,label in [('-res-usage','resources'),('-sass','sass')]:
    with (out/('tensor-'+label+'.txt')).open('w') as f:
        subprocess.run([os.environ['CUDA_HOME']+'/bin/cuobjdump',flag,module.__file__],stdout=f,stderr=subprocess.STDOUT,check=True)


def elapsed(fn):
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    a.record();fn();b.record();b.synchronize()
    return a.elapsed_time(b)


with torch.inference_mode():
    # Same CTA grid and K-tile count as the captured real H3 attention shape.
    blocks=math.ceil(109129/64)*14
    iterations=math.ceil(109129/128)
    output=torch.empty(blocks*128,device='cuda',dtype=torch.float32)
    variants=[(m,c) for c in (3,5) for m in (0,1,2)]
    times={f'{m}:{c}':[] for m,c in variants}
    for m,c in variants:
        module.run(output,blocks,iterations,m,c)
        assert (output==(256 if m==2 else 128)).all().item(), (m,c,'analytical check failed')
        for _ in range(2):module.run(output,blocks,iterations,m,c)
    torch.cuda.synchronize()
    for r in range(9):
        order=variants[r%len(variants):]+variants[:r%len(variants)]
        for m,c in order:
            times[f'{m}:{c}'].append(elapsed(lambda:module.run(output,blocks,iterations,m,c)))
    rows=[]
    for m,c in variants:
        ops=blocks*iterations*2*64*128*128*(2 if m==2 else 1)
        med=statistics.median(times[f'{m}:{c}'])
        rows.append({'mode':['int8_qk_only','fp8_pv_only','alternating_qk_pv'][m],
                     'min_ctas':c,'samples_ms':times[f'{m}:{c}'],'median_ms':med,
                     'operations':ops,'achieved_teraops_per_s':ops/(med*1e9),'analytical_output_check':True})
    results.update(blocks=blocks,iterations=iterations,tensor=rows)
    # Both arrays exceed the GPU's L2 capacity. This is bulk copy throughput,
    # not a measurement of attention's TMA/L2/shared-memory path.
    size=512*1024*1024
    src=torch.full((size,),123,dtype=torch.uint8,device='cuda');dst=torch.empty_like(src)
    for _ in range(3):dst.copy_(src)
    torch.cuda.synchronize()
    copy_ms=[elapsed(lambda:dst.copy_(src)) for _ in range(9)]
    assert torch.equal(src,dst)
    results['bulk_copy']={'bytes_each_array':size,'samples_ms':copy_ms,
        'read_plus_write_TB_s':2*size/(statistics.median(copy_ms)*1e9),
        'scope':'device bulk-copy traffic estimate, not attention DRAM counter utilization'}
(out/'tensor-result.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2),flush=True)
