"""Generate an isolated, opt-in clock64 probe from the reviewed experimental kernel.

No production source/package is changed. Markers sample the elapsed critical path
of one thread in selected CTAs, NOT exclusive instruction cycles or HW stall events.
"""
from pathlib import Path
import re

PHASES = ["k_scale_and_ready", "qk_mma_and_wait", "k_prefetch_issue",
          "softmax_rescale_and_fp8", "v_ready_wait", "pv_mma_and_wait",
          "output_accumulate_and_v_prefetch"]


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected one source anchor, found {text.count(old)}: {old[:100]}")
    return text.replace(old, new, 1)


def generate(root: Path):
    path = root / "csrc/qattn/h3_tile.cu"
    src = path.read_text().split("// Deliberately narrow experimental API:")[0]
    src = re.sub(r'#include "([^"]+)"',
                 lambda m: '#include "' + str((path.parent / m[1]).resolve()) + '"', src)
    marker = r'''
template<int Profile>
__device__ __forceinline__ void probe_stamp(unsigned long long *stamps,
    int cta_stride, int sample_iter, int iter, int phase) {
  if constexpr (Profile) {
    const unsigned int linear = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
    if (iter == sample_iter && (linear & (cta_stride - 1)) == 0 && threadIdx.x == 0) {
      unsigned long long t;
      asm volatile("mov.u64 %0, %%clock64;" : "=l"(t) :: "memory");
      reinterpret_cast<volatile unsigned long long*>(stamps)[(linear / cta_stride) * 8 + phase] = t;
    }
  }
}
'''
    src = once(src, "template<uint32_t CTA_Q", marker + "\ntemplate<uint32_t CTA_Q")
    src = once(src, "bool scratch_pipeline=false>", "bool scratch_pipeline=false, int Profile=0>")
    src = once(src, "                                        float sm_scale)",
               "                                        float sm_scale, unsigned long long *stamps, int cta_stride, int sample_iter)")
    a, rest = src.split("  int p = 1;", 1)
    loop, tail = rest.split("\n  }\n\n  { \n    p ^= 1;", 1)
    loop += "\n  }"
    def stamp(i):
        return f"    probe_stamp<Profile>(stamps, cta_stride, sample_iter, iter, {i});\n"
    loop = once(loop, "    p ^= 1;", "    p ^= 1;\n" + stamp(0))
    loop = once(loop, "    wait(&barrier_K, p);", "    wait(&barrier_K, p);\n" + stamp(1))
    loop = once(loop, "    }\n    // load K", "    }\n" + stamp(2) + "    // load K")
    loop = once(loop,
                "      load_async_4D(sK, &tensorMapK, &barrier_K, 0, iter * CTA_K, kv_head_id, batch_id);\n    }",
                "      load_async_4D(sK, &tensorMapK, &barrier_K, 0, iter * CTA_K, kv_head_id, batch_id);\n    }\n" + stamp(3))
    loop = once(loop, "    // wait for V\n    wait(&barrier_V, p);",
                stamp(4) + "    // wait for V\n    wait(&barrier_V, p);\n" + stamp(5))
    anchor = "    // Both warp groups must finish reading shared K/V before TMA overwrites it."
    # Second occurrence is the unsplit PV completion; first is QK.
    assert loop.count(anchor) == 2
    idx = loop.rindex(anchor)
    loop = loop[:idx] + stamp(6) + loop[idx:]
    loop = once(loop, "      load_async_4D(sV, &tensorMapV, &barrier_V, iter * CTA_K, 0, kv_head_id, batch_id);\n    }\n  }",
                "      load_async_4D(sV, &tensorMapV, &barrier_V, iter * CTA_K, 0, kv_head_id, batch_id);\n    }\n" + stamp(7) + "  }")
    src = a + "  int p = 1;" + loop + "\n\n  { \n    p ^= 1;" + tail
    src += r'''
template<int Profile>
void probe_launch(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o,
                  torch::Tensor qs, torch::Tensor ks, torch::Tensor vs, torch::Tensor stamps,
                  int stride, int sample_iter) {
  auto qm=create_tensor_map_4D<64,128>((int8_t*)q.data_ptr(),q.size(0),q.size(2),q.size(1),128,q.stride(0),q.stride(2),q.stride(1));
  auto km=create_tensor_map_4D<128,128>((int8_t*)k.data_ptr(),k.size(0),k.size(2),k.size(1),128,k.stride(0),k.stride(2),k.stride(1));
  auto vm=create_tensor_map_4D<128,128>((int8_t*)v.data_ptr(),v.size(0),v.size(2),128,v.size(3),v.stride(0),v.stride(2),v.stride(1));
  auto kernel=qk_int8_sv_f8_attn_kernel<64,128,128,128,QuantGranularity::kPerThread,QuantGranularity::kPerThread,nv_bfloat16,MaskMode::kNone,false,true,false,3,false,false,Profile>;
  TORCH_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,40960)==cudaSuccess);
  kernel<<<dim3(div_ceil(q.size(1),64),q.size(2),q.size(0)),128,40960,at::cuda::getCurrentCUDAStream()>>>(
    qm,km,vm,qs.data_ptr<float>(),ks.data_ptr<float>(),vs.data_ptr<float>(),
    (nv_bfloat16*)o.data_ptr(),nullptr,o.stride(0),o.stride(2),o.stride(1),q.size(1),k.size(1),1,
    0.08838834764831845f,(unsigned long long*)stamps.data_ptr<int64_t>(),stride,sample_iter);
  auto err=cudaGetLastError();TORCH_CHECK(err==cudaSuccess,cudaGetErrorString(err));
}
void run_probe(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o,
               torch::Tensor qs, torch::Tensor ks, torch::Tensor vs, torch::Tensor stamps,
               int stride, int sample_iter, bool profile) {
  c10::cuda::CUDAGuard guard(q.device());
  TORCH_CHECK(stride>0 && (stride&(stride-1))==0, "power-of-two CTA stride required");
  TORCH_CHECK(stamps.is_cuda() && stamps.device()==q.device() && stamps.scalar_type()==torch::kInt64 && stamps.is_contiguous());
  TORCH_CHECK(q.size(3)==128 && q.sizes()==k.sizes() && o.sizes()==q.sizes());
  TORCH_CHECK(stamps.numel()>=div_ceil(div_ceil(q.size(1),64)*q.size(2)*q.size(0),stride)*8);
  if(profile) probe_launch<1>(q,k,v,o,qs,ks,vs,stamps,stride,sample_iter);
  else probe_launch<0>(q,k,v,o,qs,ks,vs,stamps,stride,sample_iter);
}
template<int Profile>
std::vector<int64_t> attributes() {
  auto kernel=qk_int8_sv_f8_attn_kernel<64,128,128,128,QuantGranularity::kPerThread,QuantGranularity::kPerThread,nv_bfloat16,MaskMode::kNone,false,true,false,3,false,false,Profile>;
  cudaFuncAttributes a;int blocks;
  TORCH_CHECK(cudaFuncGetAttributes(&a,kernel)==cudaSuccess);
  TORCH_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,kernel,128,40960)==cudaSuccess);
  return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,blocks};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  m.def("run", &run_probe);
  m.def("attributes", [](bool profile){return profile ? attributes<1>() : attributes<0>();});
}
'''
    assert src.count("probe_stamp<Profile>(") == 8
    return src
