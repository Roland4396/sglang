// Diagnostic WGMMA throughput probe. NOT an attention implementation.
// Uses the same INT8 SS and FP8 RS instruction families as H3 Sage, but removes
// softmax, HBM reads and output rescaling. Achieved throughput is not a rated peak.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "wgmma.cuh"

template<int Mode, int MinCTAs>
__global__ __launch_bounds__(128, MinCTAs)
void tensor_probe_kernel(float *out, int iterations) {
  extern __shared__ __align__(128) int8_t smem[];
  int8_t *q=smem, *k=smem+8192, *v=smem+24576;
  for(int i=threadIdx.x; i<40960; i+=128) smem[i]=i<24576 ? 1 : 0x38;
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  float result=0;
  for(int it=0;it<iterations;it++) {
    float qresult=0, vresult=0;
    if constexpr(Mode==0 || Mode==2) {
      int32_t r[8][8];
      wgmma::warpgroup_arrive();
      wgmma::wgmma_s8s8s32<128,0,128>(r,q,k);
#pragma unroll
      for(int step=1;step<4;step++)
        wgmma::wgmma_s8s8s32<128,1,128>(r,q+step*32,k+step*32);
      wgmma::warpgroup_commit_batch();
      wgmma::warpgroup_wait<0>();
      qresult=static_cast<float>(r[0][0]);
    }
    if constexpr(Mode==1 || Mode==2) {
      float r[8][8];
      uint32_t a[4]={0x38383838,0x38383838,0x38383838,0x38383838};
      wgmma::warpgroup_arrive();
      wgmma::wgmma_f8f8f32<128,0,128>(r,a,v);
#pragma unroll
      for(int step=1;step<4;step++)
        wgmma::wgmma_f8f8f32<128,1,128>(r,a,v+step*32);
      wgmma::warpgroup_commit_batch();
      wgmma::warpgroup_wait<0>();
      vresult=r[0][0];
    }
    // Every iteration must remain observable. Overwriting result allowed ptxas
    // to replace three of four unrolled WGMMA groups with dummy operations.
    result+=qresult+vresult;
  }
  out[blockIdx.x*128+threadIdx.x]=result;
}

template<int Mode,int MinCTAs>
void launch_probe(torch::Tensor out,int blocks,int iterations) {
  auto kernel=tensor_probe_kernel<Mode,MinCTAs>;
  TORCH_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,40960)==cudaSuccess);
  kernel<<<blocks,128,40960,at::cuda::getCurrentCUDAStream()>>>(out.data_ptr<float>(),iterations);
  auto e=cudaGetLastError();TORCH_CHECK(e==cudaSuccess,cudaGetErrorString(e));
}
void run(torch::Tensor out,int blocks,int iterations,int mode,int min_ctas) {
  TORCH_CHECK(out.is_cuda() && out.scalar_type()==torch::kFloat32 && out.is_contiguous());
  TORCH_CHECK(blocks>0 && iterations>0 && out.numel()>=blocks*128);
  c10::cuda::CUDAGuard guard(out.device());
#define GO(M,C) if(mode==M && min_ctas==C) {launch_probe<M,C>(out,blocks,iterations);return;}
  GO(0,3);GO(1,3);GO(2,3);GO(0,5);GO(1,5);GO(2,5);
#undef GO
  TORCH_CHECK(false,"unsupported diagnostic mode");
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("run",&run);}
