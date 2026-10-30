
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdint.h>

__global__ void l2_prefetch(const uint8_t* __restrict__ p, size_t n) {
    size_t step = (size_t)gridDim.x * blockDim.x * 128;
    for (size_t o = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 128; o < n; o += step)
        asm volatile("prefetch.global.L2::evict_last [%0];" :: "l"(p + o));
}

void prefetch(torch::Tensor t, int64_t ctas) {
    l2_prefetch<<<(int)ctas, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const uint8_t*>(t.data_ptr()), (size_t)t.numel() * t.element_size());
}
