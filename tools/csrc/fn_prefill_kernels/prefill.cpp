void pre_pair_rot(torch::Tensor x, torch::Tensor row, torch::Tensor eid, torch::Tensor prow, torch::Tensor tr, torch::Tensor kb, torch::Tensor su, torch::Tensor sv, int64_t mat0, torch::Tensor xh);
void pre_pair_silu_rot(torch::Tensor y, torch::Tensor eid, torch::Tensor prow, torch::Tensor tr, torch::Tensor kb, torch::Tensor su, torch::Tensor sv, torch::Tensor xh);
void pre_group_gemm(torch::Tensor xh, torch::Tensor cnt, torch::Tensor pstart, torch::Tensor iend, torch::Tensor tr, torch::Tensor kb, torch::Tensor su, torch::Tensor sv, int64_t mat0, int64_t N, torch::Tensor y, c10::optional<torch::Tensor> dst, int64_t items, int64_t mt, int64_t kbmax);
void pre_mix_tail(torch::Tensor u, torch::Tensor n, torch::Tensor x, int64_t hc);
