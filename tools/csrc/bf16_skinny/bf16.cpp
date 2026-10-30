void bf16_skinny(torch::Tensor x, torch::Tensor w, torch::Tensor y, int64_t wk, int64_t nt, int64_t pf);
void bf16_cluster(torch::Tensor x, torch::Tensor w, torch::Tensor y, int64_t wk, int64_t cs);
