void bf16_skinny(torch::Tensor x, torch::Tensor w, torch::Tensor y, int64_t wk, int64_t nt, int64_t pf);
void bf16_cluster(torch::Tensor x, torch::Tensor w, torch::Tensor y, int64_t wk, int64_t cs);
void mixup_bf16(torch::Tensor d, torch::Tensor n, torch::Tensor w, torch::Tensor x, int64_t hc, int64_t jg);
