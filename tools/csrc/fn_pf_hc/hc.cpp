void pf_inject_norm(torch::Tensor h0, torch::Tensor out, torch::Tensor d, int64_t R, c10::optional<torch::Tensor> w, torch::Tensor hn, torch::Tensor nn, c10::optional<torch::Tensor> n32, int64_t hc, double eps);
void pf_mix_a(torch::Tensor d, torch::Tensor a, int64_t hc);
