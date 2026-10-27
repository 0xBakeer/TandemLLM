void exl3_gemm(torch::Tensor xh, torch::Tensor t, torch::Tensor svh, torch::Tensor y, int64_t kb, int64_t cfg, int64_t wk, int64_t S, int64_t b1, int64_t b2, int64_t rmax);
void exl3_rot_in(torch::Tensor x, torch::Tensor suh, torch::Tensor xh);
void exl3_rot_in_silu(torch::Tensor y, torch::Tensor suh, torch::Tensor xh);
void exl3_mlp_bridge(torch::Tensor yg, torch::Tensor yu, torch::Tensor svg, torch::Tensor svu, torch::Tensor suh, torch::Tensor xh);
void exl3_rot_out(torch::Tensor yp, torch::Tensor svh, torch::Tensor y);
void exl3_recon(torch::Tensor t, torch::Tensor wi, int64_t kb, int64_t n_off);
