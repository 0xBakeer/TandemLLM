void nvfp4_moe_plan(torch::Tensor ids, int64_t E, torch::Tensor order, torch::Tensor gstart, torch::Tensor gcount,
                    torch::Tensor gexp, torch::Tensor ng);
void nvfp4_moe_gu(torch::Tensor x, int64_t S, torch::Tensor order, torch::Tensor gstart, torch::Tensor gcount,
                  torch::Tensor gexp, torch::Tensor ng, torch::Tensor w, torch::Tensor sc, torch::Tensor s2,
                  torch::Tensor a, int64_t gy, int64_t wk);
void nvfp4_moe_down(torch::Tensor a, torch::Tensor order, torch::Tensor gstart, torch::Tensor gcount,
                    torch::Tensor gexp, torch::Tensor ng, torch::Tensor w, torch::Tensor sc, torch::Tensor s2,
                    torch::Tensor y, int64_t gy, int64_t wk, int64_t nt);
