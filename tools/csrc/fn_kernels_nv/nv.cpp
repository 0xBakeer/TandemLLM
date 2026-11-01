void nv_gemv(torch::Tensor x, torch::Tensor w, torch::Tensor y);
void nv_hc_norm(torch::Tensor x, torch::Tensor w, torch::Tensor y, int64_t G, double eps);
void nv_inject_norm(torch::Tensor h0, torch::Tensor out, torch::Tensor d, int64_t R, torch::Tensor w,
                    torch::Tensor hn, torch::Tensor nn, int64_t hc, double eps);
void nv_combine_inject_norm(torch::Tensor h0, torch::Tensor yd, torch::Tensor ids, torch::Tensor wts, torch::Tensor d,
                            int64_t R, torch::Tensor w, torch::Tensor hn, torch::Tensor nn, int64_t hc, double eps);
void fn_set_pdl(int64_t on);
void fn_set_gemv_split(int64_t on);
void fn_set_gemv_nb(int64_t on);
void fn_set_gemv_mc(int64_t mc);
