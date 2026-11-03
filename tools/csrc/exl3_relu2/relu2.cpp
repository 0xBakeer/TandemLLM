void nh_relu2_up(torch::Tensor x, torch::Tensor ids, torch::Tensor tr, torch::Tensor kb, torch::Tensor su, torch::Tensor sv, torch::Tensor a, int64_t kbmax);
void nh_relu2_down(torch::Tensor a, torch::Tensor ids, torch::Tensor tr, torch::Tensor kb, torch::Tensor su, torch::Tensor sv, torch::Tensor yd, int64_t kbmax);
void nh_relu2_groups(torch::Tensor ids, torch::Tensor order, torch::Tensor g, torch::Tensor ng);
void nh_relu2_up_group(torch::Tensor x, torch::Tensor g, torch::Tensor ng, int64_t S, torch::Tensor tr, torch::Tensor kb, torch::Tensor su, torch::Tensor sv, torch::Tensor a, int64_t kbmax);
void nh_relu2_down_group(torch::Tensor a, torch::Tensor g, torch::Tensor ng, torch::Tensor tr, torch::Tensor kb, torch::Tensor su, torch::Tensor sv, torch::Tensor yd, int64_t kbmax);
