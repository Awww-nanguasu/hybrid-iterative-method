// @author jpzxshi (jpz@pku.edu.cn)
#pragma once
#include <torch/torch.h>

namespace gen
{
	std::tuple<at::Tensor, at::Tensor> load_kf(const std::string& filename);

	std::tuple<at::Tensor, at::Tensor, at::Tensor> load_kfg(const std::string& filename);

	at::Tensor solve_sparse(const at::Tensor& k, const at::Tensor& f);

	at::Tensor solve_sparse(const at::Tensor& k, const at::Tensor& f, const at::Tensor& g);

	void generate_data_large(const std::string& loadname, const std::string& savename);

	void generate_data_boundary(const std::string& loadname, const std::string& savename);
}
