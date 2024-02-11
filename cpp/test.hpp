// @author jpzxshi (jpz@pku.edu.cn)
#pragma once
#include <torch/torch.h>
#include "cln.hpp"

void device_test()
{
	torch::Tensor tensor = torch::rand({ 2, 3 });
	if (torch::cuda::is_available()) {
		std::cout << "CUDA is available! Training on GPU" << std::endl;
		auto tensor_cuda = tensor.cuda();
		std::cout << tensor_cuda << std::endl;
	}
	else
	{
		std::cout << "CUDA is not available! Training on CPU" << std::endl;
		std::cout << tensor << std::endl;
	}
}

void model_test()
{
	auto guard = ln::Inference_mode();
	
	//auto a = torch::tensor( 2.0 ).requires_grad_(true);
	//std::cout << a.requires_grad() << std::endl;
	//auto y = a * a;
	//auto grads = torch::autograd::grad({ y }, { a });

	ln::Loader loader(at::kCUDA, at::kDouble);
	auto mionet = loader.load_net("./model/Poisson_2d_5000/mionet_poisson_2d.pt");

	auto k = torch::ones({ 2, 10000 }, at::device(at::kCUDA).dtype(at::kDouble));
	auto f = torch::ones({ 2, 10000 }, at::device(at::kCUDA).dtype(at::kDouble));
	auto x = torch::tensor({ {1, 2}, {3, 4}, {5, 6}, {7, 8} }, at::device(at::kCUDA).dtype(at::kDouble));
	ln::Inputs inputs = { std::make_tuple(k, f, x) };
	
	at::Tensor u;
	// timing
	auto start = clock();
	auto iters = 1;
	for (auto i = 0; i != iters; ++i)
	{
		u = mionet.forward(inputs).toTensor();
		//std::cout << i;
	}
	auto end = clock();
	// once: debug 680 ms ; release 580 ms // 1000average: debug 7.8 ms ; release 7.2 ms
	std::cout << "prediction is \n";
	std::cout << u << "\n";
	std::cout << iters << " iters, time: " << end - start << " ms\n";
	//std::cout << torch::jit::getProfilingMode() << torch::jit::getExecutorMode();
	std::cout << std::endl;
}

void data_test()
{
	auto guard = ln::Inference_mode();

	ln::Loader loader(at::kCUDA, at::kDouble);
	auto mionet = loader.load_net("./model/Poisson_2d_5000/mionet_poisson_2d.pt");

	auto data = loader.load_mionet_data("./data/Poisson_2d_5000/data.pth");
	auto k = data.X_test[0];
	auto f = data.X_test[1];
	auto x = data.X_test[2];
	std::cout << "device: " << loader.load_device() << "\n" << "Shapes of k, f, x are " << k.sizes() << f.sizes() << x.sizes() << "\n";

	ln::Inputs inputs = { std::make_tuple(k, f, x) };
	auto u = mionet.forward(inputs).toTensor();
	std::cout << "Shape of prediction is " << u.sizes() << "\n";

	auto error = torch::mean((data.y_test - u).pow(2));
	std::cout << "The test MSE is " << error << "\n";

	std::cout << "u_true (slice) is\n" << data.y_test[22].index({ torch::indexing::Slice(1005, 1010) }) << "\n";
	std::cout << "u_pred (slice) is\n" << u[22].index({ torch::indexing::Slice(1005, 1010) }) << "\n";
	std::cout << std::endl;
}