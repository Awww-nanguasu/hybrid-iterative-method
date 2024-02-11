// @author jpzxshi (jpz@pku.edu.cn)
#pragma once
#include <torch/script.h>

namespace ln
{
	using Inputs = torch::jit::Stack;

	c10::InferenceMode Inference_mode();

	class Module:public torch::jit::Module
	{
	public:
		Module(torch::jit::Module module, c10::DeviceType device, c10::ScalarType dtype);

		c10::DeviceType device() const;
		c10::ScalarType dtype() const;
		void set_device(c10::DeviceType device);
		void set_dtype(c10::ScalarType dtype);
	private:
		c10::DeviceType __device;
		c10::ScalarType __dtype;
	};

	class Data
	{
	public:
		at::Tensor X_train;
		at::Tensor y_train;
		at::Tensor X_test;
		at::Tensor y_test;
	};

	class Data_MIONet
	{
	public:
		std::vector<at::Tensor> X_train;
		at::Tensor y_train;
		std::vector<at::Tensor> X_test;
		at::Tensor y_test;
	};

	class Loader
	{
	public:
		Loader(c10::DeviceType device = at::kCUDA, c10::ScalarType dtype = at::kDouble);

		Module load_net(const std::string& filename);

		Data load_data(const std::string& filename);

		Data_MIONet load_mionet_data(const std::string& filename);

		c10::DeviceType load_device() const;
		c10::ScalarType load_dtype() const;
		void set_load_device(c10::DeviceType device);
		void set_load_dtype(c10::ScalarType dtype);

	private:
		c10::DeviceType __load_device;
		c10::ScalarType __load_dtype;
	};

	void pickle_save(const std::string& filename, const c10::IValue& ivalue);
	c10::IValue pickle_load(const std::string& filename);
}