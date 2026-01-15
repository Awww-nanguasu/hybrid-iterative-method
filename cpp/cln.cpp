// @author jpzxshi (jpz@pku.edu.cn)
#include "cln.hpp"
#include <torch/torch.h>

#include <memory>

// 修改返回类型为 std::unique_ptr<c10::InferenceMode>
std::unique_ptr<c10::InferenceMode> ln::Inference_mode()
{
	// 在堆内存上创建 guard，而不是在栈上
	auto guard = std::make_unique<c10::InferenceMode>(true); // ensure no autograd

	torch::jit::getProfilingMode() = false;
	torch::jit::getExecutorMode() = false;

	// 返回指针（指针是可以被移动的）
	return guard;
}

// Module
ln::Module::Module(torch::jit::Module module, c10::DeviceType device, c10::ScalarType dtype)
	: torch::jit::Module(module), __device(device), __dtype(dtype)
{
	this->to(device);
	this->to(dtype);
}

c10::DeviceType ln::Module::device() const
{
	return __device;
}

c10::ScalarType ln::Module::dtype() const
{
	return __dtype;
}

void ln::Module::set_device(c10::DeviceType device)
{
	this->to(device);
	__device = device;
}

void ln::Module::set_dtype(c10::ScalarType dtype)
{
	this->to(dtype);
	__dtype = dtype;
}

// Loader
ln::Loader::Loader(c10::DeviceType device, c10::ScalarType dtype)
	: __load_device(device), __load_dtype(dtype)
{
}

ln::Module ln::Loader::load_net(const std::string &filename)
{
	auto net = torch::jit::load(filename, __load_device);
	net.to(__load_dtype);
	return ln::Module(net, __load_device, __load_dtype);
}

ln::Data ln::Loader::load_data(const std::string &filename)
{
	ln::Data dataset;
	auto data = torch::jit::load(filename, __load_device);
	dataset.X_train = data.attr("X_train").toTensor().to(__load_dtype);
	dataset.y_train = data.attr("y_train").toTensor().to(__load_dtype);
	dataset.X_test = data.attr("X_test").toTensor().to(__load_dtype);
	dataset.y_test = data.attr("y_test").toTensor().to(__load_dtype);
	return dataset;
}

ln::Data_MIONet ln::Loader::load_mionet_data(const std::string &filename)
{
	ln::Data_MIONet dataset;
	auto data = torch::jit::load(filename, __load_device);
	// X_train
	auto X_train_temp = data.attr("X_train").toTuple()->elements();
	for (auto iter = X_train_temp.cbegin(); iter != X_train_temp.cend(); ++iter)
	{
		dataset.X_train.push_back(iter->toTensor().to(__load_dtype));
	}
	// y_train
	dataset.y_train = data.attr("y_train").toTensor().to(__load_dtype);
	// X_test
	auto X_test_temp = data.attr("X_test").toTuple()->elements();
	for (auto iter = X_test_temp.cbegin(); iter != X_test_temp.cend(); ++iter)
	{
		dataset.X_test.push_back(iter->toTensor().to(__load_dtype));
	}
	// y_test
	dataset.y_test = data.attr("y_test").toTensor().to(__load_dtype);
	return dataset;
}

c10::DeviceType ln::Loader::load_device() const
{
	return __load_device;
}

c10::ScalarType ln::Loader::load_dtype() const
{
	return __load_dtype;
}

void ln::Loader::set_load_device(c10::DeviceType device)
{
	__load_device = device;
}

void ln::Loader::set_load_dtype(c10::ScalarType dtype)
{
	__load_dtype = dtype;
}

void ln::pickle_save(const std::string &filename, const c10::IValue &ivalue)
{
	std::ofstream ofile(filename, std::ios::binary);
	std::vector<char> odata = torch::pickle_save(ivalue);
	ofile.write(reinterpret_cast<char *>(&odata[0]), odata.size() * sizeof(char));
	ofile.close();
}

c10::IValue ln::pickle_load(const std::string &filename)
{
	std::ifstream ifile(filename, std::ios::binary);
	ifile.seekg(0, std::ios::end);
	size_t fileSize = ifile.tellg();
	ifile.seekg(0, std::ios::beg);
	char *buffer = new char[fileSize];
	ifile.read(buffer, fileSize);
	ifile.close();
	std::vector<char> idata(fileSize);
	for (size_t i = 0; i != fileSize; ++i)
	{
		idata[i] = (*(buffer + i));
	}
	delete[] buffer;
	return torch::pickle_load(idata);
}