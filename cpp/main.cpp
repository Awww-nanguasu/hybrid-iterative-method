// @author jpzxshi (jpz@pku.edu.cn)
#include "test.hpp"
#include "cln.hpp"
#include "hybrid_solver.hpp"
// #include <Eigen/core>
// #include <Eigen/Dense>
#include <Eigen/Sparse>
#include "gen_data.hpp"

void him_2d()
{
	// load net
	ln::Loader loader(at::kCPU, at::kDouble);
	auto mionet = loader.load_net("C:/git-workplace/HIM/outputs/2026-01-14-17-51-47/model_best_traced.pt");
	// load data
	loader.set_load_device(at::kCPU);
	auto data = loader.load_mionet_data("C:/git-workplace/HIM/data/Poisson_2d_5000/data.pth");
	// test
	int num = 0;
	auto k = data.X_test[0][num].reshape_symint({100, 100});
	auto f = data.X_test[1][num].reshape_symint({100, 100});
	auto x = data.X_test[2];
	auto u = data.y_test[num].reshape_symint({100, 100});
	std::string mode = "GS";
	double error_threshold = 1e-14;
	int nr = 300; // 30 (smallest iterations); 300 (smallest time)
	int m = 1;

	auto hybrid_solver = Hybrid_solver(mode, nr, error_threshold, m);
	std::tuple<double, double> time_iters = hybrid_solver.solve_poisson_2d(k, f, x, u, mionet);
	std::cout << "hybrid iterative method (" + mode + " + MIONet nr=" << nr << "):\n";
	std::cout << "time: " << std::get<0>(time_iters) << " ms iterations: " << std::get<1>(time_iters) << std::endl;
}

void him_2d_512()
{
	// load net
	ln::Loader loader(at::kCUDA, at::kDouble);
	auto mionet = loader.load_net("./model/Poisson_2d_5000/mionet_poisson_2d.pt");
	// load data
	auto data = ln::pickle_load("./data/Poisson_2d_size_512_512/data_10_512_512.bin").toTuple()->elements();
	// test
	int num = 0;
	auto k = data[0].toTensor()[num];
	auto f = data[1].toTensor()[num];
	auto u = data[2].toTensor()[num];
	auto mesh = torch::meshgrid({torch::linspace(0, 1, 512, k.options()), torch::linspace(0, 1, 512, k.options())}, "ij");
	at::Tensor x = torch::cat({mesh[0].reshape_symint({-1, 1}), mesh[1].reshape_symint({-1, 1})}, 1);
	std::string mode = "GS";
	double error_threshold = 1e-13;
	int nr = 1600;
	int split = 1;

	auto hybrid_solver = Hybrid_solver(mode, nr, error_threshold, split);
	std::tuple<double, double> time_iters = hybrid_solver.solve_poisson_2d_large(k, f, x, u, mionet);
	std::cout << "size: 512 * 512" << std::endl;
	std::cout << "hybrid iterative method (" + mode + " + MIONet nr=" << nr << "):\n";
	std::cout << "time: " << std::get<0>(time_iters) << " ms iterations: " << std::get<1>(time_iters) << std::endl;
}

void him_2d_1024()
{
	// load net
	ln::Loader loader(at::kCUDA, at::kDouble);
	auto mionet = loader.load_net("./model/Poisson_2d_5000/mionet_poisson_2d.pt");
	// load data
	auto data = ln::pickle_load("./data/Poisson_2d_size_1024_1024/data_10_1024_1024.bin").toTuple()->elements();
	// test
	int num = 0;
	auto k = data[0].toTensor()[num];
	auto f = data[1].toTensor()[num];
	auto u = data[2].toTensor()[num];
	auto mesh = torch::meshgrid({torch::linspace(0, 1, 1024, k.options()), torch::linspace(0, 1, 1024, k.options())}, "ij");
	at::Tensor x = torch::cat({mesh[0].reshape_symint({-1, 1}), mesh[1].reshape_symint({-1, 1})}, 1);
	std::string mode = "GS";
	double error_threshold = 1e-12; // 1e-12;
	int nr = 512000;
	int split = 2; // reduce required GPU memory, otherwise overflow

	auto hybrid_solver = Hybrid_solver(mode, nr, error_threshold, split);
	std::tuple<double, double> time_iters = hybrid_solver.solve_poisson_2d_large(k, f, x, u, mionet);
	std::cout << "size: 1024 * 1024" << std::endl;
	std::cout << "hybrid iterative method (" + mode + " + MIONet nr=" << nr << "):\n";
	std::cout << std::setprecision(16);
	std::cout << "time: " << std::get<0>(time_iters) << " ms iterations: " << std::get<1>(time_iters) << std::endl;
}

void him_2d_1025()
{
	// load net
	ln::Loader loader(at::kCUDA, at::kDouble);
	auto mionet = loader.load_net("./model/Poisson_2d_5000/mionet_poisson_2d.pt");
	// load data
	// auto data = ln::pickle_load("./data/Poisson_2d_size_1025_1025/data_10_1025_1025.bin").toTuple()->elements();
	auto data = ln::pickle_load("./data/Poisson_2d_size_1025_1025/data_10_1025_1025_random.bin").toTuple()->elements();
	// test
	int num = 1;
	auto k = data[0].toTensor()[num];
	auto f = data[1].toTensor()[num];
	auto u = data[2].toTensor()[num];
	auto mesh = torch::meshgrid({torch::linspace(0, 1, 1025, k.options()), torch::linspace(0, 1, 1025, k.options())}, "ij");
	at::Tensor x = torch::cat({mesh[0].reshape_symint({-1, 1}), mesh[1].reshape_symint({-1, 1})}, 1);
	std::string mode = "Mg6";		// GS; Mgx(<=10);
	double error_threshold = 1e-12; // 1e-12;
	int nr = 8;
	int split = 2; // reduce required GPU memory, otherwise overflow

	auto hybrid_solver = Hybrid_solver(mode, nr, error_threshold, split);
	std::tuple<double, double> time_iters = hybrid_solver.solve_poisson_2d_large(k, f, x, u, mionet);
	std::cout << "size: 1025 * 1025" << std::endl;
	std::cout << "hybrid iterative method (" + mode + " + MIONet nr=" << nr << "):\n";
	std::cout << std::setprecision(16);
	std::cout << "time: " << std::get<0>(time_iters) << " ms iterations: " << std::get<1>(time_iters) << std::endl;
}

void him_2d_boundary_500()
{
	// load net
	ln::Loader loader(at::kCUDA, at::kDouble);
	auto mionet = loader.load_net("./model/Poisson_2d_5000_boundary/mionet_poisson_2d_boundary_50.pt");
	// load data
	auto data = ln::pickle_load("./data/Poisson_2d_boundary_size_500_500/data_boundary_10_500_500.bin").toTuple()->elements();
	// test
	int num = 0;
	auto k = data[0].toTensor()[num];
	auto f = data[1].toTensor()[num];
	auto g = data[2].toTensor()[num];
	auto u = data[3].toTensor()[num];
	auto mesh = torch::meshgrid({torch::linspace(0, 1, 500, k.options()), torch::linspace(0, 1, 500, k.options())}, "ij");
	at::Tensor x = torch::cat({mesh[0].reshape_symint({-1, 1}), mesh[1].reshape_symint({-1, 1})}, 1);
	std::string mode = "GS";
	double error_threshold = 1e-12; // 1e-12;
	int nr = 1600;					// 1600
	int split = 1;

	auto hybrid_solver = Hybrid_solver(mode, nr, error_threshold, split);
	std::tuple<double, double> time_iters = hybrid_solver.solve_poisson_2d_boundary(k, f, g, x, u, mionet);
	std::cout << "size: 500 * 500 (inhomogeneous boundary condition)" << std::endl;
	std::cout << "hybrid iterative method (" + mode + " + MIONet nr=" << nr << "):\n";
	std::cout << std::setprecision(16);
	std::cout << "time: " << std::get<0>(time_iters) << " ms iterations: " << std::get<1>(time_iters) << std::endl;
}

int main()
{
	// device_test();

	// model_test();

	// data_test();

	him_2d();

	// him_2d_512();

	// him_2d_1024();

	// him_2d_1025();

	// him_2d_boundary_500();

	// gen::generate_data_large("./data/Poisson_2d_size_512_512/kf_10_512_512.pth", "./data/Poisson_2d_size_512_512/data_10_512_512.bin"); // 3.6 s
	// gen::generate_data_large("./data/Poisson_2d_size_1024_1024/kf_10_1024_1024.pth", "./data/Poisson_2d_size_1024_1024/data_10_1024_1024.bin"); // 26 s
	// gen::generate_data_large("./data/Poisson_2d_size_1025_1025/kf_10_1025_1025.pth", "./data/Poisson_2d_size_1025_1025/data_10_1025_1025.bin"); // 26 s
	// gen::generate_data_large("./data/Poisson_2d_size_1025_1025/kf_10_1025_1025_random.pth", "./data/Poisson_2d_size_1025_1025/data_10_1025_1025_random.bin");
	// gen::generate_data_boundary("./data/Poisson_2d_boundary_size_500_500/kfg_10_500_500.pth", "./data/Poisson_2d_boundary_size_500_500/data_boundary_10_500_500.bin");

	// Eigen::VectorXd u_him = Eigen::VectorXd::Zero(5);
	// u_him << 1, -2, 3, -4, 5;
	// std::cout << u_him << std::endl;
	// std::cout << u_him.cwiseAbs() << std::endl;
	// auto x = u_him.cwiseAbs().maxCoeff();
	// std::cout << x << std::endl;

	// auto x = torch::zeros({ 5 });
	// std::cout << x.device() << std::endl;
	// std::cout << x.to(at::kCUDA).device() << std::endl;
	// std::cout << x.device() << std::endl;

	// auto x = torch::arange(10000, at::kDouble).reshape_symint({100, 100});
	// auto y = x.index({ torch::round(torch::arange(5, at::kDouble)).to(at::kInt), torch::round(torch::arange(5, at::kDouble)).to(at::kInt) });
	// std::cout << y << std::endl;

	// Eigen::MatrixXd m = Eigen::MatrixXd::Random(3, 3);
	// std::cout << m << std::endl;
	// auto sm = m.sparseView();
	// Eigen::MatrixXd m(5, 5);
	// m << 1, 8, 2, 0, 0,
	//	0, 1, 0, 5, 0,
	//	0, 0, 0, 0, 0,
	//	0, 0, 7, 1, 0,
	//	0, 0, 0, 9, 1;
	// std::cout << m << std::endl;
	// Eigen::SparseMatrix<double, Eigen::RowMajor> sm = m.sparseView();
	// std::cout << sm.coeffs() << std::endl;
	// std::cout << sm.innerSize()<< std::endl;
	// std::cout << sm.outerSize() << std::endl;

	// std::string s = "abcd56";
	// std::string s1 = s.substr(4);
	// std::string s2 = s.substr(0, 4);
	// int i = std::stoi(s1);

	// std::cout << s1 << std::endl;
	// std::cout << s2 << std::endl;
	// std::cout << i << std::endl;

	// Eigen::SparseMatrix<double, Eigen::RowMajor> R(0, 1);
	// Eigen::SparseMatrix<double, Eigen::RowMajor> P(1, 0);
	// Eigen::VectorXd r(1);
	// r << 2.0;
	// Eigen::VectorXd u = Eigen::VectorXd::Zero((R * r).size());
	// std::cout << u << " " << u.size() << std::endl;
	// std::cout << u + R * u << std::endl;
	// std::cout << P.size() << std::endl;
	// std::cout << P * u << std::endl;
	// std::cout << (P * u).size() << std::endl;

	// std::cin.get();
}