// @author jpzxshi (jpz@pku.edu.cn)
#pragma once
#include <torch/torch.h>
#include <Eigen/Sparse>
#include "cln.hpp"

class Hybrid_solver
{
public:
	using SparseMatrix = Eigen::SparseMatrix<double, Eigen::RowMajor>;

	Hybrid_solver(std::string mode = "GS", int nr = 0, double error_threshold = 1e-14, int split=1, int m = 1);

	std::tuple<double, double> solve_poisson_2d(at::Tensor k, at::Tensor f, at::Tensor x, at::Tensor u, ln::Module net);

	std::tuple<double, double> solve_poisson_2d_large(at::Tensor k, at::Tensor f, at::Tensor x, at::Tensor u, ln::Module net);

	std::tuple<double, double> solve_poisson_2d_boundary(at::Tensor k, at::Tensor f, at::Tensor g, at::Tensor x, at::Tensor u, ln::Module net);

	//at::Tensor solve_poisson_1d(at::Tensor k, at::Tensor f, at::Tensor x, at::Tensor u, torch::jit::Module net);

	std::string get_mode();
	int get_nr();
	double get_error_threshold();
	int get_split();
	int get_m();

	void set_mode(std::string mode);
	void set_nr(int nr);
	void set_error_threshold(double error_threshold);
	void set_split(int split);
	void set_m(int m);

private:
	at::Tensor eye(int n, int k, c10::TensorOptions ops);
	at::Tensor predict(ln::Module net, at::Tensor k, at::Tensor f, at::Tensor x);
	at::Tensor insert(const at::Tensor& x, int size);
	at::Tensor insert(const at::Tensor& f, const at::Tensor& g, int size);
	at::Tensor matmul(const SparseMatrix& x, const at::Tensor& u);
	at::Tensor e2t(const Eigen::VectorXd& x, c10::Device device=at::kCPU);
	Eigen::VectorXd t2e(const at::Tensor& x);
	std::vector<SparseMatrix> R;
	std::vector<SparseMatrix> P;
	std::vector<SparseMatrix> A;
	SparseMatrix init_R(int N);
	SparseMatrix init_P(int N);
	void iterate_GS(Eigen::VectorXd& u, const SparseMatrix& A, const Eigen::VectorXd& b);
	Eigen::VectorXd v_cycle_recursion(const Eigen::VectorXd& r, int nu, int L);

	// using B  52 s
	void iterate_2d_B(at::Tensor& u, const at::Tensor& A, const at::Tensor& b, const at::Tensor& B);

	// using torch::tensor (currently double only, "GS" only) 
	// accessor + for 92 s, accessor + dot 56~62 s, non-accessor 170 s
	void iterate_2d_torch(at::Tensor& u, const at::Tensor& A, const at::Tensor& b, const std::string& mode, const double& w);

	// using vector 95 s
	void iterate_2d_vector(at::Tensor& u, const std::vector<std::vector<double>>& A, const std::vector<double>& b, const std::string& mode, const double& w);

	// using array overflow!!!
	void iterate_2d_array(at::Tensor& u, const double (*A)[10000], const double* b, const std::string& mode, const double& w);

	// using sparse  1.1 s (nr=300, cpu)  2.3 s (nr=300, cuda)  3.6 s (eigen + sparse + GS)
	void iterate_2d_sparse(at::Tensor& u, const SparseMatrix& A, const at::Tensor& b, const std::string& mode, const double& w);
	void iterate_2d_sparse(Eigen::VectorXd& u, const SparseMatrix& A, const Eigen::VectorXd& b, const double& w);
	
	// direct method: 9 ms (eigen + sparse + cholesky)  1.8 s (torch + dense + cholesky)

	std::string __mode;
	int __nr;
	double __error_threshold;
	int __split;
	int __m;
};