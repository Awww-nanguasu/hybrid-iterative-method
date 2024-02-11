// @author jpzxshi (jpz@pku.edu.cn)
#include "hybrid_solver.hpp"
#include <time.h>

using namespace torch::indexing;

Hybrid_solver::Hybrid_solver(std::string mode, int nr, double error_threshold, int split, int m)
	: __mode(mode), __nr(nr), __error_threshold(error_threshold), __split(split), __m(m)
{}

std::tuple<double, double> Hybrid_solver::solve_poisson_2d(at::Tensor k, at::Tensor f, at::Tensor x, at::Tensor u, ln::Module net)
{
	auto guard = ln::Inference_mode();
	c10::TensorOptions ops = k.options();//at::device(k.device()).dtype(k.dtype());
	int n = k.size(0) - 2;
	double h = 1.0 / (k.size(0) - 1.0);

	at::Tensor D = (
		(4.0 / 3.0) * k.index({ Slice(1, -1), Slice(1, -1) }) +
		(1.0 / 3.0) * (k.index({ Slice(None, -2), Slice(2, None) }) + k.index({ Slice(2, None), Slice(None, -2) })) +
		(1.0 / 2.0) * (k.index({ Slice(1, -1), Slice(2, None) }) + k.index({ Slice(1, -1), Slice(None, -2) }) +
			k.index({ Slice(2, None), Slice(1, -1) }) + k.index({ Slice(None, -2), Slice(1, -1) }))
		).ravel() * eye(pow(n, 2), 0, ops);

	at::Tensor U1 = (
		(-1.0 / 3.0) * (k.index({ Slice(1, -1), Slice(1, -2) }) + k.index({ Slice(1, -1), Slice(2, -1) })) + 
		(-1.0 / 6.0) * (k.index({ Slice(None, -2), Slice(2, -1) }) + k.index({ Slice(2, None), Slice(1, -2) }))
		);
	U1 = torch::cat({ U1, torch::zeros({n, 1}, ops) }, 1).reshape({ -1, 1 }) * eye(pow(n, 2), 1, ops);

	at::Tensor U2 = (
		(-1.0 / 3.0) * (k.index({ Slice(1, -2), Slice(1, -1) }) + k.index({ Slice(2, -1), Slice(1, -1) })) +
		(-1.0 / 6.0) * (k.index({ Slice(1, -2), Slice(2, None) }) + k.index({ Slice(2, -1), Slice(None, -2) }))
		);
	U2 = torch::cat({ U2, torch::zeros({1, n}, ops) }, 0).reshape({ -1, 1 }) * eye(pow(n, 2), n, ops);

	auto U = U1 + U2;
	auto L = U.t();
	auto A = D + U + L;

	auto b = pow(h, 2) * (
		(1.0 / 2.0) * f.index({ Slice(1, -1), Slice(1, -1) }) +
		(1.0 / 12.0) * (f.index({ Slice(2, None), Slice(1, -1) }) + f.index({ Slice(1, -1), Slice(2, None) }) + f.index({ Slice(None, -2), Slice(2, None) }) +
			f.index({ Slice(None, -2), Slice(1, -1) }) + f.index({ Slice(1, -1), Slice(None, -2) }) + f.index({ Slice(2, None), Slice(None, -2) }))
		).ravel();

	u = u.index({ Slice(1, -1), Slice(1, -1) }).ravel();

	double w = 1.0;

	std::cout << "prepare auxiliary matrices ..." << std::endl;
	//////////////////////////////////////////////////////////////////// B
	//at::Tensor B;
	//if (__mode == "Jacobi")
	//{
	//	B = w * torch::inverse(D);
	//}
	//else if (__mode == "GS")
	//{
	//	B = torch::inverse(1.0 / w * D + L);
	//}
	///////////////////////////////////////////////////////////////////// vector
	//std::vector<double> b_v(b.size(0));
	//std::vector<std::vector<double>> A_v(A.size(0), std::vector<double>(A.size(1)));
	//auto _A = A.accessor<double, 2>();
	//auto _b = b.accessor<double, 1>();
	//for (auto i = 0; i != _A.size(0); ++i)
	//{
	//	b_v[i] = _b[i];
	//	for (auto j = 0; j != _A.size(1); ++j)
	//	{
	//		A_v[i][j] = _A[i][j];
	//	}
	//}
	//////////////////////////////////////////////////////////////////// array overflow !!!
	//double b_a[10000];
	//double A_a[10000][10000];
	//auto _A = A.accessor<double, 2>();
	//auto _b = b.accessor<double, 1>();
	//for (auto i = 0; i != _A.size(0); ++i)
	//{
	//	b_a[i] = _b[i];
	//	for (auto j = 0; j != _A.size(1); ++j)
	//	{
	//		A_a[i][j] = _A[i][j];
	//	}
	//}
	//////////////////////////////////////////////////////////////////// sparse matrix
	SparseMatrix A_s(A.size(0), A.size(1));
	Eigen::VectorXd b_s(b.size(0));
	auto _A = A.accessor<double, 2>();
	auto _b = b.accessor<double, 1>();
	for (auto i = 0; i != _A.size(0); ++i)
	{
		b_s[i] = _b[i];
		for (auto j = 0; j != _A.size(1); ++j)
		{
			if (_A[i][j] != 0)
			{ 
				A_s.insert(i, j) = _A[i][j]; 
			}
		}
	}
	A_s.makeCompressed();
	// direct method
	//std::cout << "direct method (cholesky + sparse): " << std::endl;
	//auto start_s = clock();
	//Eigen::SimplicialCholesky<Eigen::SparseMatrix<double, Eigen::RowMajor>> chol(A_s);
	//Eigen::VectorXd u_s = chol.solve(b_s);
	//auto end_s = clock();
	//at::Tensor u_s_tensor = torch::zeros_like(u);
	//auto _u_s_tensor = u_s_tensor.accessor<double, 1>();
	//for (auto i = 0; i != u.size(0); ++i)
	//{
	//	_u_s_tensor[i] = u_s[i];
	//}
	//std::cout << "error: " << torch::max(torch::abs(u_s_tensor - u)) << std::endl;
	//std::cout << end_s - start_s << " ms" << std::endl;

	//std::cout << "direct method (cholesky + dense): " << std::endl;
	//auto start_d = clock();
	//auto chol = torch::linalg::cholesky(A);
	//at::Tensor u_d = torch::cholesky_solve(b.reshape_symint({-1, 1}), chol, false);
	//auto end_d = clock();
	//std::cout << "error: " << torch::max(torch::abs(u_d.ravel() - u)) << std::endl;
	//std::cout << end_d - start_d << " ms" << std::endl;

	std::cout << "ready for iteration!" << std::endl;

	auto start = clock();
	std::vector<int> its(__m);
	for (auto i = 0; i != __m; ++i)
	{
		auto u_him = torch::zeros_like(u);
		auto C_norm = torch::max(torch::abs(u_him - u));
		while (C_norm.item().toDouble() > __error_threshold)
		{
			if (__nr > 0 && its[i] % __nr == 0)
			{
				at::Tensor resi;
				if (its[i] == 0)
				{
					resi = f;
				}
				else
				{
					resi = (b - torch::matmul(A, u_him)).reshape_symint({ n, n }) / (pow(h, 2));
					resi = torch::cat({ resi.index({Slice(), Slice(None, 1)}), resi, resi.index({Slice(), Slice(-1, None)}) }, 1);
					resi = torch::cat({ resi.index({Slice(None, 1), Slice()}), resi, resi.index({Slice(-1, None), Slice()}) }, 0);
				}
				u_him = u_him + predict(net, k.ravel(), resi.ravel(), x).reshape_symint({ n + 2, n + 2 }).index({ Slice(1, -1), Slice(1, -1) }).ravel();
			}
			else
			{
				//iterate_2d_B(u_him, A, b, B); // using B
				//iterate_2d_torch(u_him, A, b, __mode, w); // using torch::tensor
				//iterate_2d_vector(u_him, A_v, b_v, __mode, w); // using vector for A, b 
				//iterate_2d_array(u_him, A_a, b_a, __mode, w); // overflow
				iterate_2d_sparse(u_him, A_s, b, __mode, w);

				//u_him = u_him + torch::matmul(B, r);  // using B
			}
			C_norm = torch::max(torch::abs(u_him - u));
			its[i] = its[i] + 1;
			//std::cout << C_norm.item().toDouble() << std::endl;
		}
	}
	auto end = clock();

	double time = (double)(end - start) / __m;
	double iterations = torch::mean(torch::tensor(its).to(at::kDouble)).item().toDouble();
	return std::make_tuple(time, iterations);
}

std::string Hybrid_solver::get_mode()
{
	return __mode;
}

int Hybrid_solver::get_nr()
{
	return __nr;
}

double Hybrid_solver::get_error_threshold()
{
	return __error_threshold;
}

int Hybrid_solver::get_split()
{
	return __split;
}

int Hybrid_solver::get_m()
{
	return __m;
}

void Hybrid_solver::set_mode(std::string mode)
{
	__mode = mode;
}

void Hybrid_solver::set_nr(int nr)
{
	__nr = nr;
}

void Hybrid_solver::set_error_threshold(double error_threshold)
{
	__error_threshold = error_threshold;
}

void Hybrid_solver::set_split(int split)
{
	__split = split;
}

void Hybrid_solver::set_m(int m)
{
	__m = m;
}

at::Tensor Hybrid_solver::eye(int n, int k, c10::TensorOptions ops)
{
	at::Tensor I = torch::roll(torch::eye(n, ops), k, 1);
	at::Tensor modify = torch::ones({ n }, ops);
	auto slice = k >= 0 ? torch::indexing::Slice(torch::indexing::None, k) : torch::indexing::Slice(k, torch::indexing::None);
	modify.index_put_({ slice }, 0);
	return I * modify;
}

at::Tensor Hybrid_solver::predict(ln::Module net, at::Tensor k, at::Tensor f, at::Tensor x)
{
	//c10::TensorOptions ops_in = at::device(f.device()).dtype(f.dtype());
	c10::TensorOptions ops_net = at::device(net.device()).dtype(net.dtype());
	k = k.to(ops_net); f = f.to(ops_net); x = x.to(ops_net);
	std::vector<int> index(__split);
	for (auto i = 0; i != __split; ++i)
	{
		index[i] = (int)(x.size(0) / (double)(__split) * i);
	}
	index.push_back(x.size(0));
	std::vector<at::Tensor> res(__split);
	for (auto i = 0; i != __split; ++i)
	{
		res[i] = net.forward({ std::make_tuple(k, f, x.index({Slice(index[i], index[i + 1])})) }).toTensor();
	}
	return torch::cat(res, 0);
	//return net.forward({ std::make_tuple(k.to(ops_net), f.to(ops_net), x.to(ops_net)) }).toTensor();//.to(ops_in);
}

void Hybrid_solver::iterate_2d_B(at::Tensor& u, const at::Tensor& A, const at::Tensor& b, const at::Tensor& B)
{
	u.index_put_({ Slice() }, u + torch::matmul(B, b - torch::matmul(A, u)));
}

void Hybrid_solver::iterate_2d_torch(at::Tensor& u, const at::Tensor& A, const at::Tensor& b, const std::string& mode, const double& w)
{
	auto _u = u.accessor<double, 1>();
	auto _A = A.accessor<double, 2>();
	auto _b = b.accessor<double, 1>();

	auto n = u.size(0);
	for (int i = 0; i != n; ++i)
	{
		_u[i] = _u[i] + (w / _A[i][i]) * (_b[i] - torch::dot(A[i], u).item().toDouble()); // 60s

		//double sum = 0;
		//for (int j = 0; j != n; ++j)
		//{
		//	sum += _A[i][j] * _u[j];
		//}
		//_u[i] = _u[i] + (w / _A[i][i]) * (_b[i] - sum);                        // 92s
		
		//u[i] = u[i] + (w / A[i][i]) * (b[i] - torch::dot(A[i], u));            // 170s
		//u.index_put_({i}, u[i] + (w / A[i][i]) * (b[i] - torch::dot(A[i], u)));// 170s
	}
}

void Hybrid_solver::iterate_2d_vector(at::Tensor& u, const std::vector<std::vector<double>>& A, const std::vector<double>& b, const std::string& mode, const double& w)
{
	auto _u = u.accessor<double, 1>();
	auto n = _u.size(0);
	for (int i = 0; i != n; ++i)
	{
		double sum = 0;
		for (int j = 0; j != n; ++j)
		{
			sum += A[i][j] * _u[j];
		}
		_u[i] = _u[i] + (w / A[i][i]) * (b[i] - sum); 
	}
}

void Hybrid_solver::iterate_2d_array(at::Tensor& u, const double (*A)[10000], const double* b, const std::string& mode, const double& w)
{
	auto _u = u.accessor<double, 1>();
	auto n = _u.size(0);
	for (int i = 0; i != n; ++i)
	{
		double sum = 0;
		for (int j = 0; j != n; ++j)
		{
			sum += A[i][j] * _u[j];
		}
		_u[i] = _u[i] + (w / A[i][i]) * (b[i] - sum);
	}
}

void Hybrid_solver::iterate_2d_sparse(at::Tensor& u, const SparseMatrix& A, const at::Tensor& b, const std::string& mode, const double& w)
{
	auto _u = u.accessor<double, 1>();
	auto _b = b.accessor<double, 1>();
	auto n = _u.size(0);
	for (int i = 0; i != n; ++i)
	{
		double sum = 0;
		double A_ii = 0;
		for (SparseMatrix::InnerIterator it(A, i); it; ++it)
		{
			sum += it.value() * _u[it.col()];
			if (it.col() == i)
			{
				A_ii = it.value();
			}
		}
		_u[i] = _u[i] + (w / A_ii) * (_b[i] - sum);
	}
}

void Hybrid_solver::iterate_GS(Eigen::VectorXd& u, const SparseMatrix& A, const Eigen::VectorXd& b)
{
	auto n = u.size();
	for (int i = 0; i != n; ++i)
	{
		double sum = 0;
		double A_ii = 0;
		for (SparseMatrix::InnerIterator it(A, i); it; ++it)
		{
			sum += it.value() * u(it.col());
			if (it.col() == i)
			{
				A_ii = it.value();
			}
		}
		u(i) += (b(i) - sum) / A_ii;
	}
}

Eigen::VectorXd Hybrid_solver::v_cycle_recursion(const Eigen::VectorXd& r, int nu, int L)
{
	Eigen::VectorXd r_r = R[L] * r;
	Eigen::VectorXd u = Eigen::VectorXd::Zero(r_r.size());
	for (int i = 0; i != nu; ++i)
	{
		iterate_GS(u, A[L], r_r);
	}

	if (L + 1 != R.size())
	{
		u += v_cycle_recursion(r_r - A[L] * u, nu, L + 1);
	}

	for (int i = 0; i != nu; ++i)
	{
		iterate_GS(u, A[L], r_r);
	}

	Eigen::VectorXd u_p = P[L] * u;
	return u_p;
}

void Hybrid_solver::iterate_2d_sparse(Eigen::VectorXd& u, const SparseMatrix& A, const Eigen::VectorXd& b, const double& w)
{
	if (__mode == "GS")
	{
		auto n = u.size();
		for (int i = 0; i != n; ++i)
		{
			double sum = 0;
			double A_ii = 0;
			for (SparseMatrix::InnerIterator it(A, i); it; ++it)
			{
				sum += it.value() * u(it.col());
				if (it.col() == i)
				{
					A_ii = it.value();
				}
			}
			u(i) += (w / A_ii) * (b(i) - sum);
		}
	}
	else if (__mode.substr(0, 2) == "Mg")
	{
		int nu = 5;

		for (int i = 0; i != nu; ++i) {
			iterate_GS(u, A, b);
		}

		u += v_cycle_recursion(b - A * u, nu, 0);

		for (int i = 0; i != nu; ++i) {
			iterate_GS(u, A, b);
		}
	}
}


std::tuple<double, double> Hybrid_solver::solve_poisson_2d_large(at::Tensor k, at::Tensor f, at::Tensor x, at::Tensor u, ln::Module net)
{
	auto guard = ln::Inference_mode();
	c10::TensorOptions ops = k.options();
	int n = k.size(0) - 2;
	int size = pow(n, 2);
	double h = 1.0 / (k.size(0) - 1.0);
	
	// D tensor
	at::Tensor D = (
		(4.0 / 3.0) * k.index({ Slice(1, -1), Slice(1, -1) }) +
		(1.0 / 3.0) * (k.index({ Slice(None, -2), Slice(2, None) }) + k.index({ Slice(2, None), Slice(None, -2) })) +
		(1.0 / 2.0) * (k.index({ Slice(1, -1), Slice(2, None) }) + k.index({ Slice(1, -1), Slice(None, -2) }) +
			k.index({ Slice(2, None), Slice(1, -1) }) + k.index({ Slice(None, -2), Slice(1, -1) }))
		).ravel();
	// D sparse
	SparseMatrix D_s(size, size);
	auto _D = D.accessor<double, 1>();
	for (int i = 0; i != size; ++i)
	{
		D_s.insert(i, i) = _D[i];
	}
	// U1 tensor
	at::Tensor U1 = (
		(-1.0 / 3.0) * (k.index({ Slice(1, -1), Slice(1, -2) }) + k.index({ Slice(1, -1), Slice(2, -1) })) +
		(-1.0 / 6.0) * (k.index({ Slice(None, -2), Slice(2, -1) }) + k.index({ Slice(2, None), Slice(1, -2) }))
		);
	U1 = torch::cat({ U1, torch::zeros({n, 1}, ops) }, 1).ravel();
	// U1 sparse
	SparseMatrix U1_s(size, size);
	auto _U1 = U1.accessor<double, 1>();
	for (int i = 0; i != size - 1; ++i)
	{
		U1_s.insert(i, i + 1) = _U1[i];
	}
	// U2 tensor
	at::Tensor U2 = (
		(-1.0 / 3.0) * (k.index({ Slice(1, -2), Slice(1, -1) }) + k.index({ Slice(2, -1), Slice(1, -1) })) +
		(-1.0 / 6.0) * (k.index({ Slice(1, -2), Slice(2, None) }) + k.index({ Slice(2, -1), Slice(None, -2) }))
		);
	U2 = torch::cat({ U2, torch::zeros({1, n}, ops) }, 0).ravel();
	// U2 sparse
	SparseMatrix U2_s(size, size);
	auto _U2 = U2.accessor<double, 1>();
	for (int i = 0; i != size - n; ++i)
	{
		U2_s.insert(i, i + n) = _U2[i];
	}

	SparseMatrix U_s = U1_s + U2_s;
	SparseMatrix L_s = SparseMatrix(U_s.transpose());
	SparseMatrix A_s = D_s + U_s + L_s;
	A_s.makeCompressed();
	
	// b tensor
	auto b = pow(h, 2) * (
		(1.0 / 2.0) * f.index({ Slice(1, -1), Slice(1, -1) }) +
		(1.0 / 12.0) * (f.index({ Slice(2, None), Slice(1, -1) }) + f.index({ Slice(1, -1), Slice(2, None) }) + f.index({ Slice(None, -2), Slice(2, None) }) +
			f.index({ Slice(None, -2), Slice(1, -1) }) + f.index({ Slice(1, -1), Slice(None, -2) }) + f.index({ Slice(2, None), Slice(None, -2) }))
		).ravel();
	// b eigen vector
	Eigen::VectorXd b_s(size);
	auto _b = b.accessor<double, 1>();
	for (int i = 0; i != size; ++i)
	{
		b_s[i] = _b[i];
	}
	// u tensor
	u = u.index({ Slice(1, -1), Slice(1, -1) }).ravel();
	// u eigen vector
	Eigen::VectorXd u_s(size);
	auto _u = u.accessor<double, 1>();
	for (int i = 0; i != size; ++i)
	{
		u_s[i] = _u[i];
	}

	double w = 1.0;

	std::cout << "prepare auxiliary matrices ..." << std::endl;
	auto net_size = 100;
	at::Tensor k_insert = insert(k, net_size).ravel().to(net.device());
	at::Tensor f_insert = insert(f, net_size).ravel().to(net.device());
	at::Tensor x_temp = x.to(net.device());
	
	if (__mode.substr(0,2) == "Mg")
	{
		int L = std::stoi(__mode.substr(2));
		int N = n + 1;
		R = std::vector<SparseMatrix>(L - 1);
		P = std::vector<SparseMatrix>(L - 1);
		A = std::vector<SparseMatrix>(L - 1);
		for (int i = 0; i != L - 1; ++i)
		{
			R[i] = init_R(N);
			P[i] = init_P(N);
			if (i == 0)
			{
				A[i] = R[i] * A_s * P[i];
			}
			else
			{
				A[i] = R[i] * A[i - 1] * P[i];
			}
			N = N / 2;
		}
	}

	std::cout << "ready for iteration!" << std::endl;

	auto start = clock();
	std::vector<int> its(__m);
	for (auto i = 0; i != __m; ++i)
	{
		//auto u_him = torch::zeros_like(u);
		Eigen::VectorXd u_him = Eigen::VectorXd::Zero(size);
		//auto C_norm = torch::max(torch::abs(u_him - u));
		auto C_norm = (u_him - u_s).cwiseAbs().maxCoeff();
		//while (C_norm.item().toDouble() > __error_threshold)
		while (C_norm > __error_threshold)
		{
			if (__nr > 0 && its[i] % __nr == 0)
			{
				//std::cout << "mionet predicting ..." << std::endl;
				at::Tensor resi_insert;
				if (its[i] == 0)
				{
					resi_insert = f_insert;
				}
				else
				{
					//resi = (b - matmul(A_s, u_him)).reshape_symint({n, n}) / (pow(h, 2));
					resi_insert = e2t(b_s - A_s * u_him, net.device()).reshape_symint({ n, n }) / (pow(h, 2));
					resi_insert = torch::cat({ resi_insert.index({Slice(), Slice(None, 1)}), resi_insert, resi_insert.index({Slice(), Slice(-1, None)}) }, 1);
					resi_insert = torch::cat({ resi_insert.index({Slice(None, 1), Slice()}), resi_insert, resi_insert.index({Slice(-1, None), Slice()}) }, 0);
					resi_insert = insert(resi_insert, net_size).ravel();
				}
				u_him += t2e(predict(net, k_insert, resi_insert, x_temp).reshape_symint({ n + 2, n + 2 }).index({ Slice(1, -1), Slice(1, -1) }).ravel());
			}
			else
			{
				//iterate_2d_B(u_him, A, b, B); // using B
				//iterate_2d_torch(u_him, A, b, __mode, w); // using torch::tensor
				//iterate_2d_vector(u_him, A_v, b_v, __mode, w); // using vector for A, b 
				//iterate_2d_array(u_him, A_a, b_a, __mode, w); // overflow
				//iterate_2d_sparse(u_him, A_s, b, __mode, w);
				iterate_2d_sparse(u_him, A_s, b_s, w);

				//u_him = u_him + torch::matmul(B, r);  // using B
			}
			C_norm = (u_him - u_s).cwiseAbs().maxCoeff();
			its[i] = its[i] + 1;
			//std::cout << C_norm.item().toDouble() << std::endl;
			//std::cout << C_norm << std::endl;
		}
	}
	auto end = clock();

	double time = (double)(end - start) / __m;
	double iterations = torch::mean(torch::tensor(its).to(at::kDouble)).item().toDouble();
	return std::make_tuple(time, iterations);
}

at::Tensor Hybrid_solver::insert(const at::Tensor& x, int size)
{
	auto n = x.size(0);
	//double r = (n - 1.0) / (size - 1.0);
	auto mesh = torch::meshgrid({ torch::linspace(0, n - 1, size, x.options()), torch::linspace(0, n - 1, size, x.options()) }, "ij");
	return x.index({ torch::round(mesh[0]).to(at::kInt), torch::round(mesh[1]).to(at::kInt) });
	//at::Tensor x_index = mesh[0], y_index = mesh[1];
	//auto x_trunc = torch::trunc(x_index), x_frac = torch::frac(x_index);
	//auto y_trunc = torch::trunc(y_index), y_frac = torch::frac(y_index);
}

at::Tensor Hybrid_solver::matmul(const SparseMatrix& x, const at::Tensor& u)
{
	auto size = u.size(0);
	// u tensor -> eigen
	Eigen::VectorXd u_s(size);
	auto _u = u.accessor<double, 1>();
	for (int i = 0; i != size; ++i)
	{
		u_s[i] = _u[i];
	}
	// mul
	Eigen::VectorXd b_s(x * u_s);
	// b eigen -> tensor
	auto b = torch::zeros_like(u);
	auto _b = b.accessor<double, 1>();
	for (int i = 0; i != size; ++i)
	{
		_b[i] = b_s[i];
	}
	return b;
}

at::Tensor Hybrid_solver::e2t(const Eigen::VectorXd& x, c10::Device device)
{
	//std::cout << "start e2t .." << std::endl;
	auto size = x.size();
	auto x_torch = torch::zeros(size, at::device(at::kCPU).dtype(at::kDouble));
	auto _x_torch = x_torch.accessor<double, 1>();
	for (auto i = 0; i != size; ++i)
	{
		_x_torch[i] = x(i);
	}
	auto res = x_torch.to(device);
	//std::cout << "end e2t .." << std::endl;
	return res;
}

Eigen::VectorXd Hybrid_solver::t2e(const at::Tensor& x)
{
	//torch::cuda::synchronize();
	//std::cout << "start t2e .." << std::endl;
	auto x_temp = x.to(at::kCPU);
	auto size = x_temp.size(0);
	Eigen::VectorXd x_eigen(size);
	auto _x = x_temp.accessor<double, 1>();
	for (auto i = 0; i != size; ++i)
	{
		x_eigen(i) = _x[i];
	}
	//std::cout << "end t2e .." << std::endl;
	return x_eigen;
}

Hybrid_solver::SparseMatrix Hybrid_solver::init_R(int N)
{
	N = N / 2;
	SparseMatrix R(pow(N - 1, 2), pow(2 * N - 1, 2));
	for (int i = 0; i != N - 1; ++i)
	{
		for (int j = 0; j != N - 1; ++j)
		{
			R.insert(i * (N - 1) + j, i * (4 * N - 2) + j * 2) = 0.5;
			R.insert(i * (N - 1) + j, i * (4 * N - 2) + j * 2 + 1) = 0.5;
			R.insert(i * (N - 1) + j, i * (4 * N - 2) + j * 2 + 2 * N - 1) = 0.5;
			R.insert(i * (N - 1) + j, i * (4 * N - 2) + j * 2 + 2 * N) = 1.0;
			R.insert(i * (N - 1) + j, i * (4 * N - 2) + j * 2 + 2 * N + 1) = 0.5;
			R.insert(i * (N - 1) + j, i * (4 * N - 2) + j * 2 + 4 * N - 1) = 0.5;
			R.insert(i * (N - 1) + j, i * (4 * N - 2) + j * 2 + 4 * N) = 0.5;
		}
	}
	return R;
}

Hybrid_solver::SparseMatrix Hybrid_solver::init_P(int N)
{
	N = N / 2;
	SparseMatrix P(pow(2 * N - 1, 2), pow(N - 1, 2));
	for (int i = 0; i != N - 1; ++i)
	{
		for (int j = 0; j != N - 1; ++j)
		{
			P.insert(i * (4 * N - 2) + j * 2, i * (N - 1) + j) = 0.5;
			P.insert(i * (4 * N - 2) + j * 2 + 1, i * (N - 1) + j) = 0.5;
			P.insert(i * (4 * N - 2) + j * 2 + 2 * N - 1, i * (N - 1) + j) = 0.5;
			P.insert(i * (4 * N - 2) + j * 2 + 2 * N, i * (N - 1) + j) = 1.0;
			P.insert(i * (4 * N - 2) + j * 2 + 2 * N + 1, i * (N - 1) + j) = 0.5;
			P.insert(i * (4 * N - 2) + j * 2 + 4 * N - 1, i * (N - 1) + j) = 0.5;
			P.insert(i * (4 * N - 2) + j * 2 + 4 * N, i * (N - 1) + j) = 0.5;
		}
	}
	return P;
}

std::tuple<double, double> Hybrid_solver::solve_poisson_2d_boundary(at::Tensor k, at::Tensor f, at::Tensor g, at::Tensor x, at::Tensor u, ln::Module net)
{
	auto guard = ln::Inference_mode();
	c10::TensorOptions ops = k.options();
	int n = k.size(0);
	int size = pow(n, 2);
	int n_in = k.size(0) - 2;
	int size_in = pow(n_in, 2);
	double h = 1.0 / (k.size(0) - 1.0);

	SparseMatrix A_s(size, size);
	at::Tensor index = torch::arange(size).reshape_symint({ n, n });

	// D tensor
	at::Tensor D = (
		(4.0 / 3.0) * k.index({ Slice(1, -1), Slice(1, -1) }) +
		(1.0 / 3.0) * (k.index({ Slice(None, -2), Slice(2, None) }) + k.index({ Slice(2, None), Slice(None, -2) })) +
		(1.0 / 2.0) * (k.index({ Slice(1, -1), Slice(2, None) }) + k.index({ Slice(1, -1), Slice(None, -2) }) +
			k.index({ Slice(2, None), Slice(1, -1) }) + k.index({ Slice(None, -2), Slice(1, -1) }))
		).ravel();
	// D sparse
	SparseMatrix D_in(size_in, size_in);
	auto _D = D.accessor<double, 1>();
	for (int i = 0; i != size_in; ++i)
	{
		D_in.insert(i, i) = _D[i];
	}
	// U1 tensor
	at::Tensor U1 = (
		(-1.0 / 3.0) * (k.index({ Slice(1, -1), Slice(1, -2) }) + k.index({ Slice(1, -1), Slice(2, -1) })) +
		(-1.0 / 6.0) * (k.index({ Slice(None, -2), Slice(2, -1) }) + k.index({ Slice(2, None), Slice(1, -2) }))
		);
	U1 = torch::cat({ U1, torch::zeros({n_in, 1}, ops) }, 1).ravel();
	// U1 sparse
	SparseMatrix U1_in(size_in, size_in);
	auto _U1 = U1.accessor<double, 1>();
	for (int i = 0; i != size_in - 1; ++i)
	{
		U1_in.insert(i, i + 1) = _U1[i];
	}
	// U2 tensor
	at::Tensor U2 = (
		(-1.0 / 3.0) * (k.index({ Slice(1, -2), Slice(1, -1) }) + k.index({ Slice(2, -1), Slice(1, -1) })) +
		(-1.0 / 6.0) * (k.index({ Slice(1, -2), Slice(2, None) }) + k.index({ Slice(2, -1), Slice(None, -2) }))
		);
	U2 = U2.ravel();
	// U2 sparse
	SparseMatrix U2_in(size_in, size_in);
	auto _U2 = U2.accessor<double, 1>();
	for (int i = 0; i != size_in - n_in; ++i)
	{
		U2_in.insert(i, i + n_in) = _U2[i];
	}

	SparseMatrix U_in = U1_in + U2_in;
	SparseMatrix L_in = SparseMatrix(U_in.transpose());
	SparseMatrix A_in = D_in + U_in + L_in;
	A_in.makeCompressed();
	// in
	at::Tensor index_in = index.index({ Slice(1, -1), Slice(1, -1) }).ravel();
	auto _index_in = index_in.accessor<int64_t, 1>();
	for (int i = 0; i != size_in; ++i)
	{
		for (SparseMatrix::InnerIterator it(A_in, i); it; ++it)
		{
			A_s.insert(_index_in[i], _index_in[it.col()]) = it.value();
		}
	}
	// up
	at::Tensor b_up = ((-1.0 / 3.0) * (k.index({ 1, Slice(1, -1) }) + k.index({ 0, Slice(1, -1) }))
		+ (-1.0 / 6.0) * (k.index({ 0, Slice(None, -2) }) + k.index({ 1, Slice(2, None) }))).ravel();
	at::Tensor index_up_r = index.index({ 1, Slice(1, -1) }).ravel();
	at::Tensor index_up_c = index.index({ 0, Slice(1, -1) }).ravel();
	auto _b_up = b_up.accessor<double, 1>();
	auto _index_up_r = index_up_r.accessor<int64_t, 1>();
	auto _index_up_c = index_up_c.accessor<int64_t, 1>();
	for (int i = 0; i != n_in; ++i)
	{
		A_s.insert(_index_up_r[i], _index_up_c[i]) = _b_up[i];
	}
	// left
	at::Tensor b_left = ((-1.0 / 3.0) * (k.index({ Slice(1, -1), 1 }) + k.index({ Slice(1, -1), 0 }))
		+ (-1.0 / 6.0) * (k.index({ Slice(None, -2), 0 }) + k.index({ Slice(2, None), 1 }))).ravel();
	at::Tensor index_left_r = index.index({ Slice(1, -1), 1 }).ravel();
	at::Tensor index_left_c = index.index({ Slice(1, -1), 0 }).ravel();
	auto _b_left = b_left.accessor<double, 1>();
	auto _index_left_r = index_left_r.accessor<int64_t, 1>();
	auto _index_left_c = index_left_c.accessor<int64_t, 1>();
	for (int i = 0; i != n_in; ++i)
	{
		A_s.insert(_index_left_r[i], _index_left_c[i]) = _b_left[i];
	}
	// bottom
	at::Tensor b_bottom = ((-1.0 / 3.0) * (k.index({ -2, Slice(1, -1) }) + k.index({ -1, Slice(1, -1) }))
		+ (-1.0 / 6.0) * (k.index({ -2, Slice(None, -2) }) + k.index({ -1, Slice(2, None) }))).ravel();
	at::Tensor index_bottom_r = index.index({ -2, Slice(1, -1) }).ravel();
	at::Tensor index_bottom_c = index.index({ -1, Slice(1, -1) }).ravel();
	auto _b_bottom = b_bottom.accessor<double, 1>();
	auto _index_bottom_r = index_bottom_r.accessor<int64_t, 1>();
	auto _index_bottom_c = index_bottom_c.accessor<int64_t, 1>();
	for (int i = 0; i != n_in; ++i)
	{
		A_s.insert(_index_bottom_r[i], _index_bottom_c[i]) = _b_bottom[i];
	}
	// right
	at::Tensor b_right = ((-1.0 / 3.0) * (k.index({ Slice(1, -1), -2 }) + k.index({ Slice(1, -1), -1 }))
		+ (-1.0 / 6.0) * (k.index({ Slice(None, -2), -2 }) + k.index({ Slice(2, None), -1 }))).ravel();
	at::Tensor index_right_r = index.index({ Slice(1, -1), -2 }).ravel();
	at::Tensor index_right_c = index.index({ Slice(1, -1), -1 }).ravel();
	auto _b_right = b_right.accessor<double, 1>();
	auto _index_right_r = index_right_r.accessor<int64_t, 1>();
	auto _index_right_c = index_right_c.accessor<int64_t, 1>();
	for (int i = 0; i != n_in; ++i)
	{
		A_s.insert(_index_right_r[i], _index_right_c[i]) = _b_right[i];
	}
	// boundary
	at::Tensor index_boundary = torch::cat({ index.index({0}), index.index({Slice(1, -1), Slice(0, None, n - 1)}).ravel(), index.index({-1}) }, 0);
	auto _index_boundary = index_boundary.accessor<int64_t, 1>();
	for (int i = 0; i != index_boundary.size(0); ++i)
	{
		A_s.insert(_index_boundary[i], _index_boundary[i]) = 1.0;
	}

	A_s.makeCompressed();

	// b tensor
	at::Tensor b = pow(h, 2) * (
		(1.0 / 2.0) * f.index({ Slice(1, -1), Slice(1, -1) }) +
		(1.0 / 12.0) * (f.index({ Slice(2, None), Slice(1, -1) }) + f.index({ Slice(1, -1), Slice(2, None) }) + f.index({ Slice(None, -2), Slice(2, None) }) +
			f.index({ Slice(None, -2), Slice(1, -1) }) + f.index({ Slice(1, -1), Slice(None, -2) }) + f.index({ Slice(2, None), Slice(None, -2) }))
		);
	b = torch::cat({ torch::flip(g.index({ Slice(3 * n - 2, None) }), {0}).reshape_symint({1, n_in}), b, g.index({Slice(n, 2 * n - 2)}).reshape_symint({1, n_in}) }, 0);
	b = torch::cat({ g.index({Slice(None, n)}).reshape_symint({n, 1}), b, torch::flip(g.index({Slice(2 * n - 2, 3 * n - 2)}), {0}).reshape_symint({n, 1}) }, 1);
	b = b.ravel();
	// b eigen vector
	Eigen::VectorXd b_s(size);
	auto _b = b.accessor<double, 1>();
	for (int i = 0; i != size; ++i)
	{
		b_s[i] = _b[i];
	}

	// u tensor
	u = u.ravel();
	// u eigen vector
	Eigen::VectorXd u_s(size);
	auto _u = u.accessor<double, 1>();
	for (int i = 0; i != size; ++i)
	{
		u_s[i] = _u[i];
	}

	double w = 1.0;

	std::cout << "prepare auxiliary matrices ..." << std::endl;
	auto net_size = 100;
	at::Tensor k_insert = insert(k, net_size).ravel().to(net.device());
	at::Tensor fg_insert = insert(f, g, net_size).ravel().to(net.device());
	at::Tensor x_temp = x.to(net.device());

	std::cout << "ready for iteration!" << std::endl;

	auto start = clock();
	std::vector<int> its(__m);
	for (auto i = 0; i != __m; ++i)
	{
		Eigen::VectorXd u_him = Eigen::VectorXd::Zero(size);
		auto C_norm = (u_him - u_s).cwiseAbs().maxCoeff();
		while (C_norm > __error_threshold)
		{
			if (__nr > 0 && its[i] % __nr == 0)
			{
				//std::cout << "mionet predicting ..." << std::endl;
				at::Tensor resi_insert;
				if (its[i] == 0)
				{
					resi_insert = fg_insert;
				}
				else
				{
					resi_insert = e2t(b_s - A_s * u_him, net.device()).reshape_symint({ n, n });
					resi_insert.index_put_({ Slice(1, -1), Slice(1, -1) }, resi_insert.index({ Slice(1, -1), Slice(1, -1) }) / pow(h, 2));
					resi_insert = insert(resi_insert, net_size).ravel();
				}
				u_him += t2e(predict(net, k_insert, resi_insert, x_temp));
			}
			else
			{
				iterate_2d_sparse(u_him, A_s, b_s, w);
			}
			C_norm = (u_him - u_s).cwiseAbs().maxCoeff();
			its[i] = its[i] + 1;
			//std::cout << C_norm << std::endl;
		}
	}
	auto end = clock();

	double time = (double)(end - start) / __m;
	double iterations = torch::mean(torch::tensor(its).to(at::kDouble)).item().toDouble();
	return std::make_tuple(time, iterations);
}

at::Tensor Hybrid_solver::insert(const at::Tensor& f, const at::Tensor& g, int size)
{
	auto n = f.size(0);
	auto mesh = torch::meshgrid({ torch::linspace(0, n - 1, size, f.options()), torch::linspace(0, n - 1, size, f.options()) }, "ij");
	auto fg = f.index({ torch::round(mesh[0]).to(at::kInt), torch::round(mesh[1]).to(at::kInt) }).index({Slice(1, -1), Slice(1, -1)});
	auto g_size = torch::cat({ g, g.index({Slice(None, 1)}) }, 0);
	g_size = g_size.index({ torch::round(torch::linspace(0, (n - 1) * 4, (size - 1) * 4 + 1, g_size.options())).to(at::kInt) });
	fg = torch::cat({ torch::flip(g_size.index({ Slice(3 * size - 2, -1) }), {0}).reshape_symint({1, size - 2}), fg, g_size.index({Slice(size, 2 * size - 2)}).reshape_symint({1, size - 2}) }, 0);
	fg = torch::cat({ g_size.index({Slice(None, size)}).reshape_symint({size, 1}), fg, torch::flip(g_size.index({Slice(2 * size - 2, 3 * size - 2)}), {0}).reshape_symint({size, 1}) }, 1);
	return fg;
}