#include "stl/fomin_v_conjugate_gradient/include/ops_stl.hpp"

#include <cmath>
#include <cstddef>
#include <execution>
#include <numeric>
#include <thread>
#include <vector>

template <typename Func>
void ParallelFor(size_t start, size_t end, Func func, size_t num_threads) {
  std::vector<std::thread> threads;
  size_t chunk_size = (end - start) / num_threads;

  for (size_t i = 0; i < num_threads; ++i) {
    size_t chunk_start = start + i * chunk_size;
    size_t chunk_end = (i == num_threads - 1) ? end : chunk_start + chunk_size;
    threads.emplace_back([=, &func]() {
      for (size_t j = chunk_start; j < chunk_end; ++j) {
        func(j);
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }
}

double fomin_v_conjugate_gradient::FominVConjugateGradientStl::DotProduct(const std::vector<double>& a,
                                                                          const std::vector<double>& b) {
  const size_t size = a.size();
  const size_t num_threads = std::min(static_cast<size_t>(std::thread::hardware_concurrency()), size);
  std::vector<double> partial_sums(num_threads, 0.0);
  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  const size_t chunk_size = size / num_threads;

  for (size_t t = 0; t < num_threads; ++t) {
    const size_t start = t * chunk_size;
    const size_t end = (t == num_threads - 1) ? size : start + chunk_size;
    threads.emplace_back([&, start, end, t]() {
      double sum = 0.0;
      for (size_t i = start; i < end; ++i) {
        sum += a[i] * b[i];
      }
      partial_sums[t] = sum;
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  return std::accumulate(partial_sums.begin(), partial_sums.end(), 0.0);
}

std::vector<double> fomin_v_conjugate_gradient::FominVConjugateGradientStl::MatrixVectorMultiply(
    const std::vector<double>& a, const std::vector<double>& x) const {
  std::vector<double> result(n, 0.0);
  const size_t num_threads = std::thread::hardware_concurrency();

  ParallelFor(
      0, n,
      [&](size_t i) {
        for (int j = 0; j < n; ++j) {
          result[i] += a[(i * n) + j] * x[j];
        }
      },
      num_threads);

  return result;
}

std::vector<double> fomin_v_conjugate_gradient::FominVConjugateGradientStl::VectorAdd(const std::vector<double>& a,
                                                                                      const std::vector<double>& b) {
  std::vector<double> result(a.size());
  const size_t num_threads = std::thread::hardware_concurrency();

  ParallelFor(0, a.size(), [&](size_t i) { result[i] = a[i] + b[i]; }, num_threads);

  return result;
}

std::vector<double> fomin_v_conjugate_gradient::FominVConjugateGradientStl::VectorSub(const std::vector<double>& a,
                                                                                      const std::vector<double>& b) {
  std::vector<double> result(a.size());
  const size_t num_threads = std::thread::hardware_concurrency();

  ParallelFor(0, a.size(), [&](size_t i) { result[i] = a[i] - b[i]; }, num_threads);

  return result;
}

std::vector<double> fomin_v_conjugate_gradient::FominVConjugateGradientStl::VectorScalarMultiply(
    const std::vector<double>& v, double scalar) {
  std::vector<double> result(v.size());
  const size_t num_threads = std::thread::hardware_concurrency();

  ParallelFor(0, v.size(), [&](size_t i) { result[i] = v[i] * scalar; }, num_threads);

  return result;
}

bool fomin_v_conjugate_gradient::FominVConjugateGradientStl::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  std::vector<double> input(in_ptr, in_ptr + input_size);

  n = static_cast<int>((-1.0 + std::sqrt(1 + (4 * input_size))) / 2);
  a_ = std::vector<double>(input.begin(), input.begin() + (n * n));
  b_ = std::vector<double>(input.begin() + (n * n), input.end());
  output_.resize(n, 0.0);

  return true;
}

bool fomin_v_conjugate_gradient::FominVConjugateGradientStl::ValidationImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  const int calculated_n = static_cast<int>((-1.0 + std::sqrt(1 + (4 * input_size))) / 2);
  return (static_cast<unsigned int>(calculated_n * (calculated_n + 1)) == input_size) &&
         (task_data->outputs_count[0] == static_cast<unsigned int>(calculated_n));
}

bool fomin_v_conjugate_gradient::FominVConjugateGradientStl::RunImpl() {
  const double epsilon = 1e-6;
  const int max_iter = 1000;
  std::vector<double> x(n, 0.0);
  std::vector<double> r = b_;
  std::vector<double> p = r;
  double rs_old = DotProduct(r, r);

  for (int i = 0; i < max_iter; ++i) {
    std::vector<double> ap = MatrixVectorMultiply(a_, p);
    double p_ap = DotProduct(p, ap);

    if (std::abs(p_ap) < 1e-12) break;

    double alpha = rs_old / p_ap;
    x = VectorAdd(x, VectorScalarMultiply(p, alpha));
    std::vector<double> r_new = VectorSub(r, VectorScalarMultiply(ap, alpha));

    double rs_new = DotProduct(r_new, r_new);
    if (std::sqrt(rs_new) < epsilon) break;

    double beta = rs_new / rs_old;
    p = VectorAdd(r_new, VectorScalarMultiply(p, beta));
    r = r_new;
    rs_old = rs_new;
  }

  output_ = x;
  return true;
}

bool fomin_v_conjugate_gradient::FominVConjugateGradientStl::PostProcessingImpl() {
  auto* out_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
  for (size_t i = 0; i < output_.size(); ++i) {
    out_ptr[i] = output_[i];
  }
  return true;
}
