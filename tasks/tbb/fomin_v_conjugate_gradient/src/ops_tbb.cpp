#include "tbb/fomin_v_conjugate_gradient/include/ops_tbb.hpp"

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/parallel_reduce.h>

#include <cmath>
#include <cstddef>
#include <functional>
#include <vector>

double fomin_v_conjugate_gradient::FominVConjugateGradientTbb::DotProduct(const std::vector<double>& a,
                                                                          const std::vector<double>& b) {
  return tbb::parallel_reduce(
      tbb::blocked_range<size_t>(0, a.size()), 0.0,
      [&](const tbb::blocked_range<size_t>& r, double init) {
        for (size_t i = r.begin(); i < r.end(); ++i) {
          init += a[i] * b[i];
        }
        return init;
      },
      std::plus<>());
}

std::vector<double> fomin_v_conjugate_gradient::FominVConjugateGradientTbb::MatrixVectorMultiply(
    const std::vector<double>& a, const std::vector<double>& x) const {
  std::vector<double> result(n, 0.0);
  tbb::parallel_for(tbb::blocked_range<int>(0, n), [&](const tbb::blocked_range<int>& r) {
    for (int i = r.begin(); i < r.end(); ++i) {
      double temp = 0.0;
      for (int j = 0; j < n; ++j) {
        temp += a[(i * n) + j] * x[j];
      }
      result[i] = temp;
    }
  });
  return result;
}

std::vector<double> fomin_v_conjugate_gradient::FominVConjugateGradientTbb::VectorAdd(const std::vector<double>& a,
                                                                                      const std::vector<double>& b) {
  std::vector<double> result(a.size());
  tbb::parallel_for(tbb::blocked_range<size_t>(0, a.size()), [&](const tbb::blocked_range<size_t>& r) {
    for (size_t i = r.begin(); i < r.end(); ++i) {
      result[i] = a[i] + b[i];
    }
  });
  return result;
}

std::vector<double> fomin_v_conjugate_gradient::FominVConjugateGradientTbb::VectorSub(const std::vector<double>& a,
                                                                                      const std::vector<double>& b) {
  std::vector<double> result(a.size());
  tbb::parallel_for(tbb::blocked_range<size_t>(0, a.size()), [&](const tbb::blocked_range<size_t>& r) {
    for (size_t i = r.begin(); i < r.end(); ++i) {
      result[i] = a[i] - b[i];
    }
  });
  return result;
}

std::vector<double> fomin_v_conjugate_gradient::FominVConjugateGradientTbb::VectorScalarMultiply(
    const std::vector<double>& v, double scalar) {
  std::vector<double> result(v.size());
  tbb::parallel_for(tbb::blocked_range<size_t>(0, v.size()), [&](const tbb::blocked_range<size_t>& r) {
    for (size_t i = r.begin(); i < r.end(); ++i) {
      result[i] = v[i] * scalar;
    }
  });
  return result;
}

bool fomin_v_conjugate_gradient::FominVConjugateGradientTbb::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  std::vector<double> input(in_ptr, in_ptr + input_size);

  n = static_cast<int>((-1.0 + std::sqrt(1 + (4 * input_size))) / 2);
  a_ = std::vector<double>(input.begin(), input.begin() + (n * n));
  b_ = std::vector<double>(input.begin() + (n * n), input.end());
  output_.resize(n, 0.0);

  return true;
}

bool fomin_v_conjugate_gradient::FominVConjugateGradientTbb::ValidationImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  const int calculated_n = static_cast<int>((-1.0 + std::sqrt(1 + (4 * input_size))) / 2);
  return (static_cast<unsigned int>(calculated_n * (calculated_n + 1)) == input_size) &&
         (task_data->outputs_count[0] == static_cast<unsigned int>(calculated_n));
}

bool fomin_v_conjugate_gradient::FominVConjugateGradientTbb::RunImpl() {
  const double epsilon = 1e-6;
  const int max_iter = 1000;
  std::vector<double> x(n, 0.0);
  std::vector<double> r = b_;
  std::vector<double> p = r;
  double rs_old = DotProduct(r, r);

  for (int i = 0; i < max_iter; ++i) {
    std::vector<double> ap = MatrixVectorMultiply(a_, p);
    double p_ap = DotProduct(p, ap);

    if (std::abs(p_ap) < 1e-12) {
      break;
    }

    double alpha = rs_old / p_ap;
    x = VectorAdd(x, VectorScalarMultiply(p, alpha));
    std::vector<double> r_new = VectorSub(r, VectorScalarMultiply(ap, alpha));

    double rs_new = DotProduct(r_new, r_new);
    if (std::sqrt(rs_new) < epsilon) {
      break;
    }

    double beta = rs_new / rs_old;
    p = VectorAdd(r_new, VectorScalarMultiply(p, beta));
    r = r_new;
    rs_old = rs_new;
  }

  output_ = x;
  return true;
}

bool fomin_v_conjugate_gradient::FominVConjugateGradientTbb::PostProcessingImpl() {
  auto* out_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
  for (size_t i = 0; i < output_.size(); ++i) {
    out_ptr[i] = output_[i];
  }
  return true;
}
