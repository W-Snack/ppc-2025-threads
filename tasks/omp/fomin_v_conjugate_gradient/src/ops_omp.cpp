#include "omp/fomin_v_conjugate_gradient/include/ops_omp.hpp"

#include <omp.h>

#include <cmath>
#include <vector>

double fomin_v_conjugate_gradient::FominVConjugateGradientOmp::DotProduct(const std::vector<double>& a,
                                                                          const std::vector<double>& b) {
  double result = 0.0;
#pragma omp parallel for reduction(+ : result)  // Параллелим с редукцией
  for (int i = 0; i < static_cast<int>(a.size()); ++i) {
    result += a[i] * b[i];
  }
  return result;
}

std::vector<double> fomin_v_conjugate_gradient::FominVConjugateGradientOmp::MatrixVectorMultiply(
    const std::vector<double>& a, const std::vector<double>& x) const {
  std::vector<double> result(n, 0.0);
#pragma omp parallel for  // Параллелим по строкам матрицы
  for (int i = 0; i < n; ++i) {
    double sum = 0.0;
    for (int j = 0; j < n; ++j) {
      sum += a[(i * n) + j] * x[j];
    }
    result[i] = sum;
  }
  return result;
}

std::vector<double> fomin_v_conjugate_gradient::FominVConjugateGradientOmp::VectorAdd(const std::vector<double>& a,
                                                                                      const std::vector<double>& b) {
  std::vector<double> result(a.size());
#pragma omp parallel for  // Параллелим поэлементное сложение
  for (int i = 0; i < static_cast<int>(a.size()); ++i) {
    result[i] = a[i] + b[i];
  }
  return result;
}

std::vector<double> fomin_v_conjugate_gradient::FominVConjugateGradientOmp::VectorSub(const std::vector<double>& a,
                                                                                      const std::vector<double>& b) {
  std::vector<double> result(a.size());
#pragma omp parallel for  // Параллелим поэлементное вычитание
  for (int i = 0; i < static_cast<int>(a.size()); ++i) {
    result[i] = a[i] - b[i];
  }
  return result;
}

std::vector<double> fomin_v_conjugate_gradient::FominVConjugateGradientOmp::VectorScalarMultiply(
    const std::vector<double>& v, double scalar) {
  std::vector<double> result(v.size());
#pragma omp parallel for  // Параллелим умножение на скаляр
  for (int i = 0; i < static_cast<int>(v.size()); ++i) {
    result[i] = v[i] * scalar;
  }
  return result;
}

bool fomin_v_conjugate_gradient::FominVConjugateGradientOmp::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  std::vector<double> input(in_ptr, in_ptr + input_size);

  n = static_cast<int>((-1.0 + std::sqrt(1 + (4 * input_size))) / 2);
  a_ = std::vector<double>(input.begin(), input.begin() + (n * n));
  b_ = std::vector<double>(input.begin() + (n * n), input.end());
  output_.resize(n, 0.0);

  return true;
}

bool fomin_v_conjugate_gradient::FominVConjugateGradientOmp::ValidationImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  const int calculated_n = static_cast<int>((-1.0 + std::sqrt(1 + (4 * input_size))) / 2);
  return (static_cast<unsigned int>(calculated_n * (calculated_n + 1)) == input_size) &&
         (task_data->outputs_count[0] == static_cast<unsigned int>(calculated_n));
}

bool fomin_v_conjugate_gradient::FominVConjugateGradientOmp::RunImpl() {
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

bool fomin_v_conjugate_gradient::FominVConjugateGradientOmp::PostProcessingImpl() {
  auto* out_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
#pragma omp parallel for  // Параллельное копирование результатов
  for (int i = 0; i < static_cast<int>(output_.size()); ++i) {
    out_ptr[i] = output_[i];
  }
  return true;
}