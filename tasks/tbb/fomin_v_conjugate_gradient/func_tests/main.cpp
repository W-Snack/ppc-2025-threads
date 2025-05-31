#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "tbb/fomin_v_conjugate_gradient/include/ops_tbb.hpp"

TEST(FominVConjugateGradientTbb, test_small_system) {
  constexpr size_t kCount = 3;
  std::vector<double> a = {4, 1, 1, 1, 3, 0, 1, 0, 2};
  std::vector<double> b = {6, 5, 3};
  // Correct solution computed as [17/19, 26/19, 20/19]
  std::vector<double> expected_x = {17.0 / 19.0, 26.0 / 19.0, 20.0 / 19.0};

  std::vector<double> input;
  input.insert(input.end(), a.begin(), a.end());
  input.insert(input.end(), b.begin(), b.end());

  std::vector<double> out(kCount, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data_seq->inputs_count.emplace_back(input.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  fomin_v_conjugate_gradient::FominVConjugateGradientTbb test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  for (size_t i = 0; i < kCount; ++i) {
    EXPECT_NEAR(out[i], expected_x[i], 1e-6);
  }
}

TEST(FominVConjugateGradientTbb, test_large_system) {
  constexpr size_t kCount = 5;
  std::vector<double> a = {5, 1, 0, 0, 0, 1, 5, 1, 0, 0, 0, 1, 5, 1, 0, 0, 0, 1, 5, 1, 0, 0, 0, 1, 5};
  std::vector<double> b = {6, 7, 7, 7, 6};
  std::vector<double> expected_x = {1.0, 1.0, 1.0, 1.0, 1.0};
  std::vector<double> out(kCount, 0.0);

  // Correct input setup
  std::vector<double> input;
  input.insert(input.end(), a.begin(), a.end());
  input.insert(input.end(), b.begin(), b.end());

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data_seq->inputs_count.emplace_back(input.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  fomin_v_conjugate_gradient::FominVConjugateGradientTbb test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  for (size_t i = 0; i < kCount; ++i) {
    EXPECT_NEAR(out[i], expected_x[i], 1e-6);
  }
}

TEST(FominVConjugateGradientTbb, DotProduct) {
  std::vector<double> a = {1.0, 2.0, 3.0};
  std::vector<double> b = {4.0, 5.0, 6.0};
  double expected = (1.0 * 4.0) + (2.0 * 5.0) + (3.0 * 6.0);  // 32.0
  auto task_data = std::make_shared<ppc::core::TaskData>();
  fomin_v_conjugate_gradient::FominVConjugateGradientTbb task(task_data);

  EXPECT_DOUBLE_EQ(task.DotProduct(a, b), expected);
}

TEST(FominVConjugateGradientSeq, MatrixVectorMultiply) {
  std::vector<double> a = {1.0, 2.0, 3.0, 4.0};  // 2x2 матрица
  std::vector<double> x = {5.0, 6.0};
  std::vector<double> expected = {(1 * 5) + (2 * 6), (3 * 5) + (4 * 6)};  // {17, 39}

  auto task_data = std::make_shared<ppc::core::TaskData>();
  fomin_v_conjugate_gradient::FominVConjugateGradientTbb task(task_data);

  task.n = 2;

  auto result = task.MatrixVectorMultiply(a, x);
  EXPECT_EQ(result, expected);
}

TEST(FominVConjugateGradientTbb, VectorAdd) {
  std::vector<double> a = {1.0, 2.0, 3.0};
  std::vector<double> b = {4.0, 5.0, 6.0};
  std::vector<double> expected = {5.0, 7.0, 9.0};
  auto result = fomin_v_conjugate_gradient::FominVConjugateGradientTbb::VectorAdd(a, b);
  EXPECT_EQ(result.size(), expected.size());
  for (size_t i = 0; i < result.size(); ++i) {
    EXPECT_DOUBLE_EQ(result[i], expected[i]);
  }
}

TEST(FominVConjugateGradientTbb, VectorSub) {
  std::vector<double> a = {1.0, 2.0, 3.0};
  std::vector<double> b = {4.0, 5.0, 6.0};
  std::vector<double> expected = {-3.0, -3.0, -3.0};
  auto result = fomin_v_conjugate_gradient::FominVConjugateGradientTbb::VectorSub(a, b);
  EXPECT_EQ(result.size(), expected.size());
  for (size_t i = 0; i < result.size(); ++i) {
    EXPECT_DOUBLE_EQ(result[i], expected[i]);
  }
}

TEST(FominVConjugateGradientTbb, VectorScalarMultiply) {
  std::vector<double> v = {1.0, 2.0, 3.0};
  double scalar = 2.0;
  std::vector<double> expected = {2.0, 4.0, 6.0};
  auto result = fomin_v_conjugate_gradient::FominVConjugateGradientTbb::VectorScalarMultiply(v, scalar);
  EXPECT_EQ(result.size(), expected.size());
  for (size_t i = 0; i < result.size(); ++i) {
    EXPECT_DOUBLE_EQ(result[i], expected[i]);
  }
}
