#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <ctime>
#include <functional>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "stl/malyshev_v_radix_sort/include/ops_stl.hpp"

std::vector<double> malyshev_v_radix_sort_stl::GetRandomDoubleVector(int size) {
  std::vector<double> vector(size);
  std::mt19937 generator(static_cast<unsigned>(std::time(nullptr)));
  std::uniform_real_distribution<double> distribution(-1000.0, 1000.0);
  for (int i = 0; i < size; ++i) {
    vector[i] = distribution(generator);
  }
  return vector;
}

TEST(malyshev_v_radix_sort_all, ordinary_test) {
  int global_vector_size = 3;
  std::vector<double> global_vector = {5.69, -2.11, 0.52};
  std::vector<double> result(global_vector_size);
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
  task_data->inputs_count.emplace_back(global_vector.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
  task_data->outputs_count.emplace_back(result.size());
  malyshev_v_radix_sort_stl::TestTaskSTL task_stl(task_data);
  ASSERT_TRUE(task_stl.ValidationImpl());
  task_stl.PreProcessingImpl();
  task_stl.RunImpl();
  task_stl.PostProcessingImpl();
  std::vector<double> sorted_global_vector = global_vector;
  std::ranges::sort(sorted_global_vector);
  ASSERT_EQ(result, sorted_global_vector);
}

TEST(malyshev_v_radix_sort_all, random_vector_test) {
  int global_vector_size = 528;
  std::vector<double> global_vector = malyshev_v_radix_sort_stl::GetRandomDoubleVector(global_vector_size);
  std::vector<double> result(global_vector_size);
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
  task_data->inputs_count.emplace_back(global_vector.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
  task_data->outputs_count.emplace_back(result.size());
  malyshev_v_radix_sort_stl::TestTaskSTL task_stl(task_data);
  ASSERT_TRUE(task_stl.ValidationImpl());
  task_stl.PreProcessingImpl();
  task_stl.RunImpl();
  task_stl.PostProcessingImpl();
  std::vector<double> sorted_global_vector = global_vector;
  std::ranges::sort(sorted_global_vector);
  ASSERT_EQ(result, sorted_global_vector);
}

TEST(malyshev_v_radix_sort_all, negative_numbers_test) {
  int global_vector_size = 11;
  std::vector<double> global_vector = {-8.55,   1.85,   -4.0,   2.81828, 8.77,   -5.56562,
                                       -15.823, -6.971, 3.1615, 0.0,     10.1415};
  std::vector<double> result(global_vector_size);
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
  task_data->inputs_count.emplace_back(global_vector.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
  task_data->outputs_count.emplace_back(result.size());
  malyshev_v_radix_sort_stl::TestTaskSTL task_stl(task_data);
  ASSERT_TRUE(task_stl.ValidationImpl());
  task_stl.PreProcessingImpl();
  task_stl.RunImpl();
  task_stl.PostProcessingImpl();
  std::vector<double> sorted_global_vector = global_vector;
  std::ranges::sort(sorted_global_vector);
  ASSERT_EQ(result, sorted_global_vector);
}

TEST(malyshev_v_radix_sort_all, zeros_test) {
  int global_vector_size = 7;
  std::vector<double> global_vector = {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00};
  std::vector<double> result(global_vector_size);
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
  task_data->inputs_count.emplace_back(global_vector.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
  task_data->outputs_count.emplace_back(result.size());
  malyshev_v_radix_sort_stl::TestTaskSTL task_stl(task_data);
  ASSERT_TRUE(task_stl.ValidationImpl());
  task_stl.PreProcessingImpl();
  task_stl.RunImpl();
  task_stl.PostProcessingImpl();
  ASSERT_EQ(result, global_vector);
}

TEST(malyshev_v_radix_sort_all, duplicates_test) {
  int global_vector_size = 10;
  std::vector<double> global_vector = {5.0, 2.0, 5.0, -3.0, 2.0, 2.0, 0.0, -3.0, 5.0, 2.0};
  std::vector<double> result(global_vector_size);
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
  task_data->inputs_count.emplace_back(global_vector.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
  task_data->outputs_count.emplace_back(result.size());
  malyshev_v_radix_sort_stl::TestTaskSTL task_stl(task_data);
  ASSERT_TRUE(task_stl.ValidationImpl());
  task_stl.PreProcessingImpl();
  task_stl.RunImpl();
  task_stl.PostProcessingImpl();
  std::vector<double> sorted_global_vector = global_vector;
  std::ranges::sort(sorted_global_vector);
  ASSERT_EQ(result, sorted_global_vector);
}

TEST(malyshev_v_radix_sort_all, reverse_order_test) {
  int global_vector_size = 132;
  std::vector<double> global_vector = malyshev_v_radix_sort_stl::GetRandomDoubleVector(global_vector_size);
  std::ranges::sort(global_vector, std::greater<>());
  std::vector<double> result(global_vector_size);
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
  task_data->inputs_count.emplace_back(global_vector.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
  task_data->outputs_count.emplace_back(result.size());
  malyshev_v_radix_sort_stl::TestTaskSTL task_stl(task_data);
  ASSERT_TRUE(task_stl.ValidationImpl());
  task_stl.PreProcessingImpl();
  task_stl.RunImpl();
  task_stl.PostProcessingImpl();
  std::vector<double> sorted_global_vector = global_vector;
  std::ranges::sort(sorted_global_vector);
  ASSERT_EQ(result, sorted_global_vector);
}

TEST(malyshev_v_radix_sort_all, validation_test) {
  constexpr size_t kInputSize = 10;
  constexpr size_t kOutputSize = 5;

  std::vector<double> input(kInputSize);
  std::vector<double> output(kOutputSize);

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  malyshev_v_radix_sort_stl::TestTaskSTL task_stl(task_data);
  ASSERT_FALSE(task_stl.ValidationImpl());
}
