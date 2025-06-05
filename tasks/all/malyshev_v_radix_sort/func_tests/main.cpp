#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <cstdint>
#include <ctime>
#include <functional>
#include <memory>
#include <random>
#include <vector>

#include "all/malyshev_v_radix_sort/include/ops_all.hpp"
#include "core/task/include/task.hpp"

std::vector<double> malyshev_v_radix_sort_all::GetRandomDoubleVector(int size) {
  std::vector<double> vector(size);
  std::mt19937 generator(static_cast<unsigned>(std::time(nullptr)));
  std::uniform_real_distribution<double> distribution(-1000.0, 1000.0);
  for (int i = 0; i < size; ++i) {
    vector[i] = distribution(generator);
  }
  return vector;
}

TEST(malyshev_v_radix_sort_all, ordinary_test) {
  boost::mpi::communicator world;
  std::vector<double> global_vector = {5.69, -2.11, 0.52, 3.14, -1.41};
  std::vector<double> result(global_vector.size());
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
    task_data->inputs_count.emplace_back(global_vector.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
    task_data->outputs_count.emplace_back(result.size());
  }
  malyshev_v_radix_sort_all::TestTaskALL task_all(task_data);
  ASSERT_TRUE(task_all.ValidationImpl());
  task_all.PreProcessingImpl();
  task_all.RunImpl();
  task_all.PostProcessingImpl();
  if (world.rank() == 0) {
    std::vector<double> sorted = global_vector;
    std::ranges::sort(sorted);
    ASSERT_EQ(result, sorted);
  }
}

TEST(malyshev_v_radix_sort_all, random_vector_test) {
  boost::mpi::communicator world;
  const int size = 10;
  auto global_vector = malyshev_v_radix_sort_all::GetRandomDoubleVector(size);
  std::vector<double> result(size);
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
    task_data->inputs_count.emplace_back(global_vector.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
    task_data->outputs_count.emplace_back(result.size());
  }
  malyshev_v_radix_sort_all::TestTaskALL task_all(task_data);
  ASSERT_TRUE(task_all.ValidationImpl());
  task_all.PreProcessingImpl();
  task_all.RunImpl();
  task_all.PostProcessingImpl();
  if (world.rank() == 0) {
    std::vector<double> sorted = global_vector;
    std::ranges::sort(sorted);
    ASSERT_EQ(result, sorted);
  }
}

TEST(malyshev_v_radix_sort_all, negative_numbers_test) {
  boost::mpi::communicator world;
  std::vector<double> global_vector = {-8.5, -2.3, -10.1, -0.7, -5.4};
  std::vector<double> result(global_vector.size());
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
    task_data->inputs_count.emplace_back(global_vector.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
    task_data->outputs_count.emplace_back(result.size());
  }
  malyshev_v_radix_sort_all::TestTaskALL task_all(task_data);
  ASSERT_TRUE(task_all.ValidationImpl());
  task_all.PreProcessingImpl();
  task_all.RunImpl();
  task_all.PostProcessingImpl();
  if (world.rank() == 0) {
    std::vector<double> sorted = global_vector;
    std::ranges::sort(sorted);
    ASSERT_EQ(result, sorted);
  }
}

TEST(malyshev_v_radix_sort_all, zeros_test) {
  boost::mpi::communicator world;
  std::vector<double> global_vector(10, 0.0);
  std::vector<double> result(global_vector.size());
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
    task_data->inputs_count.emplace_back(global_vector.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
    task_data->outputs_count.emplace_back(result.size());
  }
  malyshev_v_radix_sort_all::TestTaskALL task_all(task_data);
  ASSERT_TRUE(task_all.ValidationImpl());
  task_all.PreProcessingImpl();
  task_all.RunImpl();
  task_all.PostProcessingImpl();
  if (world.rank() == 0) {
    ASSERT_EQ(result, global_vector);
  }
}

TEST(malyshev_v_radix_sort_all, duplicates_test) {
  boost::mpi::communicator world;
  std::vector<double> global_vector = {3.14, 3.14, 2.71, 2.71, 1.62};
  std::vector<double> result(global_vector.size());
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
    task_data->inputs_count.emplace_back(global_vector.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
    task_data->outputs_count.emplace_back(result.size());
  }
  malyshev_v_radix_sort_all::TestTaskALL task_all(task_data);
  ASSERT_TRUE(task_all.ValidationImpl());
  task_all.PreProcessingImpl();
  task_all.RunImpl();
  task_all.PostProcessingImpl();
  if (world.rank() == 0) {
    std::vector<double> sorted = global_vector;
    std::ranges::sort(sorted);
    ASSERT_EQ(result, sorted);
  }
}

TEST(malyshev_v_radix_sort_all, reverse_order_test) {
  boost::mpi::communicator world;
  const int size = 10;
  auto global_vector = malyshev_v_radix_sort_all::GetRandomDoubleVector(size);
  std::ranges::sort(global_vector, std::greater<>());
  std::vector<double> result(size);
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
    task_data->inputs_count.emplace_back(global_vector.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
    task_data->outputs_count.emplace_back(result.size());
  }
  malyshev_v_radix_sort_all::TestTaskALL task_all(task_data);
  ASSERT_TRUE(task_all.ValidationImpl());
  task_all.PreProcessingImpl();
  task_all.RunImpl();
  task_all.PostProcessingImpl();
  if (world.rank() == 0) {
    std::vector<double> sorted = global_vector;
    std::ranges::sort(sorted);
    ASSERT_EQ(result, sorted);
  }
}

TEST(malyshev_v_radix_sort_all, validation_test) {
  boost::mpi::communicator world;
  std::vector<double> global_vector = {1.0};
  std::vector<double> result(2);
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
    task_data->inputs_count.emplace_back(global_vector.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
    task_data->outputs_count.emplace_back(result.size());
  }
  malyshev_v_radix_sort_all::TestTaskALL task_all(task_data);
  if (world.rank() == 0) {
    ASSERT_FALSE(task_all.ValidationImpl());
  }
}