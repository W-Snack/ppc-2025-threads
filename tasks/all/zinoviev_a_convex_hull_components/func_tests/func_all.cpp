#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "all/zinoviev_a_convex_hull_components/include/ops_all.hpp"
#include "core/task/include/task.hpp"

namespace {

void SetupTaskData(const std::vector<int>& input, int width, int height, size_t output_size,
                   std::shared_ptr<ppc::core::TaskData>& task_data) {
  task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(input.data())));
  task_data->inputs_count.emplace_back(width);
  task_data->inputs_count.emplace_back(height);
  task_data->outputs.emplace_back(
      reinterpret_cast<uint8_t*>(new zinoviev_a_convex_hull_components_all::Point[output_size]));
  task_data->outputs_count.emplace_back(output_size);
}

void VerifyResult(const std::vector<zinoviev_a_convex_hull_components_all::Point>& actual,
                  const std::vector<zinoviev_a_convex_hull_components_all::Point>& expected) {
  auto sorted_actual = actual;
  auto sorted_expected = expected;

  auto point_comparator = [](const zinoviev_a_convex_hull_components_all::Point& a,
                             const zinoviev_a_convex_hull_components_all::Point& b) {
    return a.x < b.x || (a.x == b.x && a.y < b.y);
  };

  std::ranges::sort(sorted_actual, point_comparator);
  std::ranges::sort(sorted_expected, point_comparator);

  ASSERT_EQ(sorted_actual.size(), sorted_expected.size());
  for (size_t i = 0; i < sorted_actual.size(); ++i) {
    ASSERT_EQ(sorted_actual[i].x, sorted_expected[i].x);
    ASSERT_EQ(sorted_actual[i].y, sorted_expected[i].y);
  }
}

void RunAndValidate(const std::vector<int>& input,
                    const std::vector<zinoviev_a_convex_hull_components_all::Point>& expected, int width, int height) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::shared_ptr<ppc::core::TaskData> task_data;
  SetupTaskData(input, width, height, expected.size(), task_data);

  zinoviev_a_convex_hull_components_all::ConvexHullMPI task(task_data);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  if (rank == 0) {
    auto* output = reinterpret_cast<zinoviev_a_convex_hull_components_all::Point*>(task_data->outputs[0]);
    size_t actual_size = task_data->outputs_count[0];
    std::vector<zinoviev_a_convex_hull_components_all::Point> actual(output, output + actual_size);

    VerifyResult(actual, expected);
  }

  delete[] reinterpret_cast<zinoviev_a_convex_hull_components_all::Point*>(task_data->outputs[0]);
}

}  // namespace

TEST(zinoviev_a_convex_hull_components_all, SquareShape) {
  constexpr int kWidth = 5;
  constexpr int kHeight = 5;
  const std::vector<int> input = {1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1};
  const std::vector<zinoviev_a_convex_hull_components_all::Point> expected = {
      {.x = 0, .y = 0}, {.x = 4, .y = 0}, {.x = 0, .y = 4}, {.x = 4, .y = 4}};

  RunAndValidate(input, expected, kWidth, kHeight);
}

TEST(zinoviev_a_convex_hull_components_all, TriangleShape) {
  constexpr int kWidth = 5;
  constexpr int kHeight = 5;
  const std::vector<int> input = {1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0};
  const std::vector<zinoviev_a_convex_hull_components_all::Point> expected = {
      {.x = 0, .y = 0}, {.x = 2, .y = 2}, {.x = 0, .y = 4}};

  RunAndValidate(input, expected, kWidth, kHeight);
}

TEST(zinoviev_a_convex_hull_components_all, SinglePixel) {
  constexpr int kWidth = 3;
  constexpr int kHeight = 3;
  const std::vector<int> input = {0, 0, 0, 0, 1, 0, 0, 0, 0};
  const std::vector<zinoviev_a_convex_hull_components_all::Point> expected = {{.x = 1, .y = 1}};
  RunAndValidate(input, expected, kWidth, kHeight);
}

TEST(zinoviev_a_convex_hull_components_all, LineHorizontal) {
  constexpr int kWidth = 6;
  constexpr int kHeight = 3;
  const std::vector<int> input = {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0};
  const std::vector<zinoviev_a_convex_hull_components_all::Point> expected = {{.x = 0, .y = 1}, {.x = 4, .y = 1}};
  RunAndValidate(input, expected, kWidth, kHeight);
}

TEST(zinoviev_a_convex_hull_components_all, LineVertical) {
  constexpr int kWidth = 3;
  constexpr int kHeight = 5;
  const std::vector<int> input = {0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0};
  const std::vector<zinoviev_a_convex_hull_components_all::Point> expected = {{.x = 1, .y = 0}, {.x = 1, .y = 4}};
  RunAndValidate(input, expected, kWidth, kHeight);
}

TEST(zinoviev_a_convex_hull_components_all, LShapePattern) {
  constexpr int kWidth = 4;
  constexpr int kHeight = 4;
  const std::vector<int> input = {1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1};
  const std::vector<zinoviev_a_convex_hull_components_all::Point> expected = {
      {.x = 0, .y = 0}, {.x = 0, .y = 3}, {.x = 3, .y = 3}};
  RunAndValidate(input, expected, kWidth, kHeight);
}

TEST(zinoviev_a_convex_hull_components_all, DiagonalPattern) {
  constexpr int kWidth = 4;
  constexpr int kHeight = 4;
  const std::vector<int> input = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
  const std::vector<zinoviev_a_convex_hull_components_all::Point> expected = {{.x = 0, .y = 0}, {.x = 3, .y = 3}};
  RunAndValidate(input, expected, kWidth, kHeight);
}

TEST(zinoviev_a_convex_hull_components_all, EmptyGrid) {
  constexpr int kWidth = 3;
  constexpr int kHeight = 3;
  const std::vector<int> input = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  const std::vector<zinoviev_a_convex_hull_components_all::Point> expected = {};
  RunAndValidate(input, expected, kWidth, kHeight);
}