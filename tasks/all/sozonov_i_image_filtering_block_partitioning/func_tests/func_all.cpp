#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <numeric>
#include <vector>

#include "all/sozonov_i_image_filtering_block_partitioning/include/ops_all.hpp"
#include "boost/mpi/communicator.hpp"
#include "core/task/include/task.hpp"

namespace sozonov_i_image_filtering_block_partitioning_all {

std::vector<double> ZeroEdges(std::vector<double> img, int wdth, int hght) {
  for (int i = 0; i < wdth; ++i) {
    img[i] = 0;
    img[((hght - 1) * wdth) + i] = 0;
  }
  for (int i = 1; i < hght - 1; ++i) {
    img[i * wdth] = 0;
    img[(i * wdth) + wdth - 1] = 0;
  }

  return img;
}

}  // namespace sozonov_i_image_filtering_block_partitioning_all

TEST(sozonov_i_image_filtering_block_partitioning_all, test_empty_image) {
  boost::mpi::communicator world;

  const int width = 0;
  const int height = 0;

  // Create data
  std::vector<double> in;
  std::vector<double> out;

  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->inputs_count.emplace_back(width);
    task_data_all->inputs_count.emplace_back(height);
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());

    // Create Task
    sozonov_i_image_filtering_block_partitioning_all::TestTaskALL test_task_all(task_data_all);
    ASSERT_FALSE(test_task_all.Validation());
  }
}

TEST(sozonov_i_image_filtering_block_partitioning_all, test_image_less_than_3x3) {
  boost::mpi::communicator world;

  const int width = 2;
  const int height = 2;

  // Create data
  std::vector<double> in = {4, 6, 8, 24};
  std::vector<double> out(width * height, 0);

  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->inputs_count.emplace_back(width);
    task_data_all->inputs_count.emplace_back(height);
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());

    // Create Task
    sozonov_i_image_filtering_block_partitioning_all::TestTaskALL test_task_all(task_data_all);
    ASSERT_FALSE(test_task_all.Validation());
  }
}

TEST(sozonov_i_image_filtering_block_partitioning_all, test_wrong_pixels) {
  boost::mpi::communicator world;

  const int width = 5;
  const int height = 3;

  // Create data
  std::vector<double> in = {143, 6, 853, -24, 31, -25, 1, 5, -7, 361, 28, 98, -45, 982, 461};
  std::vector<double> out(width * height, 0);

  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->inputs_count.emplace_back(width);
    task_data_all->inputs_count.emplace_back(height);
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());

    // Create Task
    sozonov_i_image_filtering_block_partitioning_all::TestTaskALL test_task_all(task_data_all);
    ASSERT_FALSE(test_task_all.Validation());
  }
}

TEST(sozonov_i_image_filtering_block_partitioning_all, test_4x13) {
  boost::mpi::communicator world;

  const int width = 4;
  const int height = 13;

  // Create data
  std::vector<double> in = {34,  24, 27,  67, 42, 48,  93, 26, 47, 2,   34, 13, 81,  24, 32, 12, 34, 72,
                            112, 16, 123, 54, 32, 45,  67, 23, 76, 154, 52, 78, 94,  10, 23, 54, 93, 56,
                            23,  56, 72,  1,  14, 145, 76, 23, 21, 45,  23, 56, 111, 23, 45, 23};
  std::vector<double> out(width * height, 0);
  std::vector<double> ans = {0, 0,       0,       0, 0, 41,      46.75,   0, 0, 35.125,  32.875,  0,
                             0, 43.5625, 37.1875, 0, 0, 62.75,   55.4375, 0, 0, 62.8125, 60.4375, 0,
                             0, 58.9375, 68.5625, 0, 0, 63.5625, 73.5625, 0, 0, 59.8125, 66.8125, 0,
                             0, 63.625,  63.625,  0, 0, 68.8125, 61.75,   0, 0, 53.125,  46.875,  0,
                             0, 0,       0,       0};

  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->inputs_count.emplace_back(width);
    task_data_all->inputs_count.emplace_back(height);
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }

  // Create Task
  sozonov_i_image_filtering_block_partitioning_all::TestTaskALL test_task_all(task_data_all);
  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();

  if (world.rank() == 0) {
    EXPECT_EQ(out, ans);
  }
}

TEST(sozonov_i_image_filtering_block_partitioning_all, test_5x15) {
  boost::mpi::communicator world;

  const int width = 5;
  const int height = 15;

  // Create data
  std::vector<double> in(width * height);
  std::iota(in.begin(), in.end(), 0);
  std::vector<double> out(width * height, 0);
  std::vector<double> ans = {0, 0,  0,  0,  0, 0, 6,  7,  8,  0, 0, 11, 12, 13, 0, 0, 16, 17, 18, 0, 0, 21, 22, 23, 0,
                             0, 26, 27, 28, 0, 0, 31, 32, 33, 0, 0, 36, 37, 38, 0, 0, 41, 42, 43, 0, 0, 46, 47, 48, 0,
                             0, 51, 52, 53, 0, 0, 56, 57, 58, 0, 0, 61, 62, 63, 0, 0, 66, 67, 68, 0, 0, 0,  0,  0,  0};

  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->inputs_count.emplace_back(width);
    task_data_all->inputs_count.emplace_back(height);
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }

  // Create Task
  sozonov_i_image_filtering_block_partitioning_all::TestTaskALL test_task_all(task_data_all);
  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();

  if (world.rank() == 0) {
    EXPECT_EQ(out, ans);
  }
}

TEST(sozonov_i_image_filtering_block_partitioning_all, test_4x16) {
  boost::mpi::communicator world;

  const int width = 4;
  const int height = 16;

  // Create data
  std::vector<double> in(width * height);
  std::iota(in.begin(), in.end(), 0);
  std::vector<double> out(width * height, 0);
  std::vector<double> ans = {0,  0,  0,  0,  0,  5,  6,  0,  0,  9,  10, 0,  0,  13, 14, 0,  0,  17, 18, 0,  0,  21,
                             22, 0,  0,  25, 26, 0,  0,  29, 30, 0,  0,  33, 34, 0,  0,  37, 38, 0,  0,  41, 42, 0,
                             0,  45, 46, 0,  0,  49, 50, 0,  0,  53, 54, 0,  0,  57, 58, 0,  0,  0,  0,  0};

  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->inputs_count.emplace_back(width);
    task_data_all->inputs_count.emplace_back(height);
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }

  // Create Task
  sozonov_i_image_filtering_block_partitioning_all::TestTaskALL test_task_all(task_data_all);
  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();

  if (world.rank() == 0) {
    EXPECT_EQ(out, ans);
  }
}

TEST(sozonov_i_image_filtering_block_partitioning_all, test_4x17) {
  boost::mpi::communicator world;

  const int width = 4;
  const int height = 17;

  // Create data
  std::vector<double> in(width * height);
  std::iota(in.begin(), in.end(), 0);
  std::vector<double> out(width * height, 0);
  std::vector<double> ans = {0,  0, 0,  0,  0,  5, 6,  0,  0,  9, 10, 0,  0,  13, 14, 0,  0,  17, 18, 0,  0, 21, 22,
                             0,  0, 25, 26, 0,  0, 29, 30, 0,  0, 33, 34, 0,  0,  37, 38, 0,  0,  41, 42, 0, 0,  45,
                             46, 0, 0,  49, 50, 0, 0,  53, 54, 0, 0,  57, 58, 0,  0,  61, 62, 0,  0,  0,  0, 0};

  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->inputs_count.emplace_back(width);
    task_data_all->inputs_count.emplace_back(height);
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }

  // Create Task
  sozonov_i_image_filtering_block_partitioning_all::TestTaskALL test_task_all(task_data_all);
  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();

  if (world.rank() == 0) {
    EXPECT_EQ(out, ans);
  }
}

TEST(sozonov_i_image_filtering_block_partitioning_all, test_100x100) {
  boost::mpi::communicator world;

  const int width = 100;
  const int height = 100;

  // Create data
  std::vector<double> in(width * height, 1);
  std::vector<double> out(width * height, 0);
  std::vector<double> ans(width * height, 1);

  ans = sozonov_i_image_filtering_block_partitioning_all::ZeroEdges(ans, width, height);

  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->inputs_count.emplace_back(width);
    task_data_all->inputs_count.emplace_back(height);
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }

  // Create Task
  sozonov_i_image_filtering_block_partitioning_all::TestTaskALL test_task_all(task_data_all);
  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();

  if (world.rank() == 0) {
    EXPECT_EQ(out, ans);
  }
}

TEST(sozonov_i_image_filtering_block_partitioning_all, test_150x100) {
  boost::mpi::communicator world;

  const int width = 150;
  const int height = 100;

  // Create data
  std::vector<double> in(width * height, 1);
  std::vector<double> out(width * height, 0);
  std::vector<double> ans(width * height, 1);

  ans = sozonov_i_image_filtering_block_partitioning_all::ZeroEdges(ans, width, height);

  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->inputs_count.emplace_back(width);
    task_data_all->inputs_count.emplace_back(height);
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }

  // Create Task
  sozonov_i_image_filtering_block_partitioning_all::TestTaskALL test_task_all(task_data_all);
  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();

  if (world.rank() == 0) {
    EXPECT_EQ(out, ans);
  }
}

TEST(sozonov_i_image_filtering_block_partitioning_all, test_120x200) {
  boost::mpi::communicator world;

  const int width = 120;
  const int height = 200;

  // Create data
  std::vector<double> in(width * height, 1);
  std::vector<double> out(width * height, 0);
  std::vector<double> ans(width * height, 1);

  ans = sozonov_i_image_filtering_block_partitioning_all::ZeroEdges(ans, width, height);

  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->inputs_count.emplace_back(width);
    task_data_all->inputs_count.emplace_back(height);
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }

  // Create Task
  sozonov_i_image_filtering_block_partitioning_all::TestTaskALL test_task_all(task_data_all);
  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();

  if (world.rank() == 0) {
    EXPECT_EQ(out, ans);
  }
}