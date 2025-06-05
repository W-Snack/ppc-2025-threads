#include "all/sozonov_i_image_filtering_block_partitioning/include/ops_all.hpp"

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/gatherv.hpp>
#include <boost/mpi/collectives/scatterv.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <cstddef>
#include <vector>

bool sozonov_i_image_filtering_block_partitioning_all::TestTaskALL::PreProcessingImpl() {
  if (world_.rank() == 0) {
    // Init image
    image_ = std::vector<double>(task_data->inputs_count[0]);
    auto *image_ptr = reinterpret_cast<double *>(task_data->inputs[0]);
    std::ranges::copy(image_ptr, image_ptr + task_data->inputs_count[0], image_.begin());

    width_ = static_cast<int>(task_data->inputs_count[1]);
    height_ = static_cast<int>(task_data->inputs_count[2]);

    // Init filtered image
    filtered_image_ = std::vector<double>(width_ * height_, 0);
  }
  return true;
}

bool sozonov_i_image_filtering_block_partitioning_all::TestTaskALL::ValidationImpl() {
  if (world_.rank() == 0) {
    // Init image
    image_ = std::vector<double>(task_data->inputs_count[0]);
    auto *image_ptr = reinterpret_cast<double *>(task_data->inputs[0]);
    std::ranges::copy(image_ptr, image_ptr + task_data->inputs_count[0], image_.begin());

    size_t img_size = task_data->inputs_count[1] * task_data->inputs_count[2];

    // Check pixels range from 0 to 255
    for (size_t i = 0; i < img_size; ++i) {
      if (image_[i] < 0 || image_[i] > 255) {
        return false;
      }
    }

    // Check size of image
    return task_data->inputs_count[0] > 0 && task_data->inputs_count[0] == img_size &&
           task_data->outputs_count[0] == img_size && task_data->inputs_count[1] >= 3 &&
           task_data->inputs_count[2] >= 3;
  }
  return true;
}

bool sozonov_i_image_filtering_block_partitioning_all::TestTaskALL::RunImpl() {
  broadcast(world_, width_, 0);
  broadcast(world_, height_, 0);

  int rank = world_.rank();
  int num_procs = world_.size();

  if (num_procs == 0) {
    return false;
  }

  std::vector<int> block_sizes(num_procs);
  std::vector<int> displs(num_procs);

  int remaining_blocks = height_;
  int blocks_per_proc = height_ / num_procs;

  block_sizes[0] = blocks_per_proc * width_;
  displs[0] = 0;

  for (int proc = 1; proc < num_procs; ++proc) {
    remaining_blocks -= blocks_per_proc;
    blocks_per_proc = remaining_blocks / (num_procs - proc);
    block_sizes[proc] = blocks_per_proc * width_;
    displs[proc] = displs[proc - 1] + block_sizes[proc - 1];
  }

  int local_rows = block_sizes[rank] / width_;

  int halo_top = 0;
  if (rank > 0) {
    halo_top = 1;
  }

  int halo_bottom = 0;
  if (rank < num_procs - 1) {
    halo_bottom = 1;
  }

  int extended_rows = local_rows + halo_top + halo_bottom;

  std::vector<double> local_image(extended_rows * width_);

  scatterv(world_, image_.data(), block_sizes, displs, local_image.data() + (halo_top * width_), local_rows * width_,
           0);

  if (halo_top != 0) {
    world_.send(rank - 1, 0, local_image.data() + (halo_top * width_), width_);
    world_.recv(rank - 1, 1, local_image.data(), width_);
  }
  if (halo_bottom != 0) {
    world_.send(rank + 1, 1, local_image.data() + ((halo_top + local_rows - 1) * width_), width_);
    world_.recv(rank + 1, 0, local_image.data() + ((halo_top + local_rows) * width_), width_);
  }

  std::vector<double> kernel = {0.0625, 0.125, 0.0625, 0.125, 0.25, 0.125, 0.0625, 0.125, 0.0625};

  std::vector<double> local_filtered_image(local_rows * width_, 0);

#pragma omp parallel for schedule(static)
  for (int i = 1; i < extended_rows - 1; ++i) {
    for (int j = 1; j < width_ - 1; ++j) {
      double sum = 0.0;
      for (int di = -1; di <= 1; ++di) {
        for (int dj = -1; dj <= 1; ++dj) {
          int ni = i + di;
          int nj = j + dj;
          sum += local_image[(ni * width_) + nj] * kernel[((di + 1) * 3) + (dj + 1)];
        }
      }
      if (i >= halo_top && i < halo_top + local_rows) {
        local_filtered_image[((i - halo_top) * width_) + j] = sum;
      }
    }
  }

  gatherv(world_, local_filtered_image.data(), local_rows * width_, filtered_image_.data(), block_sizes, displs, 0);

  return true;
}

bool sozonov_i_image_filtering_block_partitioning_all::TestTaskALL::PostProcessingImpl() {
  if (world_.rank() == 0) {
    auto *out = reinterpret_cast<double *>(task_data->outputs[0]);
    std::ranges::copy(filtered_image_.begin(), filtered_image_.end(), out);
  }
  return true;
}