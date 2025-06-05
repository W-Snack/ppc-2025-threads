#include "all/gromov_a_fox_algorithm/include/ops_all.hpp"

#include <mpi.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>
#include <tbb/tbb.h>

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <vector>

#include "boost/mpi/collectives/broadcast.hpp"
#include "boost/mpi/collectives/gather.hpp"
#include "boost/mpi/collectives/scatter.hpp"
#include "oneapi/tbb/parallel_for.h"
#include "oneapi/tbb/task_arena.h"

namespace gromov_a_fox_algorithm_all {

bool TestTaskAll::PreProcessingImpl() {
  if (mpiCommunicator_.rank() == 0) {
    auto* input_data_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
    std::size_t input_data_size = task_data->inputs_count[0];
    matrixSize_ = static_cast<std::size_t>(std::sqrt(input_data_size / 2));
    matrixElements_ = matrixSize_ * matrixSize_;

    if (input_data_size != 2 * matrixElements_) {
      return false;
    }

    inputMatrixA_.resize(matrixElements_);
    inputMatrixB_.resize(matrixElements_);
    resultMatrix_.resize(matrixElements_, 0.0);

    std::copy(input_data_ptr, input_data_ptr + matrixElements_, inputMatrixA_.begin());
    std::copy(input_data_ptr + matrixElements_, input_data_ptr + (2 * matrixElements_), inputMatrixB_.begin());
  }
  return true;
}

bool TestTaskAll::ValidationImpl() {
  if (mpiCommunicator_.rank() != 0) {
    return true;
  }
  auto& input_counts = task_data->inputs_count;
  auto& output_counts = task_data->outputs_count;

  if (input_counts.size() != 1 || output_counts.size() != 1) {
    return false;
  }

  std::size_t input_data_size = input_counts[0];
  matrixSize_ = static_cast<std::size_t>(std::sqrt(input_data_size / 2));
  matrixElements_ = matrixSize_ * matrixSize_;
  if (input_data_size != 2 * matrixElements_ || output_counts[0] != matrixElements_) {
    return false;
  }

  auto* input_data_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  return input_data_ptr != nullptr;
}

bool TestTaskAll::RunImpl() {
  int process_rank = mpiCommunicator_.rank();
  int process_count = mpiCommunicator_.size();
  boost::mpi::broadcast(mpiCommunicator_, matrixSize_, 0);
  matrixElements_ = matrixSize_ * matrixSize_;
  boost::mpi::broadcast(mpiCommunicator_, matrixElements_, 0);
  int grid_size = ProcessGrid(process_count, matrixSize_);
  blockDimension_ = matrixSize_ / grid_size;
  int block_size = static_cast<int>(blockDimension_);
  int process_group = (process_rank < grid_size * grid_size) ? 1 : MPI_UNDEFINED;
  MPI_Comm compute_comm = MPI_COMM_NULL;
  MPI_Comm_split(mpiCommunicator_, process_group, process_rank, &compute_comm);
  if (process_group == MPI_UNDEFINED) {
    return true;
  }
  boost::mpi::communicator local_mpi_comm(compute_comm, boost::mpi::comm_take_ownership);
  process_rank = local_mpi_comm.rank();
  std::vector<double> scatter_matrix_a(matrixElements_);
  std::vector<double> scatter_matrix_b(matrixElements_);
  if (process_rank == 0) {
    scatter_matrix_a = Scatter(inputMatrixA_, matrixSize_, grid_size, block_size);
    scatter_matrix_b = Scatter(inputMatrixB_, matrixSize_, grid_size, block_size);
  }
  std::vector<double> local_matrix_a(block_size * block_size);
  std::vector<double> local_matrix_b(block_size * block_size);
  std::vector<double> local_matrix_c(block_size * block_size, 0.0);
  boost::mpi::scatter(local_mpi_comm, scatter_matrix_a, local_matrix_a.data(), static_cast<int>(local_matrix_a.size()),
                      0);
  boost::mpi::scatter(local_mpi_comm, scatter_matrix_b, local_matrix_b.data(), static_cast<int>(local_matrix_b.size()),
                      0);
  tbb::task_arena tbb_task_arena;
  tbb_task_arena.execute([&] {
    FoxStep(local_mpi_comm, process_rank, grid_size, block_size, local_matrix_a, local_matrix_b, local_matrix_c);
  });
  std::vector<double> gathered_matrix(matrixElements_);
  boost::mpi::gather(local_mpi_comm, local_matrix_c.data(), static_cast<int>(local_matrix_c.size()), gathered_matrix,
                     0);

  if (process_rank == 0) {
    resultMatrix_ = Gather(gathered_matrix, matrixSize_, grid_size, block_size);
  }
  return true;
}

bool TestTaskAll::PostProcessingImpl() {
  if (mpiCommunicator_.rank() == 0) {
    std::ranges::copy(resultMatrix_, reinterpret_cast<double*>(task_data->outputs[0]));
  }
  return true;
}

int ProcessGrid(int total_process_count, std::size_t matrix_size) {
  int grid_size = static_cast<int>(std::floor(std::sqrt(total_process_count)));
  while (grid_size > 1 && (total_process_count % grid_size != 0 || (matrix_size % grid_size) != 0)) {
    --grid_size;
  }
  return std::max(grid_size, 1);
}

void ExtractBlock(const std::vector<double>& source_matrix, double* block_buffer, int matrix_width, int block_size,
                  int block_row_idx, int block_col_idx) {
  const double* block_start_ptr =
      source_matrix.data() + ((block_row_idx * block_size) * matrix_width) + (block_col_idx * block_size);
  for (int row_idx = 0; row_idx < block_size; ++row_idx) {
    std::memcpy(block_buffer + (row_idx * block_size), block_start_ptr + (row_idx * matrix_width),
                block_size * sizeof(double));
  }
}

void MultBlocks(const double* matrix_a, const double* matrix_b, double* matrix_c, int block_size) {
  tbb::parallel_for(
      tbb::blocked_range<int>(0, block_size),
      [&](const tbb::blocked_range<int>& block_range) {
        for (int row_idx = block_range.begin(); row_idx < block_range.end(); ++row_idx) {
          const double* row_a = matrix_a + (row_idx * block_size);
          double* row_c = matrix_c + (row_idx * block_size);
          for (int inner_idx = 0; inner_idx < block_size; ++inner_idx) {
            double element_a = row_a[inner_idx];
            const double* row_b = matrix_b + (inner_idx * block_size);
            for (int col_idx = 0; col_idx < block_size; ++col_idx) {
              row_c[col_idx] += element_a * row_b[col_idx];
            }
          }
        }
      },
      tbb::auto_partitioner());
}

std::vector<double> Scatter(const std::vector<double>& source_matrix, std::size_t matrix_size, int grid_size,
                            int block_size) {
  std::vector<double> scatter_buffer(matrix_size * matrix_size);
  int buffer_index = 0;
  for (int block_row_idx = 0; block_row_idx < grid_size; ++block_row_idx) {
    for (int block_col_idx = 0; block_col_idx < grid_size; ++block_col_idx) {
      ExtractBlock(source_matrix, scatter_buffer.data() + buffer_index, static_cast<int>(matrix_size), block_size,
                   block_row_idx, block_col_idx);
      buffer_index += block_size * block_size;
    }
  }
  return scatter_buffer;
}

std::vector<double> Gather(const std::vector<double>& gathered_buffer, std::size_t matrix_size, int grid_size,
                           int block_size) {
  std::vector<double> result_matrix(matrix_size * matrix_size, 0.0);
  int buffer_index = 0;
  for (int block_row_idx = 0; block_row_idx < grid_size; ++block_row_idx) {
    for (int block_col_idx = 0; block_col_idx < grid_size; ++block_col_idx) {
      for (int row_idx = 0; row_idx < block_size; ++row_idx) {
        double* dest_ptr = result_matrix.data() + (((block_row_idx * block_size) + row_idx) * matrix_size) +
                           (block_col_idx * block_size);
        const double* source_ptr = gathered_buffer.data() + buffer_index + (row_idx * block_size);
        std::memcpy(dest_ptr, source_ptr, block_size * sizeof(double));
      }
      buffer_index += block_size * block_size;
    }
  }
  return result_matrix;
}

void FoxStep(boost::mpi::communicator& mpi_comm, int process_rank, int active_process_count, int block_size,
             std::vector<double>& local_matrix_a, std::vector<double>& local_matrix_b,
             std::vector<double>& local_matrix_c) {
  if (active_process_count == 1) {
    MultBlocks(local_matrix_a.data(), local_matrix_b.data(), local_matrix_c.data(), block_size);
    return;
  }

  std::vector<double> temp_matrix_a(block_size * block_size);
  std::vector<double> temp_matrix_b(block_size * block_size);

  int process_row = process_rank / active_process_count;
  int process_col = process_rank % active_process_count;

  for (int step_idx = 0; step_idx < active_process_count; ++step_idx) {
    if (process_col == (process_row + step_idx) % active_process_count) {
      for (int target_col_idx = 0; target_col_idx < active_process_count; ++target_col_idx) {
        if (target_col_idx == process_col) {
          continue;
        }
        int target_process = (process_row * active_process_count) + target_col_idx;
        mpi_comm.send(target_process, 0, local_matrix_a.data(), block_size * block_size);
      }
      temp_matrix_a = local_matrix_a;
    } else {
      int sender_process = (process_row * active_process_count) + ((process_row + step_idx) % active_process_count);
      mpi_comm.recv(sender_process, 0, temp_matrix_a.data(), block_size * block_size);
    }
    mpi_comm.barrier();
    MultBlocks(temp_matrix_a.data(), local_matrix_b.data(), local_matrix_c.data(), block_size);
    int send_to_process =
        (((process_row - 1 + active_process_count) % active_process_count) * active_process_count) + process_col;
    int recv_from_process = (((process_row + 1) % active_process_count) * active_process_count) + process_col;

    mpi_comm.send(send_to_process, 0, local_matrix_b.data(), block_size * block_size);
    mpi_comm.recv(recv_from_process, 0, temp_matrix_b.data(), block_size * block_size);

    mpi_comm.barrier();

    local_matrix_b.swap(temp_matrix_b);
  }
}

}  // namespace gromov_a_fox_algorithm_all