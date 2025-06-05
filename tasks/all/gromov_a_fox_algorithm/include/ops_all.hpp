#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cstddef>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace gromov_a_fox_algorithm_all {
void ExtractBlock(const std::vector<double>& source_matrix, double* block_buffer, int matrix_width, int block_size,
                  int block_row_idx, int block_col_idx);
void MultBlocks(const double* matrix_a, const double* matrix_b, double* matrix_c, int block_size);
void FoxStep(boost::mpi::communicator& mpi_comm, int process_rank, int active_process_count, int block_size,
             std::vector<double>& local_matrix_a, std::vector<double>& local_matrix_b,
             std::vector<double>& local_matrix_c);
std::vector<double> Scatter(const std::vector<double>& source_matrix, std::size_t matrix_size, int grid_size,
                            int block_size);
std::vector<double> Gather(const std::vector<double>& gathered_buffer, std::size_t matrix_size, int grid_size,
                           int block_size);
int ProcessGrid(int total_process_count, std::size_t matrix_size);
class TestTaskAll : public ppc::core::Task {
 public:
  explicit TestTaskAll(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> inputMatrixA_, inputMatrixB_;
  std::vector<double> resultMatrix_;
  std::size_t matrixSize_ = 0;
  std::size_t blockDimension_ = 0;
  std::size_t matrixElements_{};
  boost::mpi::communicator mpiCommunicator_;
};

}  // namespace gromov_a_fox_algorithm_all