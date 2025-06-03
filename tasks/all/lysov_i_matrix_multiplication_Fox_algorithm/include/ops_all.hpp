#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cstddef>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace lysov_i_matrix_multiplication_fox_algorithm_mpi_tbb {
void ExtractSubmatrixBlock(const std::vector<double>& matrix, double* block, int total_columns, int block_size,
                           int block_row_index, int block_col_index);

void MultiplyMatrixBlocks(const double* a, const double* b, double* c, int block_size);
void PerformFoxAlgorithmStep(boost::mpi::communicator& my_world, int rank, int cnt_work_process, int k,
                             std::vector<double>& local_a, std::vector<double>& local_b, std::vector<double>& local_c);
void TrivialMatrixMultiplication(const std::vector<double>& matrix_a, const std::vector<double>& matrix_b,
                                 std::vector<double>& result_matrix, size_t matrix_size);
std::vector<double> ScatterMatrix(const std::vector<double>& m, std::size_t n, int q, int k);
std::vector<double> GatherMatrix(const std::vector<double>& buf, std::size_t n, int q, int k);
std::vector<double> GetRandomMatrix(size_t size, int min_gen_value, int max_gen_value);
int ComputeProcessGrid(int world_size, std::size_t n);
class TestTaskMPITBB : public ppc::core::Task {
 public:
  explicit TestTaskMPITBB(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> a_, b_;
  std::vector<double> resultC_;

  std::size_t n_ = 0;
  std::size_t block_size_ = 0;
  std::size_t elements_{};
  boost::mpi::communicator world_;
};

}  // namespace lysov_i_matrix_multiplication_fox_algorithm_mpi_tbb
