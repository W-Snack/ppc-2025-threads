#pragma once

#include <boost/serialization/access.hpp>
#include <complex>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "boost/mpi/communicator.hpp"
#include "core/task/include/task.hpp"

namespace solovev_a_matrix_all {

struct MatrixInCcsSparse {
  int r_n;
  int c_n;
  int n_z;

  std::vector<std::complex<double>> val;
  std::vector<int> row;
  std::vector<int> col_p;

  MatrixInCcsSparse(int r_nn = 0, int c_nn = 0, int n_zz = 0)
      : r_n(r_nn), c_n(c_nn), n_z(n_zz), val(n_zz), row(n_zz), col_p(c_n + 1) {}

  friend class boost::serialization::access;

  // clang-format off
  // NOLINTBEGIN(*)
  template <class Archive>
  void serialize(Archive& ar, const unsigned int /*version*/) {
    ar & r_n;
    ar & c_n;
    ar & n_z;
    ar & val;
    ar & row;
    ar & col_p;
  }
  // NOLINTEND(*)
  // clang-format on
};

class SeqMatMultCcs : public ppc::core::Task {
 public:
  explicit SeqMatMultCcs(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void ProcessPhase1(SeqMatMultCcs* self, int col, std::vector<int>& available);
  static void ProcessPhase2(SeqMatMultCcs* self, int col, std::vector<int>& available,
                            std::vector<std::complex<double>>& cask);
  static void NotifyCompletion(SeqMatMultCcs* self);
  static void WorkerLoop(SeqMatMultCcs* self);

 private:
  MatrixInCcsSparse *M1_, *M2_;
  MatrixInCcsSparse M3_;
  boost::mpi::communicator world_;

  static void ComputeColumnRange(int rank, int size, int total_cols, int& start_col, int& end_col);

  static void ComputeSequential(
      const std::vector<int>& col_indices,
      std::vector<std::vector<std::pair<std::complex<double>, int>>>& column_results, int start_col, int end_col,
      const std::function<void(int, std::vector<std::pair<std::complex<double>, int>>&)>& func);

  static void ComputeParallel(const std::vector<int>& col_indices,
                              std::vector<std::vector<std::pair<std::complex<double>, int>>>& column_results,
                              int start_col, int end_col, int num_threads,
                              const std::function<void(int, std::vector<std::pair<std::complex<double>, int>>&)>& func);

  void ComputeColumn(int col_idx, std::vector<std::pair<std::complex<double>, int>>& column_data);

  static void FillLocalData(const std::vector<std::vector<std::pair<std::complex<double>, int>>>& column_results,
                            std::vector<std::complex<double>>& local_val, std::vector<int>& local_row,
                            std::vector<int>& local_col_p, int& local_n_z);

  static void CountColumns(const std::vector<std::vector<std::pair<std::complex<double>, int>>>& column_results,
                           std::vector<int>& local_col_counts, int start_col);
};

}  // namespace solovev_a_matrix_all
