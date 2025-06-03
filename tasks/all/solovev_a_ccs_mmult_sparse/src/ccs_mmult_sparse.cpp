#include "all/solovev_a_ccs_mmult_sparse/include/ccs_mmult_sparse.hpp"

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/collectives/all_gather.hpp>
#include <boost/mpi/collectives/all_gatherv.hpp>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/reduce.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/complex.hpp>  // NOLINT
#include <boost/serialization/vector.hpp>   // NOLINT
#include <cmath>
#include <complex>
#include <cstddef>
#include <functional>
#include <numeric>
#include <thread>
#include <utility>
#include <vector>

#include "core/util/include/util.hpp"

void solovev_a_matrix_all::SeqMatMultCcs::ComputeColumnRange(int rank, int size, int total_cols, int& start_col,
                                                             int& end_col) {
  if (size <= 0) {
    start_col = total_cols;
    end_col = total_cols;
    return;
  }

  int cols_per_process = total_cols / size;
  int remaining_cols = total_cols % size;

  start_col = rank * cols_per_process + std::min(rank, remaining_cols);
  int extra_col = (rank < remaining_cols) ? 1 : 0;
  end_col = start_col + cols_per_process + extra_col;

  if (start_col >= total_cols || end_col > total_cols) {
    start_col = total_cols;
    end_col = total_cols;
  }
}

void solovev_a_matrix_all::SeqMatMultCcs::ComputeSequential(
    const std::vector<int>& col_indices, std::vector<std::vector<std::pair<std::complex<double>, int>>>& column_results,
    int start_col, int end_col,
    const std::function<void(int, std::vector<std::pair<std::complex<double>, int>>&)>& func) {
  for (int j = start_col; j < end_col; ++j) {
    func(col_indices[j - start_col], column_results[j - start_col]);
  }
}

void solovev_a_matrix_all::SeqMatMultCcs::ComputeParallel(
    const std::vector<int>& col_indices, std::vector<std::vector<std::pair<std::complex<double>, int>>>& column_results,
    int start_col, int end_col, int num_threads,
    const std::function<void(int, std::vector<std::pair<std::complex<double>, int>>&)>& func) {
  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  int num_cols = end_col - start_col;
  int base_cols = num_cols / num_threads;
  int rem_cols = num_cols % num_threads;
  int thread_start = start_col;

  for (int t = 0; t < num_threads; ++t) {
    int this_chunk = base_cols + (t < rem_cols ? 1 : 0);
    int thread_end = thread_start + this_chunk;

    threads.emplace_back([=, &column_results, &col_indices, &func]() {
      for (int j = thread_start; j < thread_end; ++j) {
        func(col_indices[j - start_col], column_results[j - start_col]);
      }
    });

    thread_start = thread_end;
  }

  for (auto& th : threads) {
    th.join();
  }
}

void solovev_a_matrix_all::SeqMatMultCcs::ComputeColumn(
    int col_idx, std::vector<std::pair<std::complex<double>, int>>& column_data) {
  if (col_idx >= M2_->c_n) {
    return;
  }

  int cs2 = M2_->col_p[col_idx];
  int ce2 = M2_->col_p[col_idx + 1];
  if (cs2 < 0 || ce2 > M2_->n_z || cs2 > ce2) {
    return;
  }

  column_data.clear();
  column_data.reserve(std::min(M1_->r_n, M2_->n_z));

  for (int i = 0; i < M1_->r_n; ++i) {
    std::complex<double> sum(0.0, 0.0);

    for (int k = 0; k < M1_->c_n; ++k) {
      int cs1 = M1_->col_p[k];
      int ce1 = M1_->col_p[k + 1];

      for (int p = cs1; p < ce1; ++p) {
        if (M1_->row[p] != i) {
          continue;
        }
        for (int q = cs2; q < ce2; ++q) {
          if (M2_->row[q] != k) {
            continue;
          }
          sum += M1_->val[p] * M2_->val[q];
        }
      }
    }

    if (std::abs(sum.real()) > 1e-10 || std::abs(sum.imag()) > 1e-10) {
      column_data.emplace_back(sum, i);
    }
  }
}

void solovev_a_matrix_all::SeqMatMultCcs::FillLocalData(
    const std::vector<std::vector<std::pair<std::complex<double>, int>>>& column_results,
    std::vector<std::complex<double>>& local_val, std::vector<int>& local_row, std::vector<int>& local_col_p,
    int& local_n_z) {
  local_val.clear();
  local_row.clear();
  local_col_p.assign(column_results.size() + 1, 0);
  local_n_z = 0;

  for (size_t j = 0; j < column_results.size(); ++j) {
    const auto& col_data = column_results[j];
    for (const auto& pr : col_data) {
      local_val.push_back(pr.first);
      local_row.push_back(pr.second);
    }
    local_n_z += static_cast<int>(col_data.size());
    local_col_p[j + 1] = local_n_z;
  }
}

void solovev_a_matrix_all::SeqMatMultCcs::CountColumns(
    const std::vector<std::vector<std::pair<std::complex<double>, int>>>& column_results,
    std::vector<int>& local_col_counts, int start_col) {
  for (size_t j = 0; j < column_results.size(); ++j) {
    local_col_counts[start_col + j] = static_cast<int>(column_results[j].size());
  }
}

bool solovev_a_matrix_all::SeqMatMultCcs::PreProcessingImpl() {
  if (world_.rank() == 0) {
    M1_ = reinterpret_cast<MatrixInCcsSparse*>(task_data->inputs[0]);
    M2_ = reinterpret_cast<MatrixInCcsSparse*>(task_data->inputs[1]);
    M3_ = MatrixInCcsSparse(M1_->r_n, M2_->c_n, 0);
  } else {
    M1_ = new MatrixInCcsSparse();
    M2_ = new MatrixInCcsSparse();
    M3_ = MatrixInCcsSparse(0, 0, 0);
  }
  return true;
}

bool solovev_a_matrix_all::SeqMatMultCcs::ValidationImpl() {
  if (world_.rank() == 0) {
    auto* m1 = reinterpret_cast<MatrixInCcsSparse*>(task_data->inputs[0]);
    auto* m2 = reinterpret_cast<MatrixInCcsSparse*>(task_data->inputs[1]);
    return (m1->c_n == m2->r_n);
  }

  return true;
}

bool solovev_a_matrix_all::SeqMatMultCcs::RunImpl() {
  int rank = world_.rank();
  int size = world_.size();

  std::vector<std::complex<double>> local_val;
  std::vector<int> local_row;
  std::vector<int> local_col_p;
  int local_n_z = 0;

  boost::mpi::broadcast(world_, *M1_, 0);
  boost::mpi::broadcast(world_, *M2_, 0);

  M3_ = MatrixInCcsSparse(M1_->r_n, M2_->c_n, 0);

  int total_cols = M2_->c_n;
  int start_col = 0;
  int end_col = 0;

  ComputeColumnRange(rank, size, total_cols, start_col, end_col);

  int num_cols = end_col - start_col;
  std::vector<std::vector<std::pair<std::complex<double>, int>>> column_results(num_cols);
  std::vector<int> col_indices(num_cols);
  std::iota(col_indices.begin(), col_indices.end(), start_col);

  auto compute_column_lambda = [&](int col_idx, std::vector<std::pair<std::complex<double>, int>>& column_data) {
    ComputeColumn(col_idx, column_data);
  };

  if (end_col > start_col) {
    int max_threads = ppc::util::GetPPCNumThreads();
    int num_threads = std::max(1, std::min(max_threads, num_cols));

    if (num_cols < num_threads * 2) {
      ComputeSequential(col_indices, column_results, start_col, end_col, compute_column_lambda);
    } else {
      ComputeParallel(col_indices, column_results, start_col, end_col, num_threads, compute_column_lambda);
    }
  }

  FillLocalData(column_results, local_val, local_row, local_col_p, local_n_z);

  std::vector<int> all_n_z(size);
  boost::mpi::all_gather(world_, local_n_z, all_n_z);

  int total_n_z = std::accumulate(all_n_z.begin(), all_n_z.end(), 0);
  std::vector<int> displs(size, 0);
  for (int i = 1; i < size; ++i) {
    displs[i] = displs[i - 1] + all_n_z[i - 1];
  }

  M3_.val.resize(total_n_z);
  M3_.row.resize(total_n_z);
  M3_.col_p.resize(total_cols + 1);
  M3_.n_z = total_n_z;

  if (total_n_z > 0) {
    boost::mpi::all_gatherv(world_, local_val, M3_.val, all_n_z, displs);
    boost::mpi::all_gatherv(world_, local_row, M3_.row, all_n_z, displs);
  }

  std::vector<int> local_col_counts(total_cols, 0);
  CountColumns(column_results, local_col_counts, start_col);

  std::vector<int> global_col_counts(total_cols);
  boost::mpi::reduce(world_, local_col_counts, global_col_counts, std::plus<>(), 0);

  if (rank == 0) {
    M3_.col_p[0] = 0;
    for (int j = 0; j < total_cols; ++j) {
      M3_.col_p[j + 1] = M3_.col_p[j] + global_col_counts[j];
    }
  }

  boost::mpi::broadcast(world_, M3_.col_p, 0);
  boost::mpi::broadcast(world_, M3_, 0);

  return true;
}

bool solovev_a_matrix_all::SeqMatMultCcs::PostProcessingImpl() {
  if (world_.rank() == 0) {
    *reinterpret_cast<MatrixInCcsSparse*>(task_data->outputs[0]) = M3_;
  }
  return true;
}
