#include "all/lysov_i_matrix_multiplication_Fox_algorithm/include/ops_all.hpp"

#include <mpi.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/status.hpp>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <vector>

#include "boost/mpi/collectives/broadcast.hpp"
#include "boost/mpi/collectives/gather.hpp"
#include "boost/mpi/collectives/scatter.hpp"
#include "oneapi/tbb/parallel_for.h"
#include "oneapi/tbb/task_arena.h"
int lysov_i_matrix_multiplication_fox_algorithm_mpi_tbb::ComputeProcessGrid(int world_size, std::size_t n) {
  int q = static_cast<int>(std::floor(std::sqrt(world_size)));
  while (q > 1 && (world_size % q != 0 || (n % q) != 0)) {
    --q;
  }
  return std::max(q, 1);
}
void lysov_i_matrix_multiplication_fox_algorithm_mpi_tbb::ExtractSubmatrixBlock(const std::vector<double>& matrix,
                                                                                double* block, int total_columns,
                                                                                int block_size, int block_row_index,
                                                                                int block_col_index) {
  const double* src0 =
      matrix.data() + ((block_row_index * block_size) * total_columns) + (block_col_index * block_size);
  for (int i = 0; i < block_size; ++i) {
    std::memcpy(block + (i * block_size), src0 + (i * total_columns), block_size * sizeof(double));
  }
}

void lysov_i_matrix_multiplication_fox_algorithm_mpi_tbb::MultiplyMatrixBlocks(const double* a, const double* b,
                                                                               double* c, int block_size) {
  tbb::parallel_for(
      tbb::blocked_range<int>(0, block_size),
      [&](const tbb::blocked_range<int>& r) {
        for (int i = r.begin(); i < r.end(); ++i) {
          const double* ai = a + (i * block_size);
          double* ci = c + (i * block_size);
          for (int k = 0; k < block_size; ++k) {
            double aik = ai[k];
            const double* bk = b + (k * block_size);
            for (int j = 0; j < block_size; ++j) {
              ci[j] += aik * bk[j];
            }
          }
        }
      },
      tbb::auto_partitioner());
}
std::vector<double> lysov_i_matrix_multiplication_fox_algorithm_mpi_tbb::ScatterMatrix(const std::vector<double>& m,
                                                                                       std::size_t n, int q, int k) {
  std::vector<double> buf(n * n);
  int idx = 0;
  for (int br = 0; br < q; ++br) {
    for (int bc = 0; bc < q; ++bc) {
      ExtractSubmatrixBlock(m, buf.data() + idx, static_cast<int>(n), k, br, bc);
      idx += k * k;
    }
  }
  return buf;
}

std::vector<double> lysov_i_matrix_multiplication_fox_algorithm_mpi_tbb::GatherMatrix(const std::vector<double>& buf,
                                                                                      std::size_t n, int q, int k) {
  std::vector<double> c(n * n, 0.0);
  int idx = 0;
  for (int br = 0; br < q; ++br) {
    for (int bc = 0; bc < q; ++bc) {
      for (int i = 0; i < k; ++i) {
        double* dest = c.data() + (((br * k) + i) * n) + (bc * k);
        const double* src = buf.data() + idx + (i * k);
        std::memcpy(dest, src, k * sizeof(double));
      }
      idx += k * k;
    }
  }
  return c;
}
void lysov_i_matrix_multiplication_fox_algorithm_mpi_tbb::PerformFoxAlgorithmStep(boost::mpi::communicator& world,
                                                                                  int rank, int cnt_work_process, int k,
                                                                                  std::vector<double>& local_a,
                                                                                  std::vector<double>& local_b,
                                                                                  std::vector<double>& local_c) {
  if (cnt_work_process == 1) {
    MultiplyMatrixBlocks(local_a.data(), local_b.data(), local_c.data(), k);
    return;
  }

  std::vector<double> temp_a(k * k);
  std::vector<double> temp_b(k * k);
  int row = rank / cnt_work_process;
  int col = rank % cnt_work_process;

  for (int l = 0; l < cnt_work_process; ++l) {
    if (col == (row + l) % cnt_work_process) {
      for (int tc = 0; tc < cnt_work_process; ++tc) {
        if (tc == col) {
          continue;
        }
        int target = (row * cnt_work_process) + tc;
        world.send(target, 0, local_a.data(), k * k);
      }
      temp_a = local_a;
    } else {
      int sender = (row * cnt_work_process) + ((row + l) % cnt_work_process);
      world.recv(sender, 0, temp_a.data(), k * k);
    }

    world.barrier();
    MultiplyMatrixBlocks(temp_a.data(), local_b.data(), local_c.data(), k);
    int send_to = (((row - 1 + cnt_work_process) % cnt_work_process) * cnt_work_process) + col;
    int recv_from = (((row + 1) % cnt_work_process) * cnt_work_process) + col;

    MPI_Sendrecv(local_b.data(), k * k, MPI_DOUBLE, send_to, 0, temp_b.data(), k * k, MPI_DOUBLE, recv_from, 0, world,
                 MPI_STATUS_IGNORE);

    local_b.swap(temp_b);
  }
}

bool lysov_i_matrix_multiplication_fox_algorithm_mpi_tbb ::TestTaskMPITBB::PreProcessingImpl() {
  if (world_.rank() == 0) {
    n_ = reinterpret_cast<std::size_t*>(task_data->inputs[0])[0];
    block_size_ = reinterpret_cast<std::size_t*>(task_data->inputs[3])[0];
    elements_ = n_ * n_;
    a_.resize(elements_);
    b_.resize(elements_);
    resultC_.clear();
    b_.resize(elements_, 0.0);
    std::copy(reinterpret_cast<double*>(task_data->inputs[1]),
              reinterpret_cast<double*>(task_data->inputs[1]) + (n_ * n_), a_.begin());
    std::copy(reinterpret_cast<double*>(task_data->inputs[2]),
              reinterpret_cast<double*>(task_data->inputs[2]) + (n_ * n_), b_.begin());
    resultC_.assign(elements_, 0.0);
  }
  return true;
}

bool lysov_i_matrix_multiplication_fox_algorithm_mpi_tbb::TestTaskMPITBB::ValidationImpl() {
  if (world_.rank() != 0) {
    return true;
  }
  n_ = *reinterpret_cast<std::size_t*>(task_data->inputs[0]);
  std::size_t total = n_ * n_;
  if (total == 0) {
    return false;
  }
  auto& ic = task_data->inputs_count;
  auto& oc = task_data->outputs_count;
  if (ic.size() != 3 || oc.size() != 1) {
    return false;
  }
  if (ic[0] != total || ic[1] != total || ic[2] != 1) {
    return false;
  }
  if (oc[0] != total) {
    return false;
  }
  auto* ptr_a = reinterpret_cast<double*>(task_data->inputs[1]);
  auto* ptr_b = reinterpret_cast<double*>(task_data->inputs[2]);
  return (ptr_a != nullptr && ptr_b != nullptr);
  ;
}

bool lysov_i_matrix_multiplication_fox_algorithm_mpi_tbb::TestTaskMPITBB::RunImpl() {
  int rank = world_.rank();
  int size = world_.size();
  boost::mpi::broadcast(world_, n_, 0);
  elements_ = static_cast<int>(n_ * n_);
  boost::mpi::broadcast(world_, elements_, 0);
  int q = ComputeProcessGrid(size, n_);
  int k = static_cast<int>(n_ / q);
  int process_group = (rank < q * q) ? 1 : MPI_UNDEFINED;
  MPI_Comm computation_comm = MPI_COMM_NULL;
  MPI_Comm_split(world_, process_group, rank, &computation_comm);
  if (process_group == MPI_UNDEFINED) {
    return true;
  }
  boost::mpi::communicator my_comm(computation_comm, boost::mpi::comm_take_ownership);
  rank = my_comm.rank();
  std::vector<double> scatter_a(elements_);
  std::vector<double> scatter_b(elements_);
  if (rank == 0) {
    scatter_a = ScatterMatrix(a_, n_, q, k);
    scatter_b = ScatterMatrix(b_, n_, q, k);
  }
  std::vector<double> local_a(k * k);
  std::vector<double> local_b(k * k);
  std::vector<double> local_c(k * k, 0.0);
  boost::mpi::scatter(my_comm, scatter_a, local_a.data(), static_cast<int>(local_a.size()), 0);
  boost::mpi::scatter(my_comm, scatter_b, local_b.data(), static_cast<int>(local_b.size()), 0);
  tbb::task_arena arena;
  arena.execute([&] { PerformFoxAlgorithmStep(my_comm, rank, q, k, local_a, local_b, local_c); });
  std::vector<double> gathered(elements_);
  boost::mpi::gather(my_comm, local_c.data(), static_cast<int>(local_c.size()), gathered, 0);

  if (rank == 0) {
    resultC_ = GatherMatrix(gathered, n_, q, k);
  }
  return true;
}

bool lysov_i_matrix_multiplication_fox_algorithm_mpi_tbb::TestTaskMPITBB::PostProcessingImpl() {
  if (world_.rank() == 0) {
    std::ranges::copy(resultC_, reinterpret_cast<double*>(task_data->outputs[0]));
  }
  return true;
}
