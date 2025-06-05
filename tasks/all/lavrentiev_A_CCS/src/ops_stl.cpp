#include "all/lavrentiev_A_CCS/include/ops_stl.hpp"

#include <algorithm>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/gatherv.hpp>
#include <boost/serialization/vector.hpp>  // NOLINT(*-include-cleaner)
#include <cmath>
#include <cstddef>
#include <numeric>
#include <thread>
#include <utility>
#include <vector>

#include "core/util/include/util.hpp"

lavrentiev_a_ccs_all::Sparse lavrentiev_a_ccs_all::CCSALL::ConvertToSparse(std::pair<int, int> size,
                                                                           const std::vector<double> &values) {
  auto [nsize, elements, rows, columns_sum] = Sparse();
  columns_sum.resize(size.second);
  for (int i = 0; i < size.second; ++i) {
    for (int j = 0; j < size.first; ++j) {
      if (values[i + (size.second * j)] != 0) {
        elements.emplace_back(values[i + (size.second * j)]);
        rows.emplace_back(j);
        columns_sum[i] += 1;
      }
    }
    if (i != size.second - 1) {
      columns_sum[i + 1] = columns_sum[i];
    }
  }
  return {.size = size, .elements = elements, .rows = rows, .columnsSum = columns_sum};
}

lavrentiev_a_ccs_all::Sparse lavrentiev_a_ccs_all::CCSALL::Transpose(const Sparse &sparse) {
  auto [size, elements, rows, columns_sum] = Sparse();
  size.first = sparse.size.second;
  size.second = sparse.size.first;
  int need_size = std::max(sparse.size.first, sparse.size.second);
  std::vector<std::vector<std::pair<double, int>>> new_elements_and_rows(need_size);
  int counter = 0;
  for (int i = 0; i < static_cast<int>(sparse.columnsSum.size()); ++i) {
    for (int j = 0; j < GetElementsCount(i, sparse.columnsSum); ++j) {
      new_elements_and_rows[sparse.rows[counter]].emplace_back(sparse.elements[counter], i);
      counter++;
    }
  }
  elements.reserve(counter);
  rows.reserve(counter);
  for (int i = 0; i < static_cast<int>(new_elements_and_rows.size()); ++i) {
    for (int j = 0; j < static_cast<int>(new_elements_and_rows[i].size()); ++j) {
      elements.emplace_back(new_elements_and_rows[i][j].first);
      rows.emplace_back(new_elements_and_rows[i][j].second);
    }
    i > 0 ? columns_sum.emplace_back(new_elements_and_rows[i].size() + columns_sum[i - 1])
          : columns_sum.emplace_back(new_elements_and_rows[i].size());
  }
  return {.size = size, .elements = elements, .rows = rows, .columnsSum = columns_sum};
}

int lavrentiev_a_ccs_all::CCSALL::CalculateStartIndex(int index, const std::vector<int> &columns_sum) {
  return index == 0 ? 0 : columns_sum[index] - GetElementsCount(index, columns_sum);
}

lavrentiev_a_ccs_all::Sparse lavrentiev_a_ccs_all::CCSALL::MatMul(const Sparse &matrix1, const Sparse &matrix2,
                                                                  int interval_begin, int interval_end) {
  Sparse temporary_matrix;
  int resize_data = static_cast<int>(matrix2.columnsSum.size() * matrix1.columnsSum.size());
  std::vector<std::thread> threads(ppc::util::GetPPCNumThreads());
  temporary_matrix.columnsSum.resize(matrix2.size.second);
  temporary_matrix.elements.resize(resize_data);
  temporary_matrix.rows.resize(resize_data);
  auto matrix_multiplicator = [&](int begin, int end) {
    for (int i = begin; i != end; ++i) {
      if (temporary_matrix.columnsSum[i] == 0) {
        temporary_matrix.columnsSum[i]++;
      }
      for (int j = 0; j < static_cast<int>(matrix1.columnsSum.size()); ++j) {
        double s = Accumulate(i, j, matrix1, matrix2);
        if (s != 0) {
          temporary_matrix.elements[(i * matrix2.size.second) + j] = s;
          temporary_matrix.rows[(i * matrix2.size.second) + j] = j;
          temporary_matrix.columnsSum[i]++;
        }
      }
    }
  };
  int thread_data_amount = (interval_end - interval_begin) / ppc::util::GetPPCNumThreads();
  for (size_t i = 0; i < threads.size(); ++i) {
    if (i != threads.size() - 1) {
      threads[i] = std::thread(matrix_multiplicator, interval_begin + (i * thread_data_amount),
                               interval_begin + ((i + 1) * thread_data_amount));
    } else {
      threads[i] = std::thread(matrix_multiplicator, interval_begin + (i * thread_data_amount),
                               interval_begin + ((i + 1) * thread_data_amount) +
                                   ((interval_end - interval_begin) % ppc::util::GetPPCNumThreads()));
    }
  }
  std::ranges::for_each(threads, [&](std::thread &thread) { thread.join(); });
  std::vector<double> elements;
  std::vector<int> rows;
  elements.reserve(resize_data / 10);
  rows.reserve(resize_data / 10);
  for (int i = 0; i < resize_data; ++i) {
    if (temporary_matrix.elements[i] != 0.0) {
      elements.emplace_back(temporary_matrix.elements[i]);
      rows.emplace_back(temporary_matrix.rows[i]);
    }
  }
  return {.size = temporary_matrix.size, .elements = elements, .rows = rows, .columnsSum = temporary_matrix.columnsSum};
}

int lavrentiev_a_ccs_all::CCSALL::GetElementsCount(int index, const std::vector<int> &columns_sum) {
  if (index == 0) {
    return columns_sum[index];
  }
  return columns_sum[index] - columns_sum[index - 1];
}

std::vector<double> lavrentiev_a_ccs_all::CCSALL::ConvertFromSparse(const Sparse &matrix) {
  std::vector<double> nmatrix(matrix.size.first * matrix.size.second);
  int counter = 0;
  for (size_t i = 0; i < matrix.columnsSum.size(); ++i) {
    for (int j = 0; j < GetElementsCount(static_cast<int>(i), matrix.columnsSum); ++j) {
      nmatrix[i + (matrix.size.second * matrix.rows[counter])] = matrix.elements[counter];
      counter++;
    }
  }
  return nmatrix;
}

void lavrentiev_a_ccs_all::CCSALL::GetDisplacements() {
  displ_.resize(world_.size());
  int amount = static_cast<int>(B_.columnsSum.size()) % world_.size();
  int count = static_cast<int>(B_.columnsSum.size()) / world_.size();
  if (static_cast<int>(B_.columnsSum.size()) == 0) {
    return;
  }
  for (int i = 0; i < world_.size(); ++i) {
    if (i != 0) {
      displ_[i] = count + displ_[i - 1];
      if (amount != 0) {
        displ_[i]++;
        amount--;
      }
    }
  }
  displ_.emplace_back(static_cast<int>(B_.columnsSum.size()));
}
double lavrentiev_a_ccs_all::CCSALL::Accumulate(int i_index, int j_index, const Sparse &matrix1,
                                                const Sparse &matrix2) {
  double sum = 0.0;
  for (int x = 0; x < GetElementsCount(j_index, matrix1.columnsSum); x++) {
    for (int y = 0; y < GetElementsCount(i_index, matrix2.columnsSum); y++) {
      int m1_start_index = CalculateStartIndex(j_index, matrix1.columnsSum);
      int m2_start_index = CalculateStartIndex(i_index, matrix2.columnsSum);
      if (matrix1.rows[m1_start_index + x] == matrix2.rows[m2_start_index + y]) {
        sum += matrix1.elements[x + m1_start_index] * matrix2.elements[y + m2_start_index];
      }
    }
  }
  return sum;
}

bool lavrentiev_a_ccs_all::CCSALL::PreProcessingImpl() {
  if (world_.rank() == 0) {
    A_.size = {static_cast<int>(task_data->inputs_count[0]), static_cast<int>(task_data->inputs_count[1])};
    B_.size = {static_cast<int>(task_data->inputs_count[2]), static_cast<int>(task_data->inputs_count[3])};
    if (IsEmpty()) {
      return true;
    }
    auto *in_ptr = reinterpret_cast<double *>(task_data->inputs[0]);
    auto am = std::vector<double>(in_ptr, in_ptr + (A_.size.first * A_.size.second));
    A_ = ConvertToSparse(A_.size, am);
    A_ = Transpose(A_);
    auto *in_ptr2 = reinterpret_cast<double *>(task_data->inputs[1]);
    auto bm = std::vector<double>(in_ptr2, in_ptr2 + (B_.size.first * B_.size.second));
    B_ = ConvertToSparse(B_.size, bm);
    GetDisplacements();
  }
  return true;
}

bool lavrentiev_a_ccs_all::CCSALL::IsEmpty() const {
  return A_.size.first * A_.size.second == 0 || B_.size.first * B_.size.second == 0;
}

void lavrentiev_a_ccs_all::CCSALL::CollectSizes() {
  if (world_.rank() != 0) {
    world_.send(0, 1, static_cast<int>(Process_data_.columnsSum.size()));
    world_.send(0, 0, static_cast<int>(Process_data_.elements.size()));
  } else {
    sum_sizes_.resize(world_.size(), static_cast<int>(Process_data_.columnsSum.size()));
    elements_sizes_.resize(world_.size(), static_cast<int>(Process_data_.elements.size()));
    for (int i = 1; i < world_.size(); ++i) {
      world_.recv(i, 1, sum_sizes_[i]);
      world_.recv(i, 0, elements_sizes_[i]);
    }
  }
}
bool lavrentiev_a_ccs_all::CCSALL::ValidationImpl() {
  if (world_.rank() == 0) {
    return task_data->inputs_count[0] * task_data->inputs_count[3] == task_data->outputs_count[0] &&
           task_data->inputs_count[0] == task_data->inputs_count[3] &&
           task_data->inputs_count[1] == task_data->inputs_count[2];
  }
  return true;
}

bool lavrentiev_a_ccs_all::CCSALL::RunImpl() {
  boost::mpi::broadcast(world_, displ_, 0);
  boost::mpi::broadcast(world_, A_, 0);
  boost::mpi::broadcast(world_, B_, 0);
  if (displ_.empty()) {
    return true;
  }
  Process_data_ = MatMul(A_, B_, displ_[world_.rank()], displ_[world_.rank() + 1]);
  CollectSizes();
  if (world_.rank() == 0) {
    Answer_.columnsSum.clear();
    Answer_.elements.clear();
    Answer_.rows.clear();
    std::vector<int> columns_nums_collector(B_.columnsSum.size() * world_.size());
    auto size = std::accumulate(elements_sizes_.begin(), elements_sizes_.end(), 0);
    Answer_.elements.resize(size);
    Answer_.rows.resize(size);
    Answer_.columnsSum.reserve(B_.columnsSum.size());
    boost::mpi::gatherv(world_, Process_data_.elements, Answer_.elements.data(), elements_sizes_, 0);
    boost::mpi::gatherv(world_, Process_data_.rows, Answer_.rows.data(), elements_sizes_, 0);
    boost::mpi::gatherv(world_, Process_data_.columnsSum, columns_nums_collector.data(), sum_sizes_, 0);
    for (auto &column_sum : columns_nums_collector) {
      if (column_sum != 0) {
        Answer_.columnsSum.emplace_back(column_sum - 1);
      }
    }
    for (size_t i = 1; i < Answer_.columnsSum.size(); ++i) {
      Answer_.columnsSum[i] = Answer_.columnsSum[i] + Answer_.columnsSum[i - 1];
    }
    Answer_.size.first = B_.size.second;
    Answer_.size.second = B_.size.second;
  } else {
    boost::mpi::gatherv(world_, Process_data_.elements, 0);
    boost::mpi::gatherv(world_, Process_data_.rows, 0);
    boost::mpi::gatherv(world_, Process_data_.columnsSum, 0);
  }
  return true;
}

bool lavrentiev_a_ccs_all::CCSALL::PostProcessingImpl() {
  if (world_.rank() == 0) {
    auto answer = ConvertFromSparse(Answer_);
    if (!answer.empty()) {
      std::ranges::copy(answer, reinterpret_cast<double *>(task_data->outputs[0]));
    }
  }
  return true;
}
