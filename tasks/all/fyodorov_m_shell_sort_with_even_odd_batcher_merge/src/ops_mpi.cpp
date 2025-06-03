#include "all/fyodorov_m_shell_sort_with_even_odd_batcher_merge/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>  // NOLINT: Требуется для сериализации std::vector в Boost.MPI
#include <cstddef>
#include <iostream>
#include <utility>
#include <vector>

#include "boost/mpi/collectives/gatherv.hpp"
#include "boost/mpi/collectives/scatterv.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif
namespace fyodorov_m_shell_sort_with_even_odd_batcher_merge_mpi {

// boost::mpi::communicator TestTaskMPI::world_;

bool TestTaskMPI::PreProcessingImpl() {
  unsigned int input_size = 0;
  if (world_.rank() == 0) {
    input_size = task_data->inputs_count[0];
  }
  boost::mpi::broadcast(world_, input_size, 0);

  input_.resize(input_size);
  if (world_.rank() == 0) {
    auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
    std::copy(in_ptr, in_ptr + input_size, input_.begin());
  }
  boost::mpi::broadcast(world_, input_, 0);

  unsigned int output_size = task_data->outputs_count[0];
  output_ = std::vector<int>(output_size, 0);

  std::cout << "rank " << world_.rank() << " input_ (first 10): ";
  for (size_t i = 0; i < std::min<size_t>(10, input_size); ++i) {
    std::cout << input_[i] << " ";
  }
  std::cout << '\n';

  return true;
}

bool TestTaskMPI::ValidationImpl() {
  if (task_data->inputs_count.empty() || task_data->outputs_count.empty()) {
    return false;
  }
  if ((task_data->inputs_count[0] > 0 && task_data->inputs[0] == nullptr) ||
      (task_data->outputs_count[0] > 0 && task_data->outputs[0] == nullptr)) {
    return false;
  }
  return true;
}

bool TestTaskMPI::RunImpl() {  // NOLINT
  int rank = world_.rank();
  int size = world_.size();

  boost::mpi::broadcast(world_, input_, 0);

  int n = static_cast<int>(input_.size());
  if (n == 0) {
    output_.clear();
    if (rank == 0) {
      for (int dest = 1; dest < size; ++dest) {
        world_.send(dest, 0, output_);
      }
    } else {
      world_.recv(0, 0, output_);
    }
    return true;
  }

  int local_n = n / size;
  int remainder = n % size;
  std::vector<int> sendcounts(size, local_n);
  std::vector<int> displs(size, 0);
  for (int i = 0; i < remainder; ++i) {
    sendcounts[i]++;
  }
  for (int i = 1; i < size; ++i) {
    displs[i] = displs[i - 1] + sendcounts[i - 1];
  }

  // Проверка корректности sendcounts и displs
  bool valid = true;
  int total_count = 0;
  if (sendcounts.size() != static_cast<size_t>(size) || displs.size() != static_cast<size_t>(size)) {
    valid = false;
  } else {
    for (int i = 0; i < size && valid; ++i) {
      if (sendcounts[i] < 0 || (i > 0 && (displs[i] < 0 || displs[i] >= n))) {
        valid = false;
      }
      total_count += sendcounts[i];
    }
    if (total_count != n) {
      valid = false;
    }
  }

  if (!valid) {
    std::cerr << "Error: Invalid sendcounts or displs configuration\n";
    output_.clear();
    return false;
  }

  std::vector<int> local_data(sendcounts[rank]);
  boost::mpi::scatterv(world_, input_, sendcounts, displs, local_data.data(), sendcounts[rank], 0);

  ShellSort(local_data);

  std::vector<int> gathered;
  if (rank == 0) {
    gathered.resize(n);
    boost::mpi::gatherv(world_, local_data.data(), sendcounts[rank], gathered.data(), sendcounts, displs, 0);
    if (!gathered.empty()) {
      MergeSortedBlocks(gathered, sendcounts, displs, size, world_);
    } else {
      output_.clear();
      for (int dest = 1; dest < size; ++dest) {
        world_.send(dest, 0, output_);
      }
    }
  } else {
    boost::mpi::gatherv(world_, local_data.data(), sendcounts[rank], 0);
    world_.recv(0, 0, output_);
  }

  unsigned int output_size = task_data->outputs_count[0];
  if (output_.size() != output_size) {
    output_.resize(output_size, 0);
  }

  return true;
}

void TestTaskMPI::MergeSortedBlocks(const std::vector<int>& gathered, const std::vector<int>& sendcounts,
                                    const std::vector<int>& displs, int size,
                                    boost::mpi::communicator& world) {  // NOLINT
  if (sendcounts.size() != static_cast<size_t>(size) || displs.size() != static_cast<size_t>(size)) {
    std::cerr << "Error: Invalid sendcounts or displs size in MergeSortedBlocks\n";
    output_.clear();
    return;
  }

  if (gathered.empty() || size <= 0) {
    output_.clear();
    return;
  }

  std::vector<std::vector<int>> blocks(size);
  int pos = 0;
  bool valid = true;
  for (int i = 0; i < size && valid; ++i) {
    if (sendcounts[i] > 0 && pos + sendcounts[i] <= static_cast<int>(gathered.size())) {
      blocks[i] = std::vector<int>(gathered.begin() + pos, gathered.begin() + pos + sendcounts[i]);
      pos += sendcounts[i];
    } else if (sendcounts[i] < 0 || pos + sendcounts[i] > static_cast<int>(gathered.size())) {
      std::cerr << "Error: invalid range for blocks[" << i << "]: pos=" << pos << ", sendcounts[" << i
                << "]=" << sendcounts[i] << ", gathered.size()=" << gathered.size() << "\n";
      valid = false;
    } else {
      blocks[i] = std::vector<int>();
      pos += sendcounts[i];
    }
  }

  if (!valid) {
    output_.clear();
    return;
  }

  std::vector<int> merged;
  for (int i = 0; i < size; ++i) {
    if (!blocks[i].empty()) {
      if (merged.empty()) {
        merged = blocks[i];
      } else {
        std::vector<int> temp(merged.size() + blocks[i].size());
        BatcherMerge(merged, blocks[i], temp);
        merged = std::move(temp);
      }
    }
  }
  output_ = std::move(merged);

  // Рассылка результата
  for (int dest = 1; dest < size; ++dest) {
    world.send(dest, 0, output_);
  }
}

bool TestTaskMPI::PostProcessingImpl() {
  unsigned int output_size = task_data->outputs_count[0];
  if (output_.size() == output_size) {
    for (size_t i = 0; i < output_.size(); ++i) {
      reinterpret_cast<int*>(task_data->outputs[0])[i] = output_[i];
    }
  } else {
    for (size_t i = 0; i < output_size; ++i) {
      reinterpret_cast<int*>(task_data->outputs[0])[i] = 0;
    }
  }
  return true;
}

void TestTaskMPI::ShellSort(std::vector<int>& arr) {
  if (arr.empty()) {
    return;
  }
  int n = static_cast<int>(arr.size());
  std::vector<int> gaps;
  for (int k = 1; (1 << k) - 1 < n; ++k) {
    gaps.push_back((1 << k) - 1);
  }
  for (auto it = gaps.rbegin(); it != gaps.rend(); ++it) {
    int gap = *it;
#pragma omp parallel for default(none) shared(arr, n, gap)
    for (int offset = 0; offset < gap; ++offset) {
      for (int i = offset + gap; i < n; i += gap) {
        int temp = arr[i];
        int j = i;
        while (j >= gap && arr[j - gap] > temp) {
          arr[j] = arr[j - gap];
          j -= gap;
        }
        arr[j] = temp;
      }
    }
  }
}
void TestTaskMPI::BatcherMerge(std::vector<int>& left, std::vector<int>& right, std::vector<int>& result) {
  size_t i = 0;
  size_t j = 0;
  size_t k = 0;
  while (i < left.size() && j < right.size()) {
    if (left[i] <= right[j]) {
      result[k++] = left[i++];
    } else {
      result[k++] = right[j++];
    }
  }
  while (i < left.size()) {
    result[k++] = left[i++];
  }
  while (j < right.size()) {
    result[k++] = right[j++];
  }
}

}  // namespace fyodorov_m_shell_sort_with_even_odd_batcher_merge_mpi