#include "all/malyshev_v_radix_sort/include/ops_all.hpp"

#include <oneapi/tbb/combinable.h>
#include <oneapi/tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/global_control.h>
#include <tbb/partitioner.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/gatherv.hpp>
#include <boost/mpi/collectives/scatterv.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>  // NOLINT(*-include-cleaner)
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <ranges>
#include <vector>

namespace mpi = boost::mpi;

void malyshev_v_radix_sort_all::ConvertDoublesToUint64(const std::vector<double>& data,
                                                       std::vector<uint64_t>& converted, size_t first) {
  tbb::parallel_for(
      tbb::blocked_range<size_t>(0, converted.size()),
      [&](const auto& range) {
        for (size_t i = range.begin(); i != range.end(); ++i) {
          uint64_t bits = 0;
          memcpy(&bits, &data[first + i], sizeof(double));
          converted[i] = ((bits & (1ULL << 63)) != 0) ? ~bits : bits ^ (1ULL << 63);
        }
      },
      tbb::auto_partitioner());
}

void malyshev_v_radix_sort_all::ConvertUint64ToDoubles(std::vector<double>& data,
                                                       const std::vector<uint64_t>& converted, size_t first) {
  tbb::parallel_for(
      tbb::blocked_range<size_t>(0, converted.size()),
      [&](const auto& range) {
        for (size_t i = range.begin(); i != range.end(); ++i) {
          uint64_t bits = converted[i];
          bits = ((bits & (1ULL << 63)) != 0) ? (bits ^ (1ULL << 63)) : ~bits;
          memcpy(&data[first + i], &bits, sizeof(double));
        }
      },
      tbb::auto_partitioner());
}

void malyshev_v_radix_sort_all::RadixDoubleSort(std::vector<double>& data, size_t first, size_t last) {
  const size_t sort_size = last - first;
  std::vector<uint64_t> converted(sort_size);
  ConvertDoublesToUint64(data, converted, first);

  std::vector<uint64_t> buffer(sort_size);
  constexpr int kBitsInByte = 8;
  constexpr int kMaxByte = 255;
  constexpr size_t kTotalBits = sizeof(uint64_t) * CHAR_BIT;

  for (size_t shift = 0; shift < kTotalBits; shift += kBitsInByte) {
    tbb::combinable<std::array<size_t, 256>> local_counts;
    tbb::parallel_for(tbb::blocked_range<size_t>(0, sort_size), [&](const auto& range) {
      auto& counts = local_counts.local();
      for (size_t i = range.begin(); i != range.end(); ++i) {
        ++counts[(converted[i] >> shift) & kMaxByte];
      }
    });

    std::array<size_t, 256> total_counts{};
    local_counts.combine_each([&](const auto& cnt) {
      for (size_t j = 0; j < 256; ++j) {
        total_counts[j] += cnt[j];
      }
    });

    size_t sum = 0;
    for (size_t& count : total_counts) {
      size_t current = count;
      count = sum;
      sum += current;
    }

    std::array<std::atomic<size_t>, 256> next_index;
    for (size_t i = 0; i < 256; ++i) {
      next_index[i].store(total_counts[i]);
    }

    tbb::parallel_for(tbb::blocked_range<size_t>(0, sort_size), [&](const auto& range) {
      for (size_t i = range.begin(); i != range.end(); ++i) {
        const uint8_t digit = (converted[i] >> shift) & kMaxByte;
        const size_t pos = next_index[digit].fetch_add(1);
        buffer[pos] = converted[i];
      }
    });

    converted.swap(buffer);
  }

  ConvertUint64ToDoubles(data, converted, first);
}

void malyshev_v_radix_sort_all::SimpleMerge(std::vector<double>& arr, size_t l, size_t m, size_t r) {
  if (m >= r) {
    return;
  }

  std::vector<double> temp(r - l);
  size_t i = l;
  size_t j = m;
  size_t k = 0;

  while (i < m && j < r) {
    if (arr[i] <= arr[j]) {
      temp[k++] = arr[i++];
    } else {
      temp[k++] = arr[j++];
    }
  }

  while (i < m) {
    temp[k++] = arr[i++];
  }
  while (j < r) {
    temp[k++] = arr[j++];
  }

  std::ranges::copy(temp, arr.begin() + static_cast<std::ptrdiff_t>(l));
}

bool malyshev_v_radix_sort_all::TestTaskALL::PreProcessingImpl() {
  if (world_.rank() == 0) {
    if (task_data->inputs[0] == nullptr || task_data->inputs_count[0] == 0) {
      return false;
    }
    input_data_.resize(task_data->inputs_count[0]);
    std::ranges::copy(std::views::counted(reinterpret_cast<double*>(task_data->inputs[0]), task_data->inputs_count[0]),
                      input_data_.begin());
  }
  return true;
}

bool malyshev_v_radix_sort_all::TestTaskALL::RunImpl() {
  if (world_.rank() == 0) {
    n_ = input_data_.size();
    const size_t base_chunk = n_ / world_.size();
    const size_t remainder = n_ % world_.size();

    counts_.resize(world_.size());
    displs_.resize(world_.size());

    for (size_t i = 0; i < static_cast<size_t>(world_.size()); ++i) {
      counts_[i] = static_cast<int>(base_chunk + (i < remainder ? 1 : 0));
      displs_[i] = (i == 0) ? 0 : displs_[i - 1] + counts_[i - 1];
    }
  }

  mpi::broadcast(world_, n_, 0);
  mpi::broadcast(world_, counts_, 0);
  mpi::broadcast(world_, displs_, 0);

  int local_size = counts_[world_.rank()];
  std::vector<double> local_data(local_size);
  mpi::scatterv(world_, world_.rank() == 0 ? input_data_.data() : nullptr, counts_, displs_, local_data.data(),
                local_size, 0);

  RadixDoubleSort(local_data, 0, local_data.size());

  std::vector<double> global_data;
  if (world_.rank() == 0) {
    global_data.resize(n_);
  }
  mpi::gatherv(world_, local_data.data(), local_size, global_data.data(), counts_, displs_, 0);

  if (world_.rank() == 0) {
    input_data_ = global_data;
    size_t block_size = 1;
    while (block_size < n_) {
      for (size_t i = 0; i < n_; i += 2 * block_size) {
        const size_t left = i;
        const size_t mid = std::min(i + block_size, n_);
        const size_t right = std::min(i + (2 * block_size), n_);
        SimpleMerge(input_data_, left, mid, right);
      }
      block_size *= 2;
    }
    all_data_ = input_data_;
  }
  return true;
}

bool malyshev_v_radix_sort_all::TestTaskALL::ValidationImpl() {
  if (world_.rank() == 0) {
    if (task_data->inputs.empty() || task_data->inputs[0] == nullptr || task_data->inputs_count.empty() ||
        task_data->inputs_count[0] == 0) {
      return false;
    }
    if (task_data->outputs.empty() || task_data->outputs[0] == nullptr || task_data->outputs_count.empty() ||
        task_data->outputs_count[0] != task_data->inputs_count[0]) {
      return false;
    }
  }
  return true;
}

bool malyshev_v_radix_sort_all::TestTaskALL::PostProcessingImpl() {
  if (world_.rank() == 0) {
    std::ranges::copy(all_data_, reinterpret_cast<double*>(task_data->outputs[0]));
  }
  return true;
}