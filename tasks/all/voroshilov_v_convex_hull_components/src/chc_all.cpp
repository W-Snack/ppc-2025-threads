#include "../include/chc_all.hpp"

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <vector>

#include "../include/chc.hpp"

using namespace voroshilov_v_convex_hull_components_all;

bool voroshilov_v_convex_hull_components_all::ChcTaskALL::ValidationImpl() {
  if (world_.rank() == 0) {
    int *ptr = reinterpret_cast<int *>(task_data->inputs[0]);
    int height = *ptr;
    ptr = reinterpret_cast<int *>(task_data->inputs[1]);
    int width = *ptr;
    int pixels_size = static_cast<int>(task_data->inputs_count[0]);
    return height > 0 && width > 0 && (height * width) == pixels_size;
  }
  return true;
}

bool voroshilov_v_convex_hull_components_all::ChcTaskALL::PreProcessingImpl() {
  if (world_.rank() == 0) {
    int *ptr = reinterpret_cast<int *>(task_data->inputs[0]);
    height_in_ = *ptr;

    ptr = reinterpret_cast<int *>(task_data->inputs[1]);
    width_in_ = *ptr;

    pixels_in_.resize(task_data->inputs_count[0]);
    ptr = reinterpret_cast<int *>(task_data->inputs[2]);
    std::ranges::copy(ptr, ptr + task_data->inputs_count[0], pixels_in_.begin());
  }
  return true;
}

bool voroshilov_v_convex_hull_components_all::ChcTaskALL::RunImpl() {
  std::vector<Component> components;

  if (world_.rank() == 0) {
    Image image(height_in_, width_in_, pixels_in_);
    components = FindComponentsOMP(image);
  }

  if (world_.size() <= 1) {
    hulls_out_ = QuickHullAllOMP(components);
  } else {
    hulls_out_ = QuickHullAllMPIOMP(components, width_in_);
  }

  return true;
}

bool voroshilov_v_convex_hull_components_all::ChcTaskALL::PostProcessingImpl() {
  if (world_.rank() == 0) {
    int *hulls_indxs = reinterpret_cast<int *>(task_data->outputs[0]);
    int *pixels_indxs = reinterpret_cast<int *>(task_data->outputs[1]);
    PackHulls(hulls_out_, width_in_, height_in_, hulls_indxs, pixels_indxs);
    task_data->outputs_count[0] = hulls_out_.size();
  }
  return true;
}
