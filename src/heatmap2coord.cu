#include <vector>
#include <limits>

#include "caffe/layers/heatmap2coord.hpp"

namespace caffe {

template <typename Dtype>
__global__ void H2C_kernel(const int n, const Dtype* input,
    Dtype* out, const int num_points, const int h, const int w,
    Dtype max_val) {
  CUDA_KERNEL_LOOP(index, n) {
    int bi = index / num_points;
    int p_idx = index % num_points;
    int argmax_h = -1;
    int argmax_w = -1;
    Dtype tmp_val = max_val;
    for (int hi = 0; hi < h; ++hi) {
      for (int wi = 0; wi < w; ++wi) {
        tmp_val = input[((bi * num_points +  p_idx) * h + hi) * w + wi];
        if ( tmp_val > max_val ) {
          max_val = tmp_val;
          argmax_h = hi;
          argmax_w = wi;
        }
      }
    }
    // assign x and y
    out[bi * 2 * num_points + p_idx * 2] = argmax_w;
    out[bi * 2 * num_points + p_idx * 2 + 1] = argmax_h;
  }
}

template <typename Dtype>
void Heatmap2coordLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int h = bottom[0]->shape(2);
  const int w = bottom[0]->shape(3);
  Dtype max_val = std::numberic_limits<Dtype>::min();
  const int n = bottom[0]->count(0, 2);
  H2C_kernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
    n, bottom[0]->gpu_data(), top[0]->mutable_gpu_data(), num_points_,
    h, w, max_val);  
}

INSTANTIATE_LAYER_GPU_FUNCS(Heatmap2CoordLayer);

}