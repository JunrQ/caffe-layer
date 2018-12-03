#include <vector>
#include <math.h>

#include "caffe/layers/coord2heatmap_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void Coord2heatmapForward(const int n,
    const int batch_size,
    const int num_points,
    const int height,
    const int width,
    const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    int c = index % num_points;
    int b = index / num_points;
    int tmp = 2 * c * (b + 1);
    int x = int(in[tmp]);
    int y = int(in[tmp + 1]);
    x = x > (width - 1) ? width -1 : x;
    y = y > (height - 1) ? height -1 : y;
    if (x > 0 && y > 0) {
      out[((b * num_points + c) * height + y) * width + x] = Dtype(1);
    }
  }
}

template <typename Dtype>
void Coord2heatmapLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int batch_size =  bottom[0]->shape(0);
  const int count = batch_size * num_points_;
  Coord2heatmapForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, batch_size, num_points_,
      output_height_, output_width_,
      bottom_data, top_data);
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FORWARD(Coord2heatmapLayer);

}