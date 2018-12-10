#include <vector>
#include <limits>

#include "caffe/layers/heatmap2coord.hpp"

namespace caffe {

template <typename Dtype>
void Heatmap2coordLayer<Dtype>::LayerSetup(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
  num_points_ = bottom[0]->shape(1);
}

template <typename Dtype>
void Heatmap2coordLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    top[0]->Reshape(bottom[0]->shape(0), bottom[0]->shape(1) * 2);
}

template <typename Dtype>
void Heatmap2coordLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int h = bottom[0]->shape(2);
  const int w = bottom[0]->shape(3);
  int argmax_h = -1;
  int argmax_w = -1;
  Dtype max_val = std::numberic_limits<Dtype>::min();
  Dtype tmp_val;
  // int batch_size = bottom[0]->shape(0);

  for (int bi = 0; bi < batch_size; ++bi) {
    for (int p_idx = 0; p_idx < num_points_; ++p_idx) {
      for (int hi = 0; hi < h; ++hi) {
        for (int wi = 0; wi < w; ++wi) {
          tmp_val = bottom_data[bottom_data[0]->offset(bi, p_idx, hi, wi)];
          if ( tmp_val > max_val) {
            max_val = tmp_val;
            argmax_h = hi;
            argmax_w = wi;
          }
        }
      }
      // assign x and y
      top_data[bi * 2 * num_points_ + p_idx * 2] = argmax_w;
      top_data[bi * 2 * num_points_ + p_idx * 2 + 1] = argmax_h;
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(Heatmap2CoordLayer, Forward);
#endif

INSTANTIATE_CLASS(Heatmap2CoordLayer);
REGISTER_LAYER_CLASS(Heatmap2Coord);

}