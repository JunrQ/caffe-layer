#include <vector>

#include "caffe/layers/coord2heatmap_layer.hpp"

namespace caffe {

template <typename Dtype>
void Coord2heatmapLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
  output_height_ = this->layer_param_.coord2heatmap_param().height();
  CHECK_GT(output_height_, 0) << "Coord2heatmapLayer height must be positive.";
  output_width_ = this->layer_param_.coord2heatmap_param().width();
  CHECK_GT(output_width_, 0) << "Coord2heatmapLayer width must be positive.";
  num_points_ = this->layer_param_.coord2heatmap_param().num_points();
  CHECK_GT(num_points_, 0) << "Coord2heatmapLayer num_points must be positive.";
  int bottom_points = bottom[0]->shape(0) / 2;
  CHECK_LE(num_points_, bottom_points) << "Coord2heatmapLayer num_points must "
      "be less or equal to number of inputs points.";
}

template <typename Dtype>
void Coord2heatmapLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape = bottom[0]->shape();
  top_shape[1] = num_points_;
  top_shape.push_back(output_height_);
  top_shape.push_back(output_width_);
  top[0]->Reshape(top_shape);
  caffe_set(top[0]->count(), Dtype(0), top[0]->mutable_cpu_data());
}

template <typename Dtype>
void Coord2heatmapLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int batch_size =  bottom[0]->shape(0);
  for (int b = 0; b < batch_size; ++b) {
    for (int c = 0; c < num_points_; c++) {
      int x = bottom_data[batch_size * b + 2 * c];
      int y = bottom_data[batch_size * b + 2 * c + 1];
      top_data[top[0]->offset(b, c, y, x)] = Dtype(1);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(Coord2heatmapLayer, Forward);
#endif

INSTANTIATE_CLASS(Coord2heatmapLayer);
REGISTER_LAYER_CLASS(Coord2heatmap);

} // namespace caffe