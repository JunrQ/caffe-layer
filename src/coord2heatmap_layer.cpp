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
  // LOG(INFO) << bottom[0]->shape_string();
  int bottom_points = bottom[0]->shape(1) / 2;
  CHECK_LE(num_points_, bottom_points) << "Coord2heatmapLayer num_points must "
      "be less or equal to number of inputs points.";
  max_value_ = this->layer_param_.coord2heatmap_param().max_value();
  CHECK_GE(max_value_, 1) << "Coord2heatmapLayer max_value must be greater "
      "or equal to 1";
  radius_ = this->layer_param_.coord2heatmap_param().radius();
  if (radius_ != 1)
    CHECK_EQ(radius_, 5) << "Only support radius 5, you can set radius to 1"
        "to not use gaussian blur.";
}

template <typename Dtype>
void Coord2heatmapLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(bottom[0]->shape(0), num_points_, output_height_, output_width_);
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
      int tmp = 2 * c * (b + 1);
      int x = (int)bottom_data[tmp];
      int y = (int)bottom_data[tmp + 1];
      x = x > (output_width_ - 1) ? output_width_ -1 : x;
      y = y > (output_height_ - 1) ? output_height_ -1 : y;
      if (x > 0 && y > 0) {
        top_data[top[0]->offset(b, c, y, x)] = Dtype(max_value_);
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(Coord2heatmapLayer, Forward);
#endif

INSTANTIATE_CLASS(Coord2heatmapLayer);
REGISTER_LAYER_CLASS(Coord2heatmap);

} // namespace caffe