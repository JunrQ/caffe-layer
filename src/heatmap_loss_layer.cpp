#include "caffe/layers/heatmap_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
void HeatmapLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    has_weights_ = (bottom.size() == 3);
    negative_ratio_ = (Dtype)this->layer_param_.heatmap_loss_param().negative_ratio();
    eps_ = (Dtype)this->layer_param_.heatmap_loss_param().eps();
    grad_clip_ = (Dtype)this->layer_param_.heatmap_loss_param().grad_clip();
    negative_sample_prob_ = this->layer_param_.heatmap_loss_param().negative_sample_prob();
    CHECK_LE(negative_sample_prob_, 1.);
    CHECK_GE(negative_sample_prob_, 0.);
  }

template <typename Dtype>
void HeatmapLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    LossLayer<Dtype>::Reshape(bottom, top);
    CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
    CHECK_EQ(bottom[0]->height(), bottom[1]->height());
    CHECK_EQ(bottom[0]->width(), bottom[1]->width());
    if (has_weights_) {
      CHECK_EQ(bottom[0]->channels(), bottom[2]->shape(1) / 2);
    }
    diff_.Reshape(bottom[0]->shape());
    // if (negative_sample_prob_ < 0.9999) {
    rand_mask_.Reshape(bottom[0]->shape());
    // }
    vector<int> loss_shape(0);
    top[0]->Reshape(loss_shape);
  }

template <typename Dtype> 
void HeatmapLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void HeatmapLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(HeatmapLossLayer);
#endif

INSTANTIATE_CLASS(HeatmapLossLayer);
REGISTER_LAYER_CLASS(HeatmapLoss);

} // namespace caffe