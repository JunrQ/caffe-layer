// ------------------------------------------------------------------
// <Wing Loss for Robust Facial Landmark Localisation with 
//  Convolutoinal Neural Networks> CVPR 2018
// Written by ZhouJunr
// ------------------------------------------------------------------

#include "caffe/layers/wing_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
void WingLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  has_weights_ = (bottom.size() == 3);
  omega_ = this->layer_param_.wing_loss_param().omega();
  epsilon_ = this->layer_param_.wing_loss_param().epsilon();
  C_ = omega_ * (1 - log(1 + omega_ / epsilon_));
}

template <typename Dtype>
void WingLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  if (has_weights_) {
    CHECK_EQ(bottom[0]->channels(), bottom[2]->channels());
    CHECK_EQ(bottom[0]->height(), bottom[2]->height());
    CHECK_EQ(bottom[0]->width(), bottom[2]->width());
  }
  diff_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  errors_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void WingLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void WingLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(WingLossLayer);
#endif

INSTANTIATE_CLASS(WingLossLayer);
REGISTER_LAYER_CLASS(WingLoss);

}  // namespace caffe
