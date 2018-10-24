// ------------------------------------------------------------------
// <Wing Loss for Robust Facial Landmark Localisation with 
//  Convolutoinal Neural Networks> CVPR 2018
// Written by ZhouJunr
// ------------------------------------------------------------------

#include "caffe/layers/wing_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void WingForward(const int n, const Dtype* in, Dtype* out,
    const float omega, const float epsilon, const float C) {
  // w * ln( 1 + |x| / e), |x| \lt w
  // |x| - C, otherwise
  CUDA_KERNEL_LOOP(index, n) {
    Dtype val = in[index];
    Dtype abs_val = abs(val);
    if (abs_val < omega) {
      out[index] = omega * log(1 + abs_val / epsilon);
    } else {
      out[index] = abs_val - C;
    }
  }
}

template <typename Dtype>
void WingLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),
      bottom[1]->gpu_data(),
      diff_.mutable_gpu_data());    // d := b0 - b1
  if (has_weights_) {
    caffe_gpu_mul(
        count,
        bottom[2]->gpu_data(),
        diff_.gpu_data(),
        diff_.mutable_gpu_data());  // d := w * (b0 - b1)
  }
  WingForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, diff_.gpu_data(), errors_.mutable_gpu_data(),
      omega_, epsilon_, C_);
  CUDA_POST_KERNEL_CHECK;

  Dtype loss;
  caffe_gpu_asum(count, errors_.gpu_data(), &loss);

  if (has_weights_) {
    //normalize the loss
    caffe_gpu_asum(bottom[2]->count(), bottom[2]->gpu_data(), &norm_value_);
  } else {
    norm_value_ = Dtype(1) * bottom[0]->num();
  }
  top[0]->mutable_cpu_data()[0] = loss / norm_value_;
}

template <typename Dtype>
__global__ void WingBackward(const int n, const Dtype* in, Dtype* out,
    const float omega, const float epsilon, const float C) {
  // f'(x) = sign(x) * w / (1 + |x|/e) / e, if |x| < C
  //       = sign(x), otherwise
  CUDA_KERNEL_LOOP(index, n) {
    Dtype val = in[index];
    Dtype abs_val = abs(val);
    Dtype sign = (Dtype(0) < val) - (val < Dtype(0));
    if (abs_val < omega) {
      out[index] = sign * omega / (1 + abs_val / epsilon) / epsilon;
    } else {
      out[index] = sign;
    }
  }
}

template <typename Dtype>
void WingLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  int count = diff_.count();
  WingBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, diff_.gpu_data(), diff_.mutable_gpu_data(),
      omega_, epsilon_, C_);
  if (has_weights_) {
    caffe_gpu_mul(
      count,
      bottom[2]->gpu_data(),
      diff_.gpu_data(),
      diff_.mutable_gpu_data());
  }
  CUDA_POST_KERNEL_CHECK;

  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / norm_value_;
      caffe_gpu_axpby(
        bottom[i]->count(),              // count
        alpha,                           // alpha
        diff_.gpu_data(),                // x
        Dtype(0),                        // beta
        bottom[i]->mutable_gpu_diff());  // y
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(WingLossLayer);

}  // namespace caffe
