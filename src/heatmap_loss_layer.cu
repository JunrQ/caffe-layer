#include "caffe/layers/heatmap_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void CE(const int n, const Dtype* gt, const Dtype* pred, 
    Dtype* out, Dtype negative_ratio, Dtype eps) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype gt_ = gt[index];
    Dtype pred_ = pred[index];
    out[index] = gt_ * log(pred_ + eps) + (1 - gt_) * log(1 - pred_ + eps);
    if (gt_ == Dtype(0)) {
      out[index] *= negative_ratio;
    }
    out[index] = -out[index]
  }
}

template <typename Dtype>
__global__ void CE_mask(const int n, const Dtype* gt, const Dtype* pred, 
    Dtype* out, Dtype* mask, Dtype negative_ratio,
    int w, int h, int c, Dtype eps) {
  CUDA_KERNEL_LOOP(index, n) {
    int chn = index / w / h % c;
    Dtype gt_ = gt[index];
    Dtype pred_ = pred[index];
    out[index] = negative_ratio * (gt_ * log(pred_ + eps) + (1 - gt_) * log(1 - pred_ + eps));
    if (gt_ == Dtype(0)) {
      out[index] *= negative_ratio;
    }
    out[index] = -out[index] * mask[chn];
  }
}

template <typename Dtype>
__global__ void bp_CE(const int n, const Dtype* gt, const Dtype* pred,
    Dtype* out, Dtype negative_ratio, Dtype eps) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype gt_ = gt[index];
    Dtype pred_ = pred[index];
    out[index] = (1 - gt_) / (1 - pred_ + eps) - gt_ / (eps + pred_);
    if (gt_ == Dtype(0)) {
      out[index] *= negative_ratio;
    }
  }
}

template <typename Dtype>
__global__ void bp_CE_mask(const int n, const Dtype* gt, const Dtype* pred,
    Dtype* out, Dtype* mask, Dtype negative_ratio,
    int w, int h, int c, Dtype eps) {
  CUDA_KERNEL_LOOP(index, n) {
    int chn = index / w / h % c;
    Dtype gt_ = gt[index];
    Dtype pred_ = pred[index];
    out[index] = (1 - gt_) / (1 - pred_ + eps) - gt_ / (pred_ + eps);
    if (gt_ == Dtype(0)) {
      out[index] *= negative_ratio;
    }
    out[index] *= mask[chn];
  }
}


template <typename Dtype>
void HeatmapLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    switch (this->layer_param_.heatmap_loss_params().loss_type()) {
    case HeatmapLossParameter_LossType_CE:
      int count = bottom[0]->count();
      if (has_weights_) {
        CE_mask<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, bottom[0]->gpu_data(), bottom[1]->gpu_data(), 
          errors_.mutable_gpu_data(), bottom[2]->gpu_data(),
          negative_ratio_, bottom[0]->shape(3),
          bottom[0]->shape(2), bottom[0]->shape(1), eps_);
        CUDA_POST_KERNEL_CHECK;
      } else {
        CE<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, bottom[0]->gpu_data(), bottom[1]->gpu_data(), 
          errors_.mutable_gpu_data(), negative_ratio_, eps_);
        CUDA_POST_KERNEL_CHECK;
      }

      Dtype loss;
      caffe_gpu_asum(count, errors_.gpu_data(), &loss);
      top[0]->mutable_gpu_data()[0] = loss;
      break;
    }
  }

template <typename Dtype>
void HeatmapLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    switch (this->layer_param_.heatmap_loss_params().loss_type()) {
      case HeatmapLossParameter_LossType_CE:
        if (has_weights_) {
          bp_CE_mask<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, top[0]->gpu_data(), top[1]->gpu_data(), 
            bottom[1]->mutable_gpu_data(), top[2]->gpu_data(),
            negative_ratio_, top[0]->shape(3),
            top[0]->shape(2), top[0]->shape(1), eps_);
          CUDA_POST_KERNEL_CHECK;
        } else {
          bp_CE<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, top[0]->gpu_data(), top[1]->gpu_data(), 
            bottom[1]->mutable_gpu_data(), top[2]->gpu_data(),
            negative_ratio_, eps_);
          CUDA_POST_KERNEL_CHECK;
        }
      break;
    }
  }

INSTANTIATE_LAYER_GPU_FUNCS(HeatmapLossLayer);

}