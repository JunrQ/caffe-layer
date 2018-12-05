#include <stdlib.h>

#include "caffe/layers/heatmap_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

// TODO(ZhouJ) Actually we do not need to calculate loss
// in forward pass.

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
    out[index] = -out[index];
  }
}

template <typename Dtype>
__global__ void CE_mask(const int n, const Dtype* gt, const Dtype* pred, 
    Dtype* out, const Dtype* mask, Dtype negative_ratio,
    int w, int h, int c, Dtype eps) {
  CUDA_KERNEL_LOOP(index, n) {
    int chn = index / w / h % c;
    int batch_idx = index / w / h / c;
    Dtype m = mask[batch_idx * c * 2 + 2 * chn];
    if (m == Dtype(0)) {
      out[index] = Dtype(0);
    } else {
      Dtype gt_ = gt[index];
      Dtype pred_ = pred[index];
      out[index] = (gt_ * log(pred_ + eps) + (1 - gt_) * log(1 - pred_ + eps));
      if (gt_ == Dtype(0)) {
        out[index] *= negative_ratio;
      }
      out[index] = -out[index] * m;
    }
  }
}

template <typename Dtype>
__global__ void SE(const int n, const Dtype* gt, const Dtype* pred, 
    Dtype* out, Dtype negative_ratio, Dtype eps) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype gt_ = gt[index];
    Dtype pred_ = pred[index];
    out[index] = (gt_ - pred_) * (gt_ - pred_);
    if (gt_ == Dtype(0)) {
      out[index] *= negative_ratio;
    }
  }
}

template <typename Dtype>
__global__ void SE_mask(const int n, const Dtype* gt, const Dtype* pred, 
    Dtype* out, const Dtype* mask, Dtype negative_ratio,
    int w, int h, int c, Dtype eps) {
  CUDA_KERNEL_LOOP(index, n) {
    int chn = index / w / h % c;
    int batch_idx = index / w / h / c;
    Dtype m = mask[batch_idx * c * 2 + 2 * chn];
    if (m == Dtype(0)) {
      out[index] = Dtype(0);
    } else {
      Dtype gt_ = gt[index];
      Dtype pred_ = pred[index];
      out[index] = (gt_ - pred_) * (gt_ - pred_);
      if (gt_ == Dtype(0)) {
        out[index] *= negative_ratio;
      }
      out[index] *= m;
    }
  }
}

template <typename Dtype>
__global__ void bp_CE(const int n, const Dtype* gt, const Dtype* pred,
    Dtype* out, Dtype negative_ratio, Dtype eps, Dtype grad_clip) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype gt_ = gt[index];
    Dtype pred_ = pred[index];
    out[index] = (1 - gt_) / (1 - pred_ + eps) - gt_ / (eps + pred_);
    if (gt_ == Dtype(0)) {
      out[index] *= negative_ratio;
    }
    if (grad_clip > 0 && abs(out[index]) > grad_clip)
      out[index] = out[index] > 0 ? grad_clip : -grad_clip;
  }
}

template <typename Dtype>
__global__ void bp_SE(const int n, const Dtype* gt, const Dtype* pred,
    Dtype* out, Dtype negative_ratio, Dtype eps, Dtype grad_clip) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype gt_ = gt[index];
    Dtype pred_ = pred[index];
    out[index] = 2 * (pred_ - gt_);
    if (gt_ == Dtype(0)) {
      out[index] *= negative_ratio;
    }
    if (grad_clip > 0 && abs(out[index]) > grad_clip)
      out[index] = out[index] > 0 ? grad_clip : -grad_clip;
  }
}

template <typename Dtype>
__global__ void bp_CE_mask(const int n, const Dtype* gt, const Dtype* pred,
    Dtype* out, const Dtype* mask, Dtype negative_ratio,
    int w, int h, int c, Dtype eps, Dtype grad_clip) {
  CUDA_KERNEL_LOOP(index, n) {
    int chn = index / w / h % c;
    int batch_idx = index / w / h / c;
    Dtype m = mask[batch_idx * c * 2 + 2 * chn];
    if (Dtype(0) == m) {
      out[index] = Dtype(0);
    } else {
      Dtype gt_ = gt[index];
      Dtype pred_ = pred[index];
      out[index] = (1 - gt_) / (1 - pred_ + eps) - gt_ / (pred_ + eps);
      if (gt_ == Dtype(0)) {
        out[index] *= negative_ratio;
      }
      out[index] *= m;
    }
    if (grad_clip > 0 && abs(out[index]) > grad_clip)
      out[index] = out[index] > 0 ? grad_clip : -grad_clip;
  }
}

template <typename Dtype>
__global__ void bp_SE_mask(const int n, const Dtype* gt, const Dtype* pred,
    Dtype* out, const Dtype* mask, Dtype negative_ratio,
    int w, int h, int c, Dtype eps, Dtype grad_clip) {
  CUDA_KERNEL_LOOP(index, n) {
    int chn = index / w / h % c;
    int batch_idx = index / w / h / c;
    Dtype m = mask[batch_idx * c * 2 + 2 * chn];
    if (Dtype(0) == m) {
      out[index] = Dtype(0);
    } else {
      Dtype gt_ = gt[index];
      Dtype pred_ = pred[index];
      out[index] = 2 * (pred_ - gt_);
      if (gt_ == Dtype(0)) {
        out[index] *= negative_ratio;
      }
      out[index] *= m;
    }
    if (grad_clip > 0 && abs(out[index]) > grad_clip)
      out[index] = out[index] > 0 ? grad_clip : -grad_clip;
  }
}

template <typename Dtype>
__global__ void bp_CE_ns(const int n, const Dtype* gt, const Dtype* pred,
    Dtype* out, Dtype negative_ratio, Dtype eps, Dtype grad_clip,
    const float* r_mask, float threshold) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype gt_ = gt[index];
    Dtype pred_ = pred[index];
    out[index] = (1 - gt_) / (1 - pred_ + eps) - gt_ / (eps + pred_);
    if (gt_ == Dtype(0)) {
      if (r_mask[index] > threshold) {
        out[index] = Dtype(0);
      } else {
        out[index] *= negative_ratio;
      }
    }
    if (grad_clip > 0 && abs(out[index]) > grad_clip)
      out[index] = out[index] > 0 ? grad_clip : -grad_clip;
  }
}

template <typename Dtype>
__global__ void bp_SE_ns(const int n, const Dtype* gt, const Dtype* pred,
    Dtype* out, Dtype negative_ratio, Dtype eps, Dtype grad_clip,
    const float* r_mask, float threshold) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype gt_ = gt[index];
    Dtype pred_ = pred[index];
    out[index] = 2 * (pred_ - gt_);
    if (gt_ == Dtype(0)) {
      if (r_mask[index] > threshold) {
        out[index] = Dtype(0);
      } else {
        out[index] *= negative_ratio;
      }
    }
    if (grad_clip > 0 && abs(out[index]) > grad_clip)
      out[index] = out[index] > 0 ? grad_clip : -grad_clip;
  }
}

template <typename Dtype>
__global__ void bp_CE_mask_ns(const int n, const Dtype* gt, const Dtype* pred,
    Dtype* out, const Dtype* mask, Dtype negative_ratio,
    int w, int h, int c, Dtype eps, Dtype grad_clip,
    const float* r_mask, float threshold) {
  CUDA_KERNEL_LOOP(index, n) {
    int chn = index / w / h % c;
    int batch_idx = index / w / h / c;
    Dtype m = mask[batch_idx * c * 2 + 2 * chn];
    if (Dtype(0) == m) {
      out[index] = Dtype(0);
    } else {
      Dtype gt_ = gt[index];
      Dtype pred_ = pred[index];
      out[index] = (1 - gt_) / (1 - pred_ + eps) - gt_ / (pred_ + eps);
      if (gt_ == Dtype(0)) {
        if (r_mask[index] > threshold) {
          out[index] = Dtype(0);
        } else {
          out[index] *= negative_ratio;
        }
      }
      out[index] *= m;
    }
    if (grad_clip > 0 && abs(out[index]) > grad_clip)
      out[index] = out[index] > 0 ? grad_clip : -grad_clip;
  }
}

template <typename Dtype>
__global__ void bp_SE_mask_ns(const int n, const Dtype* gt, const Dtype* pred,
    Dtype* out, const Dtype* mask, Dtype negative_ratio,
    int w, int h, int c, Dtype eps, Dtype grad_clip,
    const float* r_mask, float threshold) {
  CUDA_KERNEL_LOOP(index, n) {
    int chn = index / w / h % c;
    int batch_idx = index / w / h / c;
    Dtype m = mask[batch_idx * c * 2 + 2 * chn];
    if (Dtype(0) == m) {
      out[index] = Dtype(0);
    } else {
      Dtype gt_ = gt[index];
      Dtype pred_ = pred[index];
      out[index] = 2 * (pred_ - gt_);
      if (gt_ == Dtype(0)) {
        if (r_mask[index] > threshold) {
          out[index] = Dtype(0);
        } else {
          out[index] *= negative_ratio;
        }
      }
      out[index] *= m;
    }
    if (grad_clip > 0 && abs(out[index]) > grad_clip)
      out[index] = out[index] > 0 ? grad_clip : -grad_clip;
  }
}

template <typename Dtype>
void HeatmapLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    int count = diff_.count();
    Dtype loss;
    switch (this->layer_param_.heatmap_loss_param().loss_type()) {
    case HeatmapLossParameter_LossType_CE:
      if (has_weights_) {
        CE_mask<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, bottom[0]->gpu_data(), bottom[1]->gpu_data(), 
          diff_.mutable_gpu_data(),
          bottom[2]->gpu_data(),
          negative_ratio_, bottom[0]->shape(3),
          bottom[0]->shape(2), bottom[0]->shape(1), eps_);
        CUDA_POST_KERNEL_CHECK;
      } else {
        CE<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, bottom[0]->gpu_data(), bottom[1]->gpu_data(), 
          diff_.mutable_gpu_data(), negative_ratio_, eps_);
        CUDA_POST_KERNEL_CHECK;
      }

      caffe_gpu_asum(count, diff_.gpu_data(), &loss);
      top[0]->mutable_cpu_data()[0] = loss / (bottom[0]->count(2, 4));
      break;

    case HeatmapLossParameter_LossType_SE:
      if (has_weights_) {
        SE_mask<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, bottom[0]->gpu_data(), bottom[1]->gpu_data(), 
          diff_.mutable_gpu_data(),
          bottom[2]->gpu_data(),
          negative_ratio_, bottom[0]->shape(3),
          bottom[0]->shape(2), bottom[0]->shape(1), eps_);
        CUDA_POST_KERNEL_CHECK;
      } else {
        SE<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, bottom[0]->gpu_data(), bottom[1]->gpu_data(), 
          diff_.mutable_gpu_data(), negative_ratio_, eps_);
        CUDA_POST_KERNEL_CHECK;
      }

      caffe_gpu_asum(count, diff_.gpu_data(), &loss);
      top[0]->mutable_cpu_data()[0] = loss / (bottom[0]->count(2, 4));
      break;
    }
  }

template <typename Dtype>
void HeatmapLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    int count = bottom[0]->count();
    if (negative_sample_prob_ < 0.999) {
      caffe_rng_uniform<float>(count, float(0.), float(1.), rand_mask_.mutable_cpu_data());
    }
    switch (this->layer_param_.heatmap_loss_param().loss_type()) {
      case HeatmapLossParameter_LossType_CE:
        if (negative_sample_prob_ > 0.999) {
          if (has_weights_) {
            bp_CE_mask<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
              count, bottom[0]->gpu_data(), bottom[1]->gpu_data(), 
              bottom[1]->mutable_gpu_diff(), bottom[2]->gpu_data(),
              negative_ratio_, bottom[0]->shape(3),
              bottom[0]->shape(2), bottom[0]->shape(1), eps_, grad_clip_);
            CUDA_POST_KERNEL_CHECK;
          } else {
            bp_CE<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
              count, bottom[0]->gpu_data(), bottom[1]->gpu_data(), 
              bottom[1]->mutable_gpu_diff(),
              negative_ratio_, eps_, grad_clip_);
            CUDA_POST_KERNEL_CHECK;
          }
        } else {
          if (has_weights_) {
            bp_CE_mask_ns<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
              count, bottom[0]->gpu_data(), bottom[1]->gpu_data(), 
              bottom[1]->mutable_gpu_diff(), bottom[2]->gpu_data(),
              negative_ratio_, bottom[0]->shape(3),
              bottom[0]->shape(2), bottom[0]->shape(1), eps_, grad_clip_,
              rand_mask_.gpu_data(), negative_sample_prob_);
            CUDA_POST_KERNEL_CHECK;
          } else {
            bp_CE_ns<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
              count, bottom[0]->gpu_data(), bottom[1]->gpu_data(), 
              bottom[1]->mutable_gpu_diff(),
              negative_ratio_, eps_, grad_clip_,
              rand_mask_.gpu_data(), negative_sample_prob_);
            CUDA_POST_KERNEL_CHECK;
          }
        }
      break;

      case HeatmapLossParameter_LossType_SE:
        if (negative_sample_prob_ > 0.999) {
          if (has_weights_) {
            bp_SE_mask<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
              count, bottom[0]->gpu_data(), bottom[1]->gpu_data(), 
              bottom[1]->mutable_gpu_diff(), bottom[2]->gpu_data(),
              negative_ratio_, bottom[0]->shape(3),
              bottom[0]->shape(2), bottom[0]->shape(1), eps_, grad_clip_);
            CUDA_POST_KERNEL_CHECK;
          } else {
            bp_SE<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
              count, bottom[0]->gpu_data(), bottom[1]->gpu_data(), 
              bottom[1]->mutable_gpu_diff(),
              negative_ratio_, eps_, grad_clip_);
            CUDA_POST_KERNEL_CHECK;
          }
        } else {
          if (has_weights_) {
            bp_SE_mask_ns<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
              count, bottom[0]->gpu_data(), bottom[1]->gpu_data(), 
              bottom[1]->mutable_gpu_diff(), bottom[2]->gpu_data(),
              negative_ratio_, bottom[0]->shape(3),
              bottom[0]->shape(2), bottom[0]->shape(1), eps_, grad_clip_,
              rand_mask_.gpu_data(), negative_sample_prob_);
            CUDA_POST_KERNEL_CHECK;
          } else {
            bp_SE_ns<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
              count, bottom[0]->gpu_data(), bottom[1]->gpu_data(), 
              bottom[1]->mutable_gpu_diff(),
              negative_ratio_, eps_, grad_clip_,
              rand_mask_.gpu_data(), negative_sample_prob_);
            CUDA_POST_KERNEL_CHECK;
          }
        }
      break;
    }
    caffe_gpu_scal(
      bottom[1]->count(),
      Dtype(1) / bottom[1]->count(2, 4),
      bottom[1]->mutable_gpu_diff());
  }

INSTANTIATE_LAYER_GPU_FUNCS(HeatmapLossLayer);

}