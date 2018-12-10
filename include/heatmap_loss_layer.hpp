// ------------------------------------------------------------------
// Heatmap Loss layer.
// Written by ZhouJunr
// ------------------------------------------------------------------

#ifndef CAFFE_HEATMAP_LOSS_LAYER_HPP_
#define CAFFE_HEATMAP_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/loss_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Heatmap loss layer.
 * Loss layer for heatmap input.
 */

template <typename Dtype>
class HeatmapLossLayer : public LossLayer<Dtype> {
  public:
    explicit HeatmapLossLayer(const LayerParameter& param)
        : LossLayer<Dtype>(param), diff_() {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    
    virtual inline const char* type() const { return "HeatmapLoss"; }

    virtual inline int ExactNumBottomBlobs() const { return -1; }
    virtual inline int MinBottomBlobs() const { return 2; }
    virtual inline int MaxBottomBlobs() const { return 3; }

  protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);

    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  
    Blob<Dtype> diff_;
    bool has_weights_;
    Dtype negative_ratio_;
    Dtype eps_;
    Dtype grad_clip_;
    float negative_sample_prob_;
    Blob<float> rand_mask_;
};


} // namespace caffe

#endif // CAFFE_HEATMAP_LOSS_LAYER_HPP_