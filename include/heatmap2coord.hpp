// ------------------------------------------------------------------
// Convert heatmap into coordinates(x, y) anc concat along input
// channels.
// Written by ZhouJunr
// ------------------------------------------------------------------

#ifndef CAFFE_HEATMAP2COORD_LAYER_HPP_
#define CAFFE_HEATMAP2COORD_LAYER_HPP_

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Turn heatmap into coordinates(x, y). Assume input shape
 * [n, c, h, w], output shape will be [n, 2*c]
 */
template <typename Dtype>
class Heatmap2CoordLayer : public Layer<Dtype> {
  public:
    explicit Heatmap2CoordLayer(const LayerParameter& param)
        : Layer<Dtype>(param) {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "Heatmap2coord"; }
    virtual inline int ExactNumBottomBlobs() const { return 1; }
    virtual inline int ExactNumTopBlobs() const { return 1; }

  protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    
    int num_points_;
};

} // namespace caffe

#endif