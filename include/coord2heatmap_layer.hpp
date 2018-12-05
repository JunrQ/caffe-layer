// ------------------------------------------------------------------
// Convert coordinates into multiple heatmaps.
// Written by ZhouJunr
// ------------------------------------------------------------------

#ifndef CAFFE_COORD2HEATMAP_LAYER_HPP_
#define CAFFE_COORD2HEATMAP_LAYER_HPP_

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief A layer turn coordinates to a heatmap.
 * coordinates: x_0, y_0, x_1, y_1, ...
 * heatmaps: (batch_size, num_points, h, w), all pixels are 0. except
 * keypoints, which are 1.
 */
template <typename Dtype>
class Coord2heatmapLayer : public Layer<Dtype> {
  public:
    explicit Coord2heatmapLayer(const LayerParameter& param)
        : Layer<Dtype>(param) {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    
    virtual inline const char* type() const { return "Coord2heatmap"; }
    virtual inline int ExactNumBottomBlobs() const { return 1; }
    virtual inline int ExactNumTopBlobs() const { return 1; }
  
  protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
      NOT_IMPLEMENTED;
    }

    int output_height_;
    int output_width_;
    int num_points_;
    int max_value_;
    int radius_;
};

} // namespace caffe

#endif // CAFFE_COORD2HEATMAP_LAYER_HPP_