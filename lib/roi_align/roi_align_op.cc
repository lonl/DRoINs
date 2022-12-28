#include <stdio.h>
#include <cfloat>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;
typedef Eigen::ThreadPoolDevice CPUDevice;

REGISTER_OP("RoiAlign")
    .Attr("T: {float, double}")
    .Attr("pooled_height: int")
    .Attr("pooled_width: int")
    .Attr("spatial_scale: float")
    .Input("bottom_data: T")
    .Input("bottom_rois: T")
    .Output("top_data: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ::tensorflow::shape_inference::ShapeHandle dims;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &dims));
      ::tensorflow::shape_inference::DimensionHandle channels;
      channels = c->Dim(dims, 3);

      ::tensorflow::shape_inference::ShapeHandle dims_rois;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(1), 1, &dims_rois));
      ::tensorflow::shape_inference::DimensionHandle num_rois;
      num_rois = c->Dim(dims_rois, 0);

      int64 pooled_height;
      int64 pooled_width;
      TF_RETURN_IF_ERROR(c->GetAttr("pooled_height", &pooled_height));
      TF_RETURN_IF_ERROR(c->GetAttr("pooled_width", &pooled_width));
      ::tensorflow::shape_inference::ShapeHandle output_shape =\
         c->MakeShape({num_rois, pooled_height, pooled_width, channels});
      c->set_output(0, output_shape);
      return ::tensorflow::Status::OK();
    });


REGISTER_OP("RoiAlignGrad")
    .Attr("T: {float, double}")
    .Attr("pooled_height: int")
    .Attr("pooled_width: int")
    .Attr("spatial_scale: float")
    .Input("bottom_data: T")
    .Input("bottom_rois: T")
    .Input("grad: T")
    .Output("output: T");



template <typename Device, typename T>
class RoiAlignOp : public OpKernel {
 public:
  explicit RoiAlignOp(OpKernelConstruction* context) : OpKernel(context) {}
  void Compute(OpKernelContext* context) override {}
 private:
  int pooled_height_;
  int pooled_width_;
  float spatial_scale_;
};


bool ROIAlignForwardLaucher(const float* bottom_data, const float spatial_scale, const int num_rois, const int height, const int width, const int channels, const int aligned_height, const int aligned_width, const float* bottom_rois, float* top_data, const Eigen::GpuDevice& d);

static void RoiAlignKernel(
    OpKernelContext* context, const Tensor* bottom_data, const float spatial_scale, const int num_rois, const int channels,
    const int height, const int width, const int pooled_height, const int pooled_width, const Tensor* bottom_rois, const TensorShape& tensor_output_shape) 
{
  Tensor* output = nullptr;
  OP_REQUIRES_OK(context, context->allocate_output(0, tensor_output_shape, &output));

  if (!context->status().ok()) {
    return;
  }

  ROIAlignForwardLaucher(
    bottom_data->flat<float>().data(), spatial_scale, num_rois,
    height, width, channels, pooled_height, pooled_width, bottom_rois->flat<float>().data(), 
    output->flat<float>().data(), context->eigen_device<Eigen::GpuDevice>());
}

template <class T>
class RoiAlignOp<Eigen::GpuDevice, T> : public OpKernel {
 public:
  typedef Eigen::GpuDevice Device;

  explicit RoiAlignOp(OpKernelConstruction* context) : OpKernel(context) {

    OP_REQUIRES_OK(context,
                   context->GetAttr("pooled_height", &pooled_height_));
    
    OP_REQUIRES_OK(context,
                   context->GetAttr("pooled_width", &pooled_width_));
      
    OP_REQUIRES_OK(context,
                   context->GetAttr("spatial_scale", &spatial_scale_));
  }

  void Compute(OpKernelContext* context) override 
  {
    // Grab the input tensor
    const Tensor& bottom_data = context->input(0);
    const Tensor& bottom_rois = context->input(1);

    // data should have 4 dimensions.
    OP_REQUIRES(context, bottom_data.dims() == 4,
                errors::InvalidArgument("data must be 4-dimensional"));

    // rois should have 2 dimensions.
    OP_REQUIRES(context, bottom_rois.dims() == 2,
                errors::InvalidArgument("rois must be 2-dimensional"));

    // Number of ROIs
    int num_rois = bottom_rois.dim_size(0);
    // batch size
    int batch_size = bottom_data.dim_size(0);
    // data height
    int data_height = bottom_data.dim_size(1);
    // data width
    int data_width = bottom_data.dim_size(2);
    // Number of channels
    int num_channels = bottom_data.dim_size(3);
    
    // construct the output shape
    int dims[4];
    dims[0] = num_rois;
    dims[1] = pooled_height_;
    dims[2] = pooled_width_;
    dims[3] = num_channels;
    TensorShape output_shape;
    TensorShapeUtils::MakeShape(dims, 4, &output_shape);

    RoiAlignKernel(context, &bottom_data, spatial_scale_, num_rois, num_channels, data_height, data_width, 
                       pooled_height_, pooled_width_, &bottom_rois, output_shape);

  }
 private:
  int pooled_height_;
  int pooled_width_;
  float spatial_scale_;
};

REGISTER_KERNEL_BUILDER(Name("RoiAlign").Device(DEVICE_GPU).TypeConstraint<float>("T"), RoiAlignOp<Eigen::GpuDevice, float>);








bool ROIAlignBackwardLaucher(const float* top_diff, const float spatial_scale, const int batch_size, const int num_rois, const int height, const int width, const int channels, const int aligned_height, const int aligned_width, const float* bottom_rois, float* bottom_diff, const Eigen::GpuDevice& d);

static void RoiAlignGradKernel(
    OpKernelContext* context, const Tensor* bottom_data, const Tensor* bottom_rois, const Tensor* out_backprop,
    const float spatial_scale, const int batch_size, const int num_rois, const int height,
    const int width, const int channels, const int pooled_height,
    const int pooled_width, const TensorShape& tensor_output_shape) 
{
  Tensor* output = nullptr;
  OP_REQUIRES_OK(context, context->allocate_output(0, tensor_output_shape, &output));

  if (!context->status().ok()) {
    return;
  }

  ROIAlignBackwardLaucher(
    out_backprop->flat<float>().data(), spatial_scale, batch_size, num_rois, height,
    width, channels, pooled_height, pooled_width, bottom_rois->flat<float>().data(),
    output->flat<float>().data(), context->eigen_device<Eigen::GpuDevice>());
}


// compute gradient
template <class Device, class T>
class RoiAlignGradOp : public OpKernel {
 public:
  explicit RoiAlignGradOp(OpKernelConstruction* context) : OpKernel(context) {

    // Get the pool height
    OP_REQUIRES_OK(context,
                   context->GetAttr("pooled_height", &pooled_height_));
    // Check that pooled_height is positive
    OP_REQUIRES(context, pooled_height_ >= 0,
                errors::InvalidArgument("Need pooled_height >= 0, got ",
                                        pooled_height_));
    // Get the pool width
    OP_REQUIRES_OK(context,
                   context->GetAttr("pooled_width", &pooled_width_));
    // Check that pooled_width is positive
    OP_REQUIRES(context, pooled_width_ >= 0,
                errors::InvalidArgument("Need pooled_width >= 0, got ",
                                        pooled_width_));
    // Get the spatial scale
    OP_REQUIRES_OK(context,
                   context->GetAttr("spatial_scale", &spatial_scale_));
  }

  void Compute(OpKernelContext* context) override 
  {
    // Grab the input tensor
    const Tensor& bottom_data = context->input(0);
    const Tensor& bottom_rois = context->input(1);
    const Tensor& out_backprop = context->input(2);

    // data should have 4 dimensions.
    OP_REQUIRES(context, bottom_data.dims() == 4,
                errors::InvalidArgument("data must be 4-dimensional"));

    // rois should have 2 dimensions.
    OP_REQUIRES(context, bottom_rois.dims() == 2,
                errors::InvalidArgument("rois must be 2-dimensional"));

    OP_REQUIRES(context, out_backprop.dims() == 4,
                errors::InvalidArgument("out_backprop must be 4-dimensional"));

    // Number of ROIs
    int num_rois = bottom_rois.dim_size(0);
    // batch size
    int batch_size = bottom_data.dim_size(0);
    // data height
    int height = bottom_data.dim_size(1);
    // data width
    int width = bottom_data.dim_size(2);
    // Number of channels
    int channels = bottom_data.dim_size(3);

    // construct the output shape
    TensorShape output_shape = bottom_data.shape();

    RoiAlignGradKernel(
      context, &bottom_data, &bottom_rois, &out_backprop,
      spatial_scale_, batch_size, num_rois, height, width, channels, pooled_height_,
      pooled_width_, output_shape);

  }
 private:
  int pooled_height_;
  int pooled_width_;
  float spatial_scale_;
};

REGISTER_KERNEL_BUILDER(Name("RoiAlignGrad").Device(DEVICE_GPU).TypeConstraint<float>("T"), RoiAlignGradOp<Eigen::GpuDevice, float>);




