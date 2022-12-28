import tensorflow as tf
from tensorflow.python.framework import ops
import roi_align_op

@ops.RegisterGradient("RoiAlign")
def _roi_align_grad(op, grad):

  data = op.inputs[0]
  rois = op.inputs[1]
 
  pooled_height = op.get_attr('pooled_height')
  pooled_width = op.get_attr('pooled_width')
  spatial_scale = op.get_attr('spatial_scale')

  # compute gradient
  data_grad = roi_align_op.roi_align_grad(data, rois, grad, pooled_height, pooled_width, spatial_scale)

  return [data_grad, None]  # List of one Tensor, since we have one input
