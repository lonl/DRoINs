#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <stdio.h>
#include <cfloat>
#include "roi_align_op_gpu.h"
#include "cuda_kernel_helper.h"

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

using std::max;
using std::min;

using namespace tensorflow;


    __global__ void ROIAlignForward(const int nthreads, const float* bottom_data, const float spatial_scale, const int height, const int width, const int channels, const int aligned_height, const int aligned_width, const float* bottom_rois, float* top_data) {
        CUDA_1D_KERNEL_LOOP(index, nthreads) {
            // (n, c, ph, pw) is an element in the aligned output
            int n = index;
            int pw = n % aligned_width;
            n /= aligned_width;
            int ph = n % aligned_height;
            n /= aligned_height;
            int c = n % channels;
            n /= channels;

            bottom_rois += n * 5;
            float roi_batch_ind = bottom_rois[0];
            float roi_start_w = bottom_rois[1] * spatial_scale;
            float roi_start_h = bottom_rois[2] * spatial_scale;
            float roi_end_w = bottom_rois[3] * spatial_scale;
            float roi_end_h = bottom_rois[4] * spatial_scale;

            // Force malformed ROIs to be 1x1
            float roi_width = fmaxf(roi_end_w - roi_start_w + 1., 0.);
            float roi_height = fmaxf(roi_end_h - roi_start_h + 1., 0.);
            float bin_size_h = roi_height / (aligned_height - 1.);
            float bin_size_w = roi_width / (aligned_width - 1.);

            float h = (float)(ph) * bin_size_h + roi_start_h;
            float w = (float)(pw) * bin_size_w + roi_start_w;

            int hstart = fminf(floor(h), height - 2);
            int wstart = fminf(floor(w), width - 2);

            int img_start = roi_batch_ind * channels * height * width;

            // bilinear interpolation
            if (h < 0 || h >= height || w < 0 || w >= width) {
                top_data[index] = 0.;
            } else {
                float h_ratio = h - (float)(hstart);
                float w_ratio = w - (float)(wstart);
                int upleft = img_start + (c * height + hstart) * width + wstart;
                int upright = upleft + 1;
                int downleft = upleft + width;
                int downright = downleft + 1;

                top_data[index] = bottom_data[upleft] * (1. - h_ratio) * (1. - w_ratio)
                    + bottom_data[upright] * (1. - h_ratio) * w_ratio
                    + bottom_data[downleft] * h_ratio * (1. - w_ratio)
                    + bottom_data[downright] * h_ratio * w_ratio;
            }
        }
    }


    bool ROIAlignForwardLaucher(const float* bottom_data, const float spatial_scale, const int num_rois, const int height, const int width, const int channels, const int aligned_height, const int aligned_width, const float* bottom_rois, float* top_data, const Eigen::GpuDevice& d) {
        const int kThreadsPerBlock = 1024;
        const int output_size = num_rois * aligned_height * aligned_width * channels;
        cudaError_t err;


        ROIAlignForward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, d.stream()>>>(output_size, bottom_data, spatial_scale, height, width, channels, aligned_height, aligned_width, bottom_rois, top_data);

        err = cudaGetLastError();
        if(cudaSuccess != err) {
            fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
            exit( -1 );
        }

        return d.ok();
    }


    __global__ void ROIAlignBackward(const int nthreads, const float* top_diff, const float spatial_scale, const int height, const int width, const int channels, const int aligned_height, const int aligned_width, float* bottom_diff, const float* bottom_rois) {
        CUDA_1D_KERNEL_LOOP(index, nthreads) {

            // (n, c, ph, pw) is an element in the aligned output
            int n = index;
            int pw = n % aligned_width;
            n /= aligned_width;
            int ph = n % aligned_height;
            n /= aligned_height;
            int c = n % channels;
            n /= channels;

            bottom_rois += n * 5;
            float roi_batch_ind = bottom_rois[0];
            float roi_start_w = bottom_rois[1] * spatial_scale;
            float roi_start_h = bottom_rois[2] * spatial_scale;
            float roi_end_w = bottom_rois[3] * spatial_scale;
            float roi_end_h = bottom_rois[4] * spatial_scale;
            /* int roi_start_w = round(bottom_rois[1] * spatial_scale); */
            /* int roi_start_h = round(bottom_rois[2] * spatial_scale); */
            /* int roi_end_w = round(bottom_rois[3] * spatial_scale); */
            /* int roi_end_h = round(bottom_rois[4] * spatial_scale); */

            // Force malformed ROIs to be 1x1
            float roi_width = fmaxf(roi_end_w - roi_start_w + 1., 0.);
            float roi_height = fmaxf(roi_end_h - roi_start_h + 1., 0.);
            float bin_size_h = roi_height / (aligned_height - 1.);
            float bin_size_w = roi_width / (aligned_width - 1.);

            float h = (float)(ph) * bin_size_h + roi_start_h;
            float w = (float)(pw) * bin_size_w + roi_start_w;

            int hstart = fminf(floor(h), height - 2);
            int wstart = fminf(floor(w), width - 2);

            int img_start = roi_batch_ind * channels * height * width;

            // bilinear interpolation
            if (!(h < 0 || h >= height || w < 0 || w >= width)) {
                float h_ratio = h - (float)(hstart);
                float w_ratio = w - (float)(wstart);
                int upleft = img_start + (c * height + hstart) * width + wstart;
                int upright = upleft + 1;
                int downleft = upleft + width;
                int downright = downleft + 1;

                atomicAdd(bottom_diff + upleft, top_diff[index] * (1. - h_ratio) * (1 - w_ratio));
                atomicAdd(bottom_diff + upright, top_diff[index] * (1. - h_ratio) * w_ratio);
                atomicAdd(bottom_diff + downleft, top_diff[index] * h_ratio * (1 - w_ratio));
                atomicAdd(bottom_diff + downright, top_diff[index] * h_ratio * w_ratio);
            }
        }
    }

    bool ROIAlignBackwardLaucher(const float* top_diff, const float spatial_scale, const int batch_size, const int num_rois, const int height, const int width, const int channels, const int aligned_height, const int aligned_width, const float* bottom_rois, float* bottom_diff, const Eigen::GpuDevice& d) {
        const int kThreadsPerBlock = 1024;
        const int output_size = num_rois * aligned_height * aligned_width * channels;
        cudaError_t err;

        ROIAlignBackward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, d.stream()>>>(output_size, top_diff, spatial_scale, height, width, channels, aligned_height, aligned_width, bottom_diff, bottom_rois);

        err = cudaGetLastError();
        if(cudaSuccess != err) {
            fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
            exit( -1 );
        }

        return d.ok();
    }
#endif  // GOOGLE_CUDA

