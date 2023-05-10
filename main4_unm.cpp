// ***********************************************************************
//
// Demo program for education in subject
// Computer Architectures and Parallel Systems.
// Petr Olivka, dep. of Computer Science, FEI, VSB-TU Ostrava, 2020/11
// email:petr.olivka@vsb.cz
//
// Example of CUDA Technology Usage with unified memory.
//
// Image transformation from RGB to BW schema. 
// Image manipulation is performed by OpenCV library. 
//
// ***********************************************************************

#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include "uni_mem_allocator.h"
#include "cuda_img.h"

namespace cv {
}

// Function prototype from .cu file
void cu_run_grayscale( CudaImg t_bgr_cuda_img, CudaImg t_bw_cuda_img );
void cu_rotate_90_clockwise(CudaImg input_cuda_img, CudaImg output_cuda_img);
void cu_rotate_90_anticlokwise(CudaImg input_cuda_img, CudaImg output_cuda_img);

int main( int t_numarg, char **t_arg )
{
    // Uniform Memory allocator for Mat
    UniformAllocator allocator;
    cv::Mat::setDefaultAllocator( &allocator );

    if ( t_numarg < 2 )
    {
        printf( "Enter picture filename!\n" );
        return 1;
    }

    // Load image
    cv::Mat input_image_mat = cv::imread( t_arg[ 1 ], cv::IMREAD_UNCHANGED ); // CV_LOAD_IMAGE_COLOR );

    if ( !input_image_mat.data )
    {
        printf( "Unable to read file '%s'\n", t_arg[ 1 ] );
        return 1;
    }

    // create empty BW image
    cv::Size rotated_size(input_image_mat.size().height, input_image_mat.size().width);
    cv::Mat output_image_mat( rotated_size, CV_8UC4  );

    // data for CUDA
    CudaImg input_image, output_image;
    input_image.m_size.x  = input_image_mat.size().width;
    input_image.m_size.y = input_image_mat.size().height;

    output_image.m_size.x  = output_image_mat.size().width;
    output_image.m_size.y = output_image_mat.size().height;

    input_image.m_p_uchar4 = ( uchar4 * ) input_image_mat.data;
    output_image.m_p_uchar4 = ( uchar4 * ) output_image_mat.data;

    // Function calling from .cu file
    cu_rotate_90_anticlokwise( input_image, output_image );

    // Show the Color and BW image
    cv::imshow( "Original image", input_image_mat );
    cv::imshow( "Rotated 90 clockwise", output_image_mat );
    cv::waitKey( 0 );
}

