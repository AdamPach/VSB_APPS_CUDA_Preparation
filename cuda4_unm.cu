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
//
// ***********************************************************************

#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>

#include "cuda_img.h"

// Demo kernel to transform RGB color schema to BW schema
__global__ void kernel_grayscale( CudaImg t_color_cuda_img, CudaImg t_bw_cuda_img )
{
    // X,Y coordinates and check image dimensions
    int l_y = blockDim.y * blockIdx.y + threadIdx.y;
    int l_x = blockDim.x * blockIdx.x + threadIdx.x;
    if ( l_y >= t_color_cuda_img.m_size.y ) return;
    if ( l_x >= t_color_cuda_img.m_size.x ) return;

    // Get point from color picture
    uchar3 l_bgr = t_color_cuda_img.m_p_uchar3[ l_y * t_color_cuda_img.m_size.x + l_x ];

    // Store BW point to new image
   // t_bw_cuda_img.m_p_uchar1[ l_y * t_bw_cuda_img.m_size.x + l_x ].x = l_bgr.x * 0.11 + l_bgr.y * 0.59 + l_bgr.z * 0.30;
   int flip_index = t_color_cuda_img.m_size.x - l_x;
   t_bw_cuda_img.m_p_uchar3[ l_y * t_bw_cuda_img.m_size.x + flip_index ] = l_bgr ;
}

void cu_run_grayscale( CudaImg t_color_cuda_img, CudaImg t_bw_cuda_img )
{
    cudaError_t l_cerr;

    // Grid creation, size of grid must be equal or greater than images
    int l_block_size = 16;
    dim3 l_blocks( ( t_color_cuda_img.m_size.x + l_block_size - 1 ) / l_block_size, ( t_color_cuda_img.m_size.y + l_block_size - 1 ) / l_block_size );
    dim3 l_threads( l_block_size, l_block_size );
    kernel_grayscale<<< l_blocks, l_threads >>>( t_color_cuda_img, t_bw_cuda_img );

    if ( ( l_cerr = cudaGetLastError() ) != cudaSuccess )
        printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( l_cerr ) );

    cudaDeviceSynchronize();
}

//rotation 90 clockwise


__global__ void kernel_rotate_90_clockwise(CudaImg input_cuda_img, CudaImg output_cuda_img)
{
    int l_y = blockDim.y * blockIdx.y + threadIdx.y;
    int l_x = blockDim.x * blockIdx.x + threadIdx.x;
    if ( l_y >= input_cuda_img.m_size.y ) return;
    if ( l_x >= input_cuda_img.m_size.x ) return;
   
    int new_x = input_cuda_img.m_size.y - l_y;
    int new_y = l_x;

    output_cuda_img.at4(new_x, new_y) = input_cuda_img.at4(l_x, l_y);
}

void cu_rotate_90_clockwise(CudaImg input_cuda_img, CudaImg output_cuda_img)
{
    cudaError_t cuda_error;

    int l_block_size = 16;
    dim3 l_blocks( ( input_cuda_img.m_size.x + l_block_size - 1 ) / l_block_size, ( input_cuda_img.m_size.y + l_block_size - 1 ) / l_block_size );
    dim3 l_threads( l_block_size, l_block_size );

    kernel_rotate_90_clockwise<<<l_blocks, l_threads>>>(input_cuda_img, output_cuda_img);

    if ( ( cuda_error = cudaGetLastError() ) != cudaSuccess )
        printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( cuda_error ) );

    cudaDeviceSynchronize();
}

__global__ void kernel_rotate_90_anticlockwise(CudaImg input_cuda_img, CudaImg output_cuda_img)
{
    int l_y = blockDim.y * blockIdx.y + threadIdx.y;
    int l_x = blockDim.x * blockIdx.x + threadIdx.x;
    if ( l_y >= input_cuda_img.m_size.y ) return;
    if ( l_x >= input_cuda_img.m_size.x ) return;
   
    int new_x = l_y;
    int new_y = input_cuda_img.m_size.x - l_x;

    output_cuda_img.at4(new_x, new_y) = input_cuda_img.at4(l_x, l_y);
}

void cu_rotate_90_anticlokwise(CudaImg input_cuda_img, CudaImg output_cuda_img)
{
    cudaError_t cuda_error;

    int l_block_size = 16;
    dim3 l_blocks( ( input_cuda_img.m_size.x + l_block_size - 1 ) / l_block_size, ( input_cuda_img.m_size.y + l_block_size - 1 ) / l_block_size );
    dim3 l_threads( l_block_size, l_block_size );

    kernel_rotate_90_anticlockwise<<<l_blocks, l_threads>>>(input_cuda_img, output_cuda_img);

    if ( ( cuda_error = cudaGetLastError() ) != cudaSuccess )
        printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( cuda_error ) );

    cudaDeviceSynchronize();
}
