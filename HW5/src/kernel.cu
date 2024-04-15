#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
// #include <omp.h>


__global__ void mandelKernel(float lowerX, float lowerY, int* ker_mem, int resX, int resY, int maxIterations,float stepX,float stepY) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;

    // int totalX = blockDim.x * gridDim.x;
    // int totalY = blockDim.y * gridDim.y;

    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int globalIdy = blockIdx.y * blockDim.y + threadIdx.y;
    if(globalIdx >= resX) return;
    if(globalIdy >= resY) return;

    // float stepX = (upperX - lowerX) / resX;
    // float stepY = (upperY - lowerY) / resY;



    float x = lowerX + globalIdx * stepX;
    float y = lowerY + globalIdy * stepY;
    int index = (globalIdy*resX + globalIdx);
        
    float z_re = x, z_im = y;
    int k;
    #pragma unroll
    for (k = 0; k < maxIterations; ++k)
    {
        if (z_re * z_re + z_im * z_im > 4.f)
        break;
        float new_re = z_re * z_re - z_im * z_im;
        float new_im = 2.f * z_re * z_im;
        z_re = x + new_re;
        z_im = y + new_im;
    }
    ker_mem[index] = k;
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{

    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;
    /*   kernel memory    */
    int *ker_mem;
    size_t pitch;
    cudaMallocPitch((void **)&ker_mem,&pitch,sizeof(int) * resX ,resY);
    // cudaMalloc((void**)&ker_mem,sizeof(int) * resX * resY);

    /*    Blcok setting    */
    dim3 dimGrid(resX/16,resY/16);
    dim3 dimBlock(16,16);
    
    mandelKernel<<<dimGrid,dimBlock>>>(lowerX,lowerY, ker_mem, resX, resY, maxIterations,stepX,stepY);
    cudaDeviceSynchronize();

    cudaMemcpy(img,ker_mem,sizeof(int)*resX*resY,cudaMemcpyDeviceToHost);

    cudaFree(ker_mem);
}

