#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>


__global__ void mandelKernel(float stepX, float stepY, float lowerX, float lowerY, int* ker_mem, int resX, int resY, int maxIterations,int pitch) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;



    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int globalIdy = blockIdx.y * blockDim.y + threadIdx.y;




    float x = lowerX + globalIdx * stepX;
    float y = lowerY + globalIdy * stepY;

    int rowStart = globalIdy*pitch / sizeof(int);
    int index = rowStart + globalIdx;
        
    float z_re = x, z_im = y;
    int k;
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
    int *host_mem;
    size_t pitch;
    cudaMallocPitch((void **)&ker_mem,&pitch,sizeof(int) * resX ,resY);
    cudaHostAlloc((void **)&host_mem,sizeof(int) * resX * resY,cudaHostAllocDefault);

    /*    Blcok setting    */
    dim3 dimGrid(resX/16,resY/16);
    dim3 dimBlock(16,16);
    
    mandelKernel<<<dimGrid,dimBlock>>>(stepX,stepY,lowerX,lowerY, ker_mem, resX, resY, maxIterations , pitch);
    cudaDeviceSynchronize();

    cudaMemcpy2D(host_mem,resX * sizeof(int) , ker_mem , pitch , resX * sizeof(int),resY,cudaMemcpyDeviceToHost);
    // cudaMemcpy2D(host_mem, pitch , ker_mem , pitch , resX * sizeof(int) , resY, cudaMemcpyDeviceToHost);
    memcpy(img,host_mem,sizeof(int)*resX*resY);

    cudaFree(ker_mem);
    cudaFreeHost(host_mem);
}

