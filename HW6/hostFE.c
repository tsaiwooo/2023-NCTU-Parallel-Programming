#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    
    cl_int status;
    int filterSize = filterWidth * filterWidth;

    cl_command_queue queue;
    queue = clCreateCommandQueue(*context, *device, 0, NULL);

    // Create memory buffers for input and output
    cl_mem input_buffer,filter_buffer,output_buffer;
    input_buffer = clCreateBuffer(*context, CL_MEM_READ_ONLY, sizeof(float) * imageHeight * imageWidth, NULL, NULL);
    filter_buffer = clCreateBuffer(*context, CL_MEM_READ_ONLY, sizeof(float) * filterSize, NULL, NULL);
    output_buffer = clCreateBuffer(*context, CL_MEM_WRITE_ONLY, sizeof(float) * imageHeight * imageWidth, NULL, NULL);

    // Write input array to memory buffer
    clEnqueueWriteBuffer(queue, input_buffer, CL_TRUE, 0, sizeof(float) * imageHeight * imageWidth, inputImage, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, filter_buffer, CL_TRUE, 0, sizeof(float) * filterSize, filter, 0, NULL, NULL);

    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(*program, "convolution", NULL);

    // Set the arguments of the kernel
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &filter_buffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &output_buffer);
    clSetKernelArg(kernel, 3, sizeof(int), &imageHeight);
    clSetKernelArg(kernel, 4, sizeof(int), &imageWidth);
    clSetKernelArg(kernel, 5, sizeof(int), &filterWidth);

    // Execute the OpenCL kernel
    size_t globalSize[2] = { imageHeight, imageWidth };
    size_t localSize[2] = { 8, 8 };
    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, localSize, 0, NULL, NULL);

    // Read the result from the memory buffer
    clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0, sizeof(float) * imageHeight * imageWidth, outputImage, 0, NULL, NULL);


    // Clean up
    clReleaseKernel(kernel);
    clReleaseMemObject(input_buffer);
    clReleaseMemObject(filter_buffer);
    clReleaseMemObject(output_buffer);
    clReleaseCommandQueue(queue);

    return 0;
}