__kernel void convolution(__global float* input,__global float* filter, __global float* output,
                            const int imageHeight, const int imageWidth, const int filterWidth) 
{
   int i = get_global_id(0);
   int j = get_global_id(1);

   int halffilterSize = filterWidth / 2;
   int  k, l;
   float sum=0;

   __local float localFilter[100];
   localFilter[get_local_id(0) * filterWidth + get_local_id(1)] = filter[get_local_id(0) * filterWidth + get_local_id(1)];


   for (k = -halffilterSize; k <= halffilterSize; ++k)
   {
        for (l = -halffilterSize; l <= halffilterSize; ++l)
        {
            if (i + k >= 0 && i + k < imageHeight &&
                j + l >= 0 && j + l < imageWidth)
            {
                sum += input[(i + k) * imageWidth + j + l] *
                       localFilter[(k + halffilterSize) * filterWidth +
                              l + halffilterSize];
            }
        }
    }
    output[i * imageWidth + j] = sum;
}
