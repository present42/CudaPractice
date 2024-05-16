#include <iostream>
#include <math.h>

// CUDA kernel function to add the elements of two
// arrays on GPU
__global__ // these global functions are known as kernels
void add(int n, float *x, float *y) {
    // threadIdx.x contains the index of the current thread within its block
    // blockDim.x contains the number of threads in the block

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride) {
        y[i] = x[i] + y[i];
    }
}

int main(void) {
    int N = 1 << 20;

    // CPU Version
    // float *x = new float[N];
    // float *y = new float[N];
    
    float *x, *y;
    // allocate data in unified memory
    // accessible from gpu & cpu
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    add<<<numBlocks, blockSize>>>(N, x, y);
    // add(N, x, y);
    // run kernel on 1M element on GPU
    // add<<<1, 256>>>(N, x, y);
    // add<<<1, 256>>>(N, x, y); will do the computation
    // once per thread, rather than spreading the
    // computation across the parallel threads

    // need CPU to wait until the kernel is done
    // before it accesses the results
    cudaDeviceSynchronize();

    // check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++) {
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    }
    std::cout << "Max error: " << maxError << std::endl;

    // delete [] x;
    // delete [] y;

    cudaFree(x);
    cudaFree(y);

    return 0;
}