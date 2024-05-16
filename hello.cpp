#include <iostream>
#include <math.h>

// CUDA kernel function to add the elements of two
// arrays on GPU
__global__ // these global functions are known as kernels
void add(int n, float *x, float *y) {
    for (int i = 0; i < n; i++) {
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

    // add(N, x, y);
    // run kernel on 1M element on GPU
    add<<<1, 1>>>(N, x, y);

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