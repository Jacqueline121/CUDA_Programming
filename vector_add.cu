#include <stdio.h>
#include <assert.h>

#define VECTOR_LENGTH 10000
#define MAX_ERR 1e-4

__global__ void vector_add(float *out, float *a, float *b, int n){
    for(int i=0; i < n; i++){
        out[i] = a[i] + b[i];
    }
}

int main(){
    float *a, *b, *out;
    float *d_a, *d_b, *d_out;

    // step1: allocate memory on CPU
    a = (float*)malloc(sizeof(float)*VECTOR_LENGTH);
    b = (float*)malloc(sizeof(float)*VECTOR_LENGTH);
    out = (float*)malloc(sizeof(float)*VECTOR_LENGTH);

    // step2: data initilization
    for(int i = 0; i < VECTOR_LENGTH; i++){
        a[i] = 3.0f;
        b[i] = 0.14f;
    }

    // step3: allocate memory on GPU
    cudaMalloc((void**)&d_a, sizeof(float)*VECTOR_LENGTH);
    cudaMalloc((void**)&d_b, sizeof(float)*VECTOR_LENGTH);
    cudaMalloc((void**)&d_out, sizeof(float)*VECTOR_LENGTH);

    // step4: transfer input data from host(CPU) to device(GPU) memory
    cudaMemcpy(d_a, a, sizeof(float)*VECTOR_LENGTH, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float)*VECTOR_LENGTH, cudaMemcpyHostToDevice);

    // step5: execute kernel function on GPU
    vector_add<<<1, 1>>>(d_out, d_a, d_b, VECTOR_LENGTH);

    // step6: transfer output from device(GPU) memory to host(CPU)
    cudaMemcpy(out, d_out, sizeof(float)*VECTOR_LENGTH, cudaMemcpyDeviceToHost);

    for(int i = 0; i < VECTOR_LENGTH; i++){
        assert(fabs(out[i] - a[i] - b[i]) < MAX_ERR);
    }

    printf("out[0] is %f\n", out[0]);
    printf("PASSED\n");

    // step7: free the memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    free(a);
    free(b);
    free(out);
}