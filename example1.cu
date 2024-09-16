#include<stdio.h>
#include<math.h>
#include<assert.h>

//CUDA Kernel for vector Addition
__global__ void vectorAdd(int *a, int *b, int *c, int n){
    //Calculate global thread ID
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    //vector boundry guard
    if (tid<n){
        //Each thread adds a single element 
        c[tid] = a[tid] + b[tid];
    }
}

// Initialize vector of size n to int between 0 to 99
void matrix_init(int* a, int n){
    for(int i=0; i<n;i++){
        a[i] = rand()%100;
    }
}


int main(){
    //vector size of 2^16 (65536 elements)
    int n=1<<16;
    // Host vector pointers
    int *h_a, *h_b, *h_c;
    // Device vector pointers
    int *d_a, *d_b, *d_c;
    // Allocate size for all vectors
    size_t bytes = sizeof(int) * n;

    // Allocate host memory
    h_a = (int*)malloc(bytes);
    h_b = (int*)malloc(bytes);
    h_c = (int*)malloc(bytes);

    // Allocate device memory
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Initialize vectors a and b with random values between 0 and 99
    matrix_init(h_a, n);  // Defined function
    matrix_init(h_b, n);  // Defined function

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // ThreadBlock size  - we have to do tuning inorder to find correct block size
    int NUM_THREADS = 256;

    // Grid size
    int NUM_Blocks = (int)ceil(n / NUM_THREADS); // So that all the threads perform one operation each

    // Launch kernel on default stream without shared memory
    vectorAdd<<<NUM_Blocks, NUM_THREADS>>>(d_a, d_b, d_c, n);

    // Copy sum vector from device to host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    //check result for errors
    error_check(h_a, h_b, h_c, n)

    printf("Completed Successfully");

    return 0;
}
