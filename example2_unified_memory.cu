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
    //Get the device ID for other CUDA calls
    int id = cudaGetDevice(&id);

    //vector size of 2^16 (65536 elements)
    int n=1<<16;

    // Allocate size for all vectors
    size_t bytes = sizeof(int) * n;

    // Declare Unified memory pointers
    int *a, *b, *c;
    
    // Allocate memory for these pointers - automatic transfer of data from cpu to GPU or vice-versae when needed
    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);

    // Initialize vectors a and b with random values between 0 and 99
    matrix_init(a, n);  // Defined function
    matrix_init(b, n);  // Defined function

    // ThreadBlock size  - we have to do tuning inorder to find correct block size
    int NUM_THREADS = 256;

    // Grid size
    int NUM_Blocks = (int)ceil(n / NUM_THREADS); // So that all the threads perform one operation each

    // Launch kernel on default stream without shared memory
    vectorAdd<<<NUM_Blocks, NUM_THREADS>>>(a, b, c, n);

    // wait for all the previous operations before using values
    cudaDeviceSynchronize();
    // Copy sum vector from device to host

    //check result for errors
    // error_check(h_a, h_b, h_c, n)

    printf("Completed Successfully");

    return 0;
}