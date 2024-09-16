#include <cuda_runtime.h>
#include <device_launch_parameter.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

// Static shmem calculation for convenience (Int 16x16 matrix)
#define SHMEM_SIZE 16*16*4

// Initialization function for matrix
void init_matrix(int *a, int *b, int n){
    for (int i=0; i<n; i++){
        for (int j =0; j<n; j++){
            a[i*n+j] = rand() % 100;
            b[i*n+j] = rand() % 100;
        }
    }
}

// Kernel
__global__ void tiledMatrixMul(int *a, int *b, int *c, int n, int tile_size){

    // Two statically-sized pieces of shared memory
    __shared__ int A[SHMEM_SIZE];
    __shared__ int B[SHMEM_SIZE];

    // Shorten these parameters for clean re-use
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Calculate global row and column position for this thread
    int row = by * tile_size + ty;
    int col = bx * tile_size + tx;
   
    // Intermediate sum for element being written
    int temp_sum = 0;

    // Sweap tiles over entire matrix
    for (int i=0; i<(n/tile_size); i++){
        A[ty*tile_size + tx] = a[row*n + (i*tile_size + tx)];
        B[ty*tile_size + tx] = b[(i*tile_size*n +ty*n) + col];

        //Ensure all threads have loaded their data before processing
        __syncthreads();

        //  calculate all temp values for this tile
        for (int j=0; j< tile_size; j++){
            temp_val += A[(ty*tile_size)+j]*B[(j*tile_size)+tx];
        }

        // Ensure some threads don't progress and stop current shared memory values
        __syncthreads();
    }
    c[(row*n)+col] = temp_val;
}

int main(){
    // Matrix size of 1024x1024
    int n = 1<<10;

    //size (in bytes) of matrix
    size_t bytes = n*sizeof(int);

    // Host patrameters
    int *h_a, *h_b, *h_c;

    // Allocate memory
    h_a = (int*)malloc(bytes);
    h_b = (int*)malloc(bytes);
    h_c = (int*)malloc(bytes);

    // device pointers
    int *d_a, *d_b, *d_c;

    // Allocate device memory
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Initialize Matrices
    init_matrix(h_a, h_b, n);

    // Copy data to the device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, bytes, cudaMemcpyHostToDevice);

    // Threads per block
    int BLOCK_SIZE = 16;

    //Blocks in each dimension
    int GRID_SIZE = (int)ceil(n/BLOCK_SIZE);

    // Use dim3 objects
    dim3 blocks(GRID_SIZE, GRID_SIZE);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

    // Launch Kernel
    tiledMatrixMul <<<blocks, threads>>> (d_a, d_b, d_c, n, BLOCK_SIZE);

    // Copy back to the host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    printf("Task Completed \n");

    //free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    //free host memory
    free(h_a);
    free(h_b);
    free(h_c);
}