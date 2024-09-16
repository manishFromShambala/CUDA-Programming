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
__global__ void matrixMul(int *a, int *b, int *c, int n){
    // Compute each thread's row and column
    int row = blockIdx.y *blockDim.y + threadIdx.y;
    int col = blockIdx.x *blockDim.x + threadIdx.x;

    int temp_sum = 0;
    // Boundary
    if (row<n && col<n){
        for(int k=0; K<n; k++){
            temp_sum +=a[row*n+k]*b[k*n+col];
        }
        col[row*n+col]=temp_sum;
    }
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
    matrixMul <<<blocks, threads>>> (d_a, d_b, d_c, n);

    // Copy back to the host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    printf("Task Completed \n");
}