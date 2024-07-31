#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void caculate_forcing_term(int* N, double* dx, int* lambda_squared, double* constant, double* dev_g, double* dev_u, double* dev_max_residue) {
    int i = blockIdx.x;
    int j = threadIdx.x;
    int g_i = i*(2*(*N)-1)+j;
    double term1, term2;
    term1 = exp(-(*lambda_squared)*((j*(*dx)-3)*(j*(*dx)-3)+(i*(*dx)-1)*(i*(*dx)-1)));
    term2 = exp(-(*lambda_squared)*((j*(*dx)-1)*(j*(*dx)-1)+(i*(*dx)-1)*(i*(*dx)-1)));
    dev_g[g_i] = (*constant)*(term1 - term2);
    dev_u[g_i] = 0;
    dev_max_residue[g_i] = 0;
    if(g_i < *N){
        dev_max_residue[g_i + (*N)*(2*(*N)-1)-1] = 0;
    }
}

__global__ void point_updates(int* N, double* omega, double* dx, double* dev_g, double* dev_u, double* dev_max_residue, int* colour){
    double residue;
    int i = blockIdx.x;
    int j = threadIdx.x;

    extern __shared__ double localu[];

    int g_i = j + i * blockDim.x;
    int g_i_up = g_i - 2*(*N) + 1;
    int g_i_down = g_i + 2*(*N) - 1; 

    int l_i = (2*(*N)-1) + j;
    int l_i_up = l_i - 2*(*N) + 1;
    int l_i_down = l_i + 2*(*N) - 1;

    localu[l_i] = dev_u[g_i];
    localu[l_i_up] = dev_u[g_i_up];
    localu[l_i_down] = dev_u[g_i_down];

    __syncthreads();

    if((blockIdx.x + threadIdx.x)%2 == *colour){ 
        if(threadIdx.x == 0){
            residue = 0.25*(localu[l_i_down] + localu[l_i_up] + localu[l_i + 1] - 3*localu[l_i] - (*dx)*(*dx)*dev_g[g_i]);
            dev_max_residue[g_i] = residue;
            dev_u[g_i] += (*omega)*residue; 
        }
        else if(threadIdx.x == blockDim.x - 1){
            residue = 0.25*(localu[l_i_down] + localu[l_i_up] + localu[l_i - 1] - 3*localu[l_i] - (*dx)*(*dx)*dev_g[g_i]);
            dev_max_residue[g_i] = residue;
            dev_u[g_i] += (*omega)*residue;            
        }
        else{
            residue = 0.25*(localu[l_i_down] + localu[l_i_up] + localu[l_i - 1] + localu[l_i + 1] - 4*localu[l_i] - (*dx)*(*dx)*dev_g[g_i]);
            dev_max_residue[g_i] = residue;
            dev_u[g_i] += (*omega)*residue;     
        }
    }    
}

__global__  void reduction(double* dev_test){
    int i = blockIdx.x;
    int j = threadIdx.x;
    int g_i =  j + i * blockDim.x;
    double x = dev_test[g_i];
    double y;
    int mask, level;
    for (mask = 1, level = 0; level < 6; mask *= 2, ++level) {
        y = __shfl_xor_sync(0xFFFFFFFF, x, mask);
        x = x > y ? x : y;
    }
    dev_test[g_i] = x;
}

double* reduction_call(int step, int N, double* test){
    int num_blocks, num_threads, check;
    check = N/pow(32,step-1);
    num_threads =  check > 1024? 1024 : check;
    num_blocks = check > 1024? check/1024 : 1;
    double* temp = (double*)malloc(check*sizeof(double));
    double* dev_temp;
    cudaMalloc((void**)&dev_temp,check*sizeof(double));
    int mul = step == 1? 1 : 32;
    for(int i = 0; i < check; ++i){
        temp[i] = test[i*mul]; 
    }
    cudaMemcpy(dev_temp, temp, check*sizeof(double), cudaMemcpyHostToDevice);
    reduction<<<num_blocks,num_threads>>>(dev_temp);
    cudaMemcpy(temp, dev_temp, check*sizeof(double), cudaMemcpyDeviceToHost);
    return temp;
}

int main(int argc, char* argv[]) {

    // start clock
    clock_t start = clock();

    // take inputs and open file
    int argi = 0;
    int N = atoi(argv[++argi]);
    double omega  = atof(argv[++argi]), tolerance = atof(argv[++argi]);
    int iterations = atoi(argv[++argi]);
    FILE *fileid = fopen("Sources.out", "w");

    // define forcing term and related constants
    double* g = (double*)malloc((2*N-1)*N*sizeof(double));
    double* u = (double*)malloc((2*N-1)*N*sizeof(double)); 
    double* max_residue = (double*)malloc((2*N)*N*sizeof(double)); 
    double* arr = (double*)malloc((2*N)*N*sizeof(double)); 
    int i, j, lambda = 100, lambda_squared = lambda*lambda, num_iter = 0, colour; 
    double constant = 10*lambda/sqrt(M_PI), dx = 2./(N-1), max;

    int *dev_N;
    double *dev_omega;
    double *dev_dx;
    double *dev_constant;
    int *dev_lambda_squared;
    double* dev_g;
    double* dev_u;
    double* dev_max_residue;
    int* dev_levels;
    int* dev_colour;

    cudaMalloc((void**)&dev_N, sizeof(int));
    cudaMalloc((void**)&dev_colour, sizeof(int));
    cudaMalloc((void**)&dev_levels, sizeof(int));
    cudaMalloc((void**)&dev_omega, sizeof(double));
    cudaMalloc((void**)&dev_dx, sizeof(double));
    cudaMalloc((void**)&dev_constant, sizeof(double));
    cudaMalloc((void**)&dev_lambda_squared, sizeof(int));
    cudaMalloc((void**)&dev_g, (2*N-1)*N*sizeof(double));
    cudaMalloc((void**)&dev_u, (2*N-1)*N*sizeof(double));
    cudaMalloc((void**)&dev_max_residue, (2*N)*N*sizeof(double));

    cudaMemcpy(dev_N, &N, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_omega, &omega, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_dx, &dx, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_constant, &constant, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_lambda_squared, &lambda_squared, sizeof(int), cudaMemcpyHostToDevice);


    // calculate g, initialize u and max residue
    caculate_forcing_term<<<N,(2*N-1)>>>(dev_N, dev_dx, dev_lambda_squared, dev_constant, dev_g, dev_u, dev_max_residue);
    cudaMemcpy(max_residue, dev_max_residue, 2*N*N*sizeof(double), cudaMemcpyDeviceToHost);

    // main for loop
    arr[0] = 1.0;
    while(arr[0] > tolerance){
        
        num_iter += 1;
        if(num_iter > iterations){
            break;
        }

        // update one iteration of points
        colour = 0;
        cudaMemcpy(dev_colour, &colour, sizeof(int), cudaMemcpyHostToDevice);
        point_updates<<<N,(2*N-1),(6*N-3)*sizeof(double)>>>(dev_N, dev_omega, dev_dx, dev_g, dev_u, dev_max_residue, dev_colour);
        colour = 1;
        cudaMemcpy(dev_colour, &colour, sizeof(int), cudaMemcpyHostToDevice);
        point_updates<<<N,(2*N-1),(6*N-3)*sizeof(double)>>>(dev_N, dev_omega, dev_dx, dev_g, dev_u, dev_max_residue, dev_colour);

        // find maximum residue
        cudaMemcpy(max_residue, dev_max_residue, 2*N*N*sizeof(double), cudaMemcpyDeviceToHost);
        max = max_residue[0];
        for(i = 0; i < N; ++i){
            for(j = 0; j < 2*N-1; ++j){
                if(max_residue[i*(2*N-1) + j] > max){
                    max = max_residue[i*(2*N-1) + j];
                }
            }
        }
        arr = reduction_call(1,2*N*N,max_residue);
        for(int step = 2; step < ceil(log2(2*N*N)/5) + 1; ++step){
            arr = reduction_call(step,2*N*N,arr);
        }
    }

    // write to file (transpose for matlab output)
    cudaMemcpy(u, dev_u, (2*N-1)*N*sizeof(double), cudaMemcpyDeviceToHost);
    // double (*u_actual)[N] = (double(*)[N])malloc((2*N-1)*sizeof(*u_actual));
    // for(i = 0; i < 2*N - 1; ++i){
    //         for(j = 0; j < N; ++j){
    //             u_actual[i][j] = u[j*(2*N-1) + i];
    //         }
    //     }
    // fwrite(&(u_actual[0][0]), sizeof(double), (2*N-1)*N, fileid);

    cudaFree(dev_N);
    cudaFree(dev_dx);
    cudaFree(dev_constant);
    cudaFree(dev_lambda_squared);
    cudaFree(dev_g);
    free(g);

    printf("Elapsed time: %g\n", (float)(clock()-start)/CLOCKS_PER_SEC); 

return 0;
}

