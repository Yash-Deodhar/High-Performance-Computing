#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <curand.h>
#include <cusolverDn.h>

__global__ void transpose(double* arr){
  double temp;
  temp = arr[blockIdx.x + threadIdx.x * blockDim.x];
  __syncthreads();
  arr[threadIdx.x + blockIdx.x * blockDim.x] = temp;
}

__global__ void set_LHS(double* dev_LHS, double* dev_D_Op){
  int g_i = threadIdx.x + blockIdx.x * blockDim.x;
  int target = blockIdx.x + threadIdx.x * blockDim.x;
  double identity_matrix = 0.0;
  if(threadIdx.x == blockIdx.x){
    identity_matrix = 1.0;
  }
  dev_LHS[target] = identity_matrix - dev_D_Op[g_i];
}

__global__ void step_1(double* dev_U, double* dev_V, double* dev_W, double* alpha, int* N, double* dt, double* dx){
  int g_i = threadIdx.x + blockIdx.x * blockDim.x;
  int l_i = threadIdx.x;

  __shared__ double localu[1024];
  __shared__ double localv[1024];
  __shared__ double localw[1024];

  localu[l_i] = dev_U[g_i];
  localv[l_i] = dev_V[g_i];
  localw[l_i] = dev_W[g_i];

  __syncthreads();

  double RhoU = localu[l_i]*(1 - localu[l_i] - *alpha*localw[l_i]);
  double RhoV = localv[l_i]*(1 - localv[l_i] - *alpha*localu[l_i]);
  double RhoW = localw[l_i]*(1 - localw[l_i] - *alpha*localv[l_i]);

  if(blockIdx.x == 0){
    dev_U[g_i] += (*dt/2)*(-2*dev_U[g_i] + 2*dev_U[g_i+(*N)])/(*dx)/(*dx) + (*dt/2)*(RhoU);
    dev_V[g_i] += (*dt/2)*(-2*dev_V[g_i] + 2*dev_V[g_i+(*N)])/(*dx)/(*dx) + (*dt/2)*(RhoV);
    dev_W[g_i] += (*dt/2)*(-2*dev_W[g_i] + 2*dev_W[g_i+(*N)])/(*dx)/(*dx) + (*dt/2)*(RhoW);
  }
  else if(blockIdx.x == *N-1){
    dev_U[g_i] += (*dt/2)*(-2*dev_U[g_i] + 2*dev_U[g_i-(*N)])/(*dx)/(*dx) + (*dt/2)*(RhoU);
    dev_V[g_i] += (*dt/2)*(-2*dev_V[g_i] + 2*dev_V[g_i-(*N)])/(*dx)/(*dx) + (*dt/2)*(RhoV);
    dev_W[g_i] += (*dt/2)*(-2*dev_W[g_i] + 2*dev_W[g_i-(*N)])/(*dx)/(*dx) + (*dt/2)*(RhoW);
  }
  else{
    dev_U[g_i] += (*dt/2)*(-2*dev_U[g_i] + dev_U[g_i-(*N)] + dev_U[g_i+(*N)])/(*dx)/(*dx) + (*dt/2)*(RhoU);
    dev_V[g_i] += (*dt/2)*(-2*dev_V[g_i] + dev_V[g_i-(*N)] + dev_V[g_i+(*N)])/(*dx)/(*dx) + (*dt/2)*(RhoV);
    dev_W[g_i] += (*dt/2)*(-2*dev_W[g_i] + dev_W[g_i-(*N)] + dev_W[g_i+(*N)])/(*dx)/(*dx) + (*dt/2)*(RhoW);
  }
}

__global__ void step_3(double* dev_U, double* dev_V, double* dev_W, double* alpha, int* N, double* dt, double* dx){
  int g_i = threadIdx.x + blockIdx.x * blockDim.x;
  int l_i = threadIdx.x;

  __shared__ double localu[1024];
  __shared__ double localv[1024];
  __shared__ double localw[1024];

  localu[l_i] = dev_U[g_i];
  localv[l_i] = dev_V[g_i];
  localw[l_i] = dev_W[g_i];

  __syncthreads();

  double RhoU = localu[l_i]*(1 - localu[l_i] - *alpha*localw[l_i]);
  double RhoV = localv[l_i]*(1 - localv[l_i] - *alpha*localu[l_i]);
  double RhoW = localw[l_i]*(1 - localw[l_i] - *alpha*localv[l_i]);

  if(threadIdx.x == 0){
    dev_U[g_i] = localu[l_i] + (*dt/2)*(-2*localu[l_i] + 2*localu[l_i+1])/(*dx)/(*dx) + (*dt/2)*(RhoU);
    dev_V[g_i] = localv[l_i] + (*dt/2)*(-2*localv[l_i] + 2*localv[l_i+1])/(*dx)/(*dx) + (*dt/2)*(RhoV);
    dev_W[g_i] = localw[l_i] + (*dt/2)*(-2*localw[l_i] + 2*localw[l_i+1])/(*dx)/(*dx) + (*dt/2)*(RhoW);
  }
  else if(threadIdx.x == *N-1){
    dev_U[g_i] = localu[l_i] + (*dt/2)*(-2*localu[l_i] + 2*localu[l_i-1])/(*dx)/(*dx) + (*dt/2)*(RhoU);
    dev_V[g_i] = localv[l_i] + (*dt/2)*(-2*localv[l_i] + 2*localv[l_i-1])/(*dx)/(*dx) + (*dt/2)*(RhoV);
    dev_W[g_i] = localw[l_i] + (*dt/2)*(-2*localw[l_i] + 2*localw[l_i-1])/(*dx)/(*dx) + (*dt/2)*(RhoW);
  }
  else{
    dev_U[g_i] = localu[l_i] + (*dt/2)*(-2*localu[l_i] + localu[l_i-1] + localu[l_i+1])/(*dx)/(*dx) + (*dt/2)*(RhoU);
    dev_V[g_i] = localv[l_i] + (*dt/2)*(-2*localv[l_i] + localv[l_i-1] + localv[l_i+1])/(*dx)/(*dx) + (*dt/2)*(RhoV);
    dev_W[g_i] = localw[l_i] + (*dt/2)*(-2*localw[l_i] + localw[l_i-1] + localw[l_i+1])/(*dx)/(*dx) + (*dt/2)*(RhoW);
  }
}

__global__ void scale_input(double* dev_UVW, double* target, double* alpha){
  int g_i = threadIdx.x + blockIdx.x * blockDim.x;
  double scale = 1./(1 + *alpha);
  target[g_i] = scale*dev_UVW[g_i];
}

int main(int argc, char* argv[]) {

    // start clock
    clock_t start = clock();

    // take inputs and open files
    int argi = 0;
    int N = atoi(argv[++argi]);
    double alpha  = atof(argv[++argi]);
    int M = atoi(argv[++argi]);
    long int seed;
    double dt = 200./M, dx = 120./(N-1);
    if (argi < argc-1)
      seed = atol(argv[++argi]);
    else 
      seed = (long int)time(NULL);
    printf("N: %d\talpha: %f\tM: %d\tseed: %ld\n", N, alpha, M, seed);
    printf("dt: %f\tdx: %f\n", dt, dx);
    FILE *fileidu = fopen("RPSU.out", "w");
    FILE *fileidv = fopen("RPSV.out", "w");
    FILE *fileidw = fopen("RPSW.out", "w");

    // initialze U, V and W
    double *U = (double*)malloc(N*N*sizeof(double));
    double *V = (double*)malloc(N*N*sizeof(double));
    double *W = (double*)malloc(N*N*sizeof(double));

    double *dev_UVW;
    double *dev_U;
    double *dev_V;
    double *dev_W;
    double *dev_alpha;
    double *dev_dt;
    double *dev_dx;
    int *dev_N;
    
    cudaMalloc((void**)&dev_UVW, 3*N*N*sizeof(double));
    cudaMalloc((void**)&dev_U, N*N*sizeof(double));
    cudaMalloc((void**)&dev_V, N*N*sizeof(double));
    cudaMalloc((void**)&dev_W, N*N*sizeof(double));
    cudaMalloc((void**)&dev_alpha, sizeof(double));
    cudaMalloc((void**)&dev_N, sizeof(int));
    cudaMalloc((void**)&dev_dx, sizeof(double));
    cudaMalloc((void**)&dev_dt, sizeof(double));

    cudaMemcpy(dev_N, &N, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_alpha, &alpha, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_dt, &dt, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_dx, &dx, sizeof(double), cudaMemcpyHostToDevice);

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    curandGenerateUniformDouble(gen, dev_UVW, 3*N*N);

    scale_input<<<N,N>>>(dev_UVW, dev_U, dev_alpha);
    scale_input<<<N,N>>>(&dev_UVW[N*N], dev_V, dev_alpha);
    scale_input<<<N,N>>>(&dev_UVW[2*N*N], dev_W, dev_alpha);
    cudaMemcpy(U, dev_U, N*N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(V, dev_V, N*N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(W, dev_W, N*N*sizeof(double), cudaMemcpyDeviceToHost);  
    fwrite(U, sizeof(double), N*N, fileidu);
    fwrite(V, sizeof(double), N*N, fileidv);
    fwrite(W, sizeof(double), N*N, fileidw);

    double *LHS = (double*)malloc(N*N*sizeof(double));
    double *D_Op = (double*)malloc(N*N*sizeof(double));
    double *dev_LHS;
    double *dev_D_Op;
    cudaMalloc((void**)&dev_D_Op, N*N*sizeof(double)) ;
    cudaMalloc((void**)&dev_LHS, N*N*sizeof(double)) ;
    memset(D_Op, 0, N*N*sizeof(double));
    for(int i = 1; i < N - 1; i++){
      D_Op[i*(N+1)] = -(dt/2)*2./(dx*dx);
      D_Op[i*(N+1)+1] = (dt/2)*1./(dx*dx);
      D_Op[i*(N+1)-1] = (dt/2)*1./(dx*dx);
    }
    D_Op[0] = -(dt/2)*2./(dx*dx);
    D_Op[1] = (dt/2)*2./(dx*dx);
    D_Op[N*N-2] = (dt/2)*2./(dx*dx);
    D_Op[N*N-1] = -(dt/2)*2./(dx*dx);
    cudaMemcpy(dev_D_Op, D_Op, N*N*sizeof(double), cudaMemcpyHostToDevice);
    set_LHS<<<N,N>>>(dev_LHS, dev_D_Op);
    
    cusolverDnHandle_t cusolverH = NULL;
    int *dev_info = NULL;
    double *dev_work  = NULL;
    int *dev_pivot = NULL;
    cudaMalloc((void**)&dev_pivot, N*sizeof(int)) ;
    cudaMalloc((void**)&dev_info, sizeof(int));
    cudaStream_t stream;
    cusolverDnCreate(&cusolverH);
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cusolverDnSetStream(cusolverH, stream);
    int lwork = 0;
    cusolverDnDgetrf_bufferSize(cusolverH, N, N, dev_LHS , N, &lwork);
    cudaMalloc((void**)&dev_work, lwork*sizeof(double));
    cusolverDnDgetrf(cusolverH, N, N, dev_LHS, N, dev_work, dev_pivot, dev_info);

    // main for loop
    for(int k = 0; k < M; ++k){
      

      // step 1 
      step_1<<<N,N>>>(dev_U, dev_V, dev_W, dev_alpha, dev_N, dev_dt, dev_dx); 
      

      // step 2
      cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, N, N, dev_LHS, N, dev_pivot, dev_U, N, dev_info);
      cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, N, N, dev_LHS, N, dev_pivot, dev_V, N, dev_info);
      cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, N, N, dev_LHS, N, dev_pivot, dev_W, N, dev_info);
      cudaDeviceSynchronize();

      // step 3
      step_3<<<N,N>>>(dev_U, dev_V, dev_W, dev_alpha, dev_N, dev_dt, dev_dx);
      cudaDeviceSynchronize();

      // step 4
      transpose<<<N,N>>>(dev_U);
      transpose<<<N,N>>>(dev_V);
      transpose<<<N,N>>>(dev_W);
      cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, N, N, dev_LHS, N, dev_pivot, dev_U, N, dev_info);
      cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, N, N, dev_LHS, N, dev_pivot, dev_V, N, dev_info);
      cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, N, N, dev_LHS, N, dev_pivot, dev_W, N, dev_info);
      cudaDeviceSynchronize();
      transpose<<<N,N>>>(dev_U);
      transpose<<<N,N>>>(dev_V);
      transpose<<<N,N>>>(dev_W);

      if((k+1)%(M/10) == 0){
        cudaMemcpy(U, dev_U, N*N*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(V, dev_V, N*N*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(W, dev_W, N*N*sizeof(double), cudaMemcpyDeviceToHost);  
        fwrite(U, sizeof(double), N*N, fileidu);
        fwrite(V, sizeof(double), N*N, fileidv);
        fwrite(W, sizeof(double), N*N, fileidw);
        printf("%d%% done\n", 100*(k+1)/M);
      } 
    }

    printf("Elapsed time: %g\n", (float)(clock()-start)/CLOCKS_PER_SEC); 

    return 0;
}