#define _XOPEN_SOURCE
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<fftw3-mpi.h>
#include<mpi.h>

void Laplacian(int N, int localn, int local0, fftw_complex *source, fftw_complex *target, fftw_plan p, fftw_plan pinv)
{
  double x;
  fftw_execute(p);
  for (int i = 0; i < localn; ++i)
    {
      x = local0 + i;
      for (int j = 0; j < N; ++j)
	      {
          if(x <= N/2)
            {
              if(j <= N/2)
                {
                 target[i*N +j][0] *= -(x*x + j*j);
                 target[i*N +j][1] *= -(x*x + j*j);                 
                }
              else
                {
                 target[i*N +j][0] *= -(x*x + (N-j)*(N-j));
                 target[i*N +j][1] *= -(x*x + (N-j)*(N-j));                  
                }
            }
          else
            {
              if(j <= N/2)
                {
                  target[i*N +j][0] *= -((N-x)*(N-x) + j*j);
                  target[i*N +j][1] *= -((N-x)*(N-x) + j*j);
                }
              else
                {
                  target[i*N +j][0] *= -((N-x)*(N-x) + (N-j)*(N-j));
                  target[i*N +j][1] *= -((N-x)*(N-x) + (N-j)*(N-j));
                }
            }  
        }
    }
  fftw_execute(pinv);
}

void complex_mul(fftw_complex z1, fftw_complex z2, fftw_complex z3)
{
  z3[0] = z1[0]*z2[0] - z1[1]*z2[1];
  z3[1] = z1[1]*z2[0] + z1[0]*z2[1];
}

int main(int argc, char* argv[])
{   
    //initialze MPI
    MPI_Init(&argc, &argv);
    fftw_mpi_init();
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    double precision = MPI_Wtick();
    double starttime = MPI_Wtime();

    
    //take inputs, print them out and set seed
    int argi = 0;
    int i, j, k;
    const ptrdiff_t N = atoi(argv[++argi]);;
    float c1 = atof(argv[++argi]), c3 = atof(argv[++argi]);
    int M = atoi(argv[++argi]);
    long int input_seed;
    if (argi < argc-1)
      input_seed = atol(argv[++argi]);
    else 
      input_seed = (long int)time(NULL);
    srand48(input_seed + rank);

    // open file
    FILE* fileid = fopen("CGL.out","w");

    // fftw stuff
    ptrdiff_t localn, local0;
    ptrdiff_t alloc_local = fftw_mpi_local_size_2d(N, N, MPI_COMM_WORLD,  &localn, &local0);
    fftw_complex* A = fftw_alloc_complex(alloc_local);
    fftw_complex* A1 = fftw_alloc_complex(alloc_local);
    fftw_complex* LapA = fftw_alloc_complex(alloc_local);
    fftw_complex (*final_A)[N] = fftw_malloc(N*sizeof(*final_A));
    fftw_plan pf, pf1, pb;
    pf = fftw_mpi_plan_dft_2d(N, N, A, LapA, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_ESTIMATE);
    pf1 = fftw_mpi_plan_dft_2d(N, N, A1, LapA, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_ESTIMATE); 
    pb = fftw_mpi_plan_dft_2d(N, N, LapA, LapA, MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_ESTIMATE);
    
    //initialize grid
    double dx = 2*M_PI/N;
    for (i = 0; i < localn; ++i)
      {
        double x = (local0+i);
	      for (j = 0; j < N; ++j)
	        {
	          A[i*N + j][0] = 3*drand48() - 1.5;
	          A[i*N + j][1] = 3*drand48() - 1.5;
	        }
      }
    MPI_Gather(A, N*localn*2, MPI_DOUBLE, final_A, N*localn*2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if(rank == 0){
            fwrite(final_A, sizeof(double), N*N*2, fileid);
    }

    //define important quantities and constants used in each iteration of Runge-Kutta loop
    int total_time = 10000;
    double dt = (double)total_time/M;
    fftw_complex k1 = {(1./64)*(1./64), c1*(1./64)*(1./64)}, k2 = {1. , -c3};
    double abs = 0, dt1 = dt/4, dt2 = dt/3, dt3 = dt/2;
    fftw_complex term1, term2;

    // runge kutta
    for(k = 0; k < M; ++k)
      {
        //step 1
        Laplacian(N, localn, local0, A, LapA, pf, pb);
        for(i = 0; i < localn; ++i)
          {
            for(j = 0; j < N; ++j)
              {
                abs = A[i*N + j][0]*A[i*N + j][0] + A[i*N + j][1]*A[i*N + j][1];
                complex_mul(k1, LapA[i*N + j], term1);
                complex_mul(k2, A[i*N + j], term2);
                A1[i*N + j][0] = A[i*N + j][0] + dt1*(A[i*N + j][0] + term1[0]/N/N - abs*term2[0]);
                A1[i*N + j][1] = A[i*N + j][1] + dt1*(A[i*N + j][1] + term1[1]/N/N - abs*term2[1]);
              }
          }
        
        //step 2
        Laplacian(N, localn, local0, A1, LapA, pf1, pb);
          for(i = 0; i < localn; ++i)
            {
              for(j = 0; j < N; ++j)
                {
                  abs = A[i*N + j][0]*A[i*N + j][0] + A[i*N + j][1]*A[i*N + j][1];
                  complex_mul(k1, LapA[i*N + j], term1);
                  complex_mul(k2, A[i*N + j], term2);
                  A1[i*N + j][0] = A[i*N + j][0] + dt2*(A[i*N + j][0] + term1[0]/N/N - abs*term2[0]);
                  A1[i*N + j][1] = A[i*N + j][1] + dt2*(A[i*N + j][1] + term1[1]/N/N - abs*term2[1]);
                }
            }
        
        //step 3
        Laplacian(N, localn, local0, A1, LapA, pf1, pb);
          for(i = 0; i < localn; ++i)
            {
              for(j = 0; j < N; ++j)
                {
                  abs = A[i*N + j][0]*A[i*N + j][0] + A[i*N + j][1]*A[i*N + j][1];
                  complex_mul(k1, LapA[i*N + j], term1);
                  complex_mul(k2, A[i*N + j], term2);
                  A1[i*N + j][0] = A[i*N + j][0] + dt3*(A[i*N + j][0] + term1[0]/N/N - abs*term2[0]);
                  A1[i*N + j][1] = A[i*N + j][1] + dt3*(A[i*N + j][1] + term1[1]/N/N - abs*term2[1]);
                }
            }

        //step 4
        Laplacian(N, localn, local0, A1, LapA, pf1, pb);
          for(i = 0; i < localn; ++i)
            {
              for(j = 0; j < N; ++j)
                {
                  abs = A[i*N + j][0]*A[i*N + j][0] + A[i*N + j][1]*A[i*N + j][1];
                  complex_mul(k1, LapA[i*N + j], term1);
                  complex_mul(k2, A[i*N + j], term2);
                  A[i*N + j][0] = A[i*N + j][0] + dt*(A[i*N + j][0] + term1[0]/N/N - abs*term2[0]);
                  A[i*N + j][1] = A[i*N + j][1] + dt*(A[i*N + j][1] + term1[1]/N/N - abs*term2[1]);
                }
            }

        if((k+1)%(M/10) == 0)
          {
            MPI_Gather(A, N*localn*2, MPI_DOUBLE, final_A, N*localn*2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            if(rank == 0){
            fwrite(final_A, sizeof(double), N*N*2, fileid);
            printf("%d%% completed\n", 100*(k+1)/M);
            }    
          }
          MPI_Barrier(MPI_COMM_WORLD);
      }
  double elapsedtime = MPI_Wtime()-starttime;
  if (rank == 0) {
  printf("Execution time = %le seconds\n", elapsedtime);
  printf("Precision of timing is %le seconds\n", precision);
  }
  fclose(fileid);
  MPI_Finalize();  
}