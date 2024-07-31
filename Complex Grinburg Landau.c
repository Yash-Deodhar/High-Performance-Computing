/*  Complex Grinsburg Landau Equation
    Program written by Yash Deodhar
*/

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<fftw3.h>


//define a function that takes a source array as an input and stores its Laplacian in the target array using plans p and pinv
void Laplacian(int N, fftw_complex *source, fftw_complex *target, fftw_plan p, fftw_plan pinv)
{
  fftw_execute(p);
  for (int i = 0; i < N; ++i)
    {
      for (int j = 0; j < N; ++j)
	      {
          if(i <= N/2)
            {
              if(j <= N/2)
                {
                 target[i*N +j][0] *= -(i*i + j*j);
                 target[i*N +j][1] *= -(i*i + j*j);                 
                }
              else
                {
                 target[i*N +j][0] *= -(i*i + (N-j)*(N-j));
                 target[i*N +j][1] *= -(i*i + (N-j)*(N-j));                  
                }
            }
          else
            {
              if(j <= N/2)
                {
                  target[i*N +j][0] *= -((N-i)*(N-i) + j*j);
                  target[i*N +j][1] *= -((N-i)*(N-i) + j*j);
                }
              else
                {
                  target[i*N +j][0] *= -((N-i)*(N-i) + (N-j)*(N-j));
                  target[i*N +j][1] *= -((N-i)*(N-i) + (N-j)*(N-j));
                }
            }  
        }
    }
  fftw_execute(pinv);
}


//function to store the multiplication of two complex numbers. Useful for making the Runge Kutta code easier to code 
void complex_mul(fftw_complex z1, fftw_complex z2, fftw_complex z3)
{
  z3[0] = z1[0]*z2[0] - z1[1]*z2[1];
  z3[1] = z1[1]*z2[0] + z1[0]*z2[1];
}

int main(int argc, char* argv[])
{   
    //start timer
    clock_t start = clock();
    
    //take inputs, print them out and set seed
    int argi = 0;
    int i, j, k;
    int N = atoi(argv[++argi]);
    float c1 = atof(argv[++argi]), c3 = atof(argv[++argi]);
    int M = atoi(argv[++argi]);
    long int seed;
    if (argi < argc-1)
      seed = atol(argv[++argi]);
    else 
      seed = (long int)time(NULL);
    srand48(seed);
    printf("N = %d\t c1 = %f\t c2 = %f\t M = %d\t seed = %ld\n",N,c1,c3,M,seed);

    // open file
    FILE* fileid = fopen("CGL.out","w");

    //define important quantities and constants used in each iteration of Runge-Kutta loop
    int total_time = 10000;
    double dt = (float)total_time/M;
    fftw_complex k1 = {(1./64)*(1./64), c1*(1./64)*(1./64)}, k2 = {(1./64)*(1./64) , -c3*(1./64)*(1./64)};
    double abs = 0, dt1 = dt/4, dt2 = dt/3, dt3 = dt/2;
    fftw_complex term1, term2;

    // define arrays and plans
    fftw_complex (*A)[N] = fftw_malloc(N*sizeof(*A));
    fftw_complex (*A1)[N] = fftw_malloc(N*sizeof(*A1));
    fftw_complex (*LapA)[N] = fftw_malloc(N*sizeof(*LapA));
    fftw_plan p, p2, pinv;
    p = fftw_plan_dft_2d(N, N, &(A[0][0]), &(LapA[0][0]), FFTW_FORWARD, FFTW_ESTIMATE);
    p2 = fftw_plan_dft_2d(N, N, &(A1[0][0]), &(LapA[0][0]), FFTW_FORWARD, FFTW_ESTIMATE);
    pinv = fftw_plan_dft_2d(N, N, &(LapA[0][0]), &(LapA[0][0]), FFTW_BACKWARD, FFTW_ESTIMATE);
    
    //initialize grid
    for (i = 0; i < N; ++i)
      {
	      for (j = 0; j < N; ++j)
	        {
	          A[i][j][0] = 3*drand48() - 1.5;
	          A[i][j][1] = 3*drand48() - 1.5;
	        }
      }

    //write inital grid
    fwrite(A, sizeof(double), N*N*2, fileid);  

    //Runge-Kutta
    //outer loop for moving through time and inner loops for iterating over the arrays
    for(k = 0; k < M; ++k)
      {
        //step 1
        Laplacian(N, &(A[0][0]), &(LapA[0][0]), p, pinv);
        for(i = 0; i < N; ++i)
          {
            for(j = 0; j < N; ++j)
              {
                abs = A[i][j][0]*A[i][j][0] + A[i][j][1]*A[i][j][1];
                complex_mul(k1, LapA[i][j], term1);
                complex_mul(k2, A[i][j], term2);
                A1[i][j][0] = A[i][j][0] + dt1*(A[i][j][0] + term1[0]/N/N - abs*term2[0]);
                A1[i][j][1] = A[i][j][1] + dt1*(A[i][j][1] + term1[1]/N/N - abs*term2[1]);
              }
          }

        //step 2
        Laplacian(N, &(A1[0][0]), &(LapA[0][0]), p2, pinv);
        for(i = 0; i < N; ++i)
          {
            for(j = 0; j < N; ++j)
              {
                abs = A1[i][j][0]*A1[i][j][0] + A1[i][j][1]*A1[i][j][1];    
                complex_mul(k1, LapA[i][j], term1);
                complex_mul(k2, A1[i][j], term2);
                A1[i][j][0] = A1[i][j][0] + dt2*(A1[i][j][0] + term1[0]/N/N - abs*term2[0]);
                A1[i][j][1] = A1[i][j][1] + dt2*(A1[i][j][1] + term1[1]/N/N - abs*term2[1]);
              }
          }

        //step 3
        Laplacian(N, &(A1[0][0]), &(LapA[0][0]), p2, pinv);
        for(i = 0; i < N; ++i)
          {
            for(j = 0; j < N; ++j)
              {
                abs = A1[i][j][0]*A1[i][j][0] + A1[i][j][1]*A1[i][j][1];  
                complex_mul(k1, LapA[i][j], term1);
                complex_mul(k2, A1[i][j], term2);
                A1[i][j][0] = A1[i][j][0] + dt3*(A1[i][j][0] + term1[0]/N/N - abs*term2[0]);
                A1[i][j][1] = A1[i][j][1] + dt3*(A1[i][j][1] + term1[1]/N/N - abs*term2[1]);
              }
          }

        //step 4
        Laplacian(N, &(A1[0][0]), &(LapA[0][0]), p2, pinv);
        for(i = 0; i < N; ++i)
          {
            for(j = 0; j < N; ++j)
              {
                abs = A1[i][j][0]*A1[i][j][0] + A1[i][j][1]*A1[i][j][1];    
                complex_mul(k1, LapA[i][j], term1);
                complex_mul(k2, A1[i][j], term2);
                A[i][j][0] = A1[i][j][0] + dt*(A1[i][j][0] + term1[0]/N/N - abs*term2[0]);
                A[i][j][1] = A1[i][j][1] + dt*(A1[i][j][1] + term1[1]/N/N - abs*term2[1]);
              }
          }
          if((k+1)%(M/10) == 0)
          {
            fwrite(A, sizeof(double), N*N*2, fileid);  
            printf("%d%% completed\n", 100*(k+1)/M);
          } 
      }
    
    //destroy plans and free arrays
    fftw_destroy_plan(p);
    fftw_destroy_plan(p2);
    fftw_destroy_plan(pinv);
    fftw_free(A);
    fftw_free(A1);
    fftw_free(LapA);
    fclose(fileid);

    //print time elapsed
    printf("Elapsed time: %g\n", (float)(clock()-start)/CLOCKS_PER_SEC); 
    return 0;
}
