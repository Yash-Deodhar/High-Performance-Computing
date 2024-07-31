/*  Van Der Pol problem
    Program written by Yash Deodhar
*/


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Define Function RandomNormal to generate samples from a normal distributition using the Polar Marsaglia Method:
double RandomNormal(void)
{
  static int i = 0;
  static double temp_var[2] = {0,0};
  if (i == 1)
    {
      --i;
      return temp_var[1];      
    }
  double U1, U2, W, common_term;
  do
    {
      U1 = drand48();
      U2 = drand48();
      // for some reason defining V1 = 2*U1-1 and V2 = 2*U2-1 and then using those variables gives the a marginally higher total time
      W = (2*U1-1)*(2*U1-1) + (2*U2-1)*(2*U2-1);
    }while(W >= 1.);
  common_term = sqrt(-2*log(W)/W);
  temp_var[0] = common_term*(2*U1-1);
  temp_var[1] = common_term*(2*U2-1);
  ++i;
  return temp_var[0];
}


int main(int argc, char* argv[])
{
    //start timer
    clock_t start = clock();
    
    // Open file Prob.out to write to
    FILE* fileid = fopen("Prob.out","w");
    
    // accept input arguments including the optional argument of the seed 
    int argi = 0;
    double alpha = atof(argv[++argi]), sigma = atof(argv[++argi]);
    int M = atoi(argv[++argi]), N = atoi(argv[++argi]);
    long int seed;
    if (argi < argc-1)
      seed = atol(argv[++argi]);
    else 
      seed = (long int)time(NULL);
    
    // set seed for srand48()
    srand48(seed);
    
    // define time step size dt and value to be incremented while calculating the probability of reaching an equilibrium point
    // Important note: N should not be extremely large or 'inc' may become 0 due to machine precision 
    double dt = 10./M, inc = 1./N;

    // allocate memory for p
    double* p = (double*)malloc((int)(M/10 + 1)*sizeof(double));

    // print the input values
    printf("alpha: %f\tsigma: %f\ttime steps: %d\tIterations: %d\tseed: %ld\n",alpha,sigma,M,N,seed);

    //define constants used in every iteration
    double lim = alpha*alpha/4, term1 = alpha*alpha, term2 = sigma*sqrt(dt), X_old;
    
    for(int i = 0; i < N; ++i)
      {
	      // define X0 and Y0 and calculate p[0]
	      double X = 0.1*RandomNormal(), Y = 0;
        // alternate method: p[0] += (((X - alpha)*(X - alpha)) <= lim)? inc : 0 + (((X + alpha)*(X + alpha)) <= lim)? inc : 0;
	      if (((X - alpha)*(X - alpha)) <= lim)
	        p[0] += inc;
	      if (((X + alpha)*(X + alpha)) <= lim)
	        p[0] += inc;

	      // update X and Y according to given stochastic differential equations 
	      for(int j = 1; j < M; ++j)
	        {
            X_old = X;
            X += Y*dt;
            Y += ((term1 - X_old*X_old)*X_old - Y)*dt + term2*X_old*RandomNormal();

	          // update probability at 100 equidistant points in time
	          if ((j+1)%(M/100) == 0)
	            {
                // alternate method: p[(j+1)/10] += (((X - alpha)*(X - alpha) + Y*Y) <= lim)? inc : 0 + (((X + alpha)*(X + alpha) + Y*Y) <= lim)? inc : 0;
		            if (((X - alpha)*(X - alpha) + Y*Y) <= lim)
		              p[(j+1)/10] += inc;
		            if (((X + alpha)*(X + alpha) + Y*Y) <= lim)
		              p[(j+1)/10] += inc;
	            }
	        }
      }
    fwrite(p, sizeof(double), 101, fileid);
    fclose(fileid);
    free(p);
    printf("Time required: %g\n",(float)(clock()-start)/CLOCKS_PER_SEC);
    return 0;
} 
