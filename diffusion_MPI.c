#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<mpi.h>
#include<time.h>

int main(int argc, char* argv[])
{
    // initialize MPI
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    double precision = MPI_Wtick();
    double starttime = MPI_Wtime();
    
    // take inputs and open file
    int argi = 0;
    int N = atoi(argv[++argi]);
    double omega  = atof(argv[++argi]), tolerance = atof(argv[++argi]);
    int iterations = atoi(argv[++argi]);
    FILE *fileid = fopen("Sources.out", "w");
    
    // allocate memory for grid and RHS of equation  
    double (*u0)[2*N-1] = malloc(N*sizeof(*u0));
    double (*g)[2*N-1] = malloc(N*sizeof(*g));

    // define important quantities, initialize u and calculate g
    double dx = 2./(N-1), term1, term2;
    double max_residue = 1.+tolerance, residue;
    int i, j, lambda = 100, num_iter = 0, index;
    double constant = 10*lambda/sqrt(M_PI);
    for(i = 0; i < N; ++i){
        for(j = 0; j < 2*N-1; ++j){
            term1 = exp(-lambda*lambda*((j*dx-3)*(j*dx-3)+(i*dx-1)*(i*dx-1)));
            term2 = exp(-lambda*lambda*((j*dx-1)*(j*dx-1)+(i*dx-1)*(i*dx-1)));
            g[i][j] = constant*(term1-term2);
            u0[i][j] = 0;
        }
    }

    // send data to all cores
    int grid_size_per_block = (2*N-1)*(N/size + 2), rank_reccount, num_rows = N/size + 2;
    int (*srccount) = malloc(size*sizeof(*srccount));
    int (*offsets) = malloc(size*sizeof(*offsets));
    if(rank == 0 || rank == size - 1){
        num_rows = N/size + 1;
    }
    if(size == 1){
        num_rows = N;
    }
    double (*g1)[2*N-1] = malloc((num_rows)*sizeof(*g1));
    double (*u1)[2*N-1] = malloc((num_rows)*sizeof(*u1));
    if(size == 1){
        for(i = 0; i < N; ++i){
            for(j = 0; j < 2*N-1; ++j){
                    g1[i][j] = g[i][j];
                    u1[i][j] = u0[i][j];
            }
        }
    }
    if(size > 1){
    offsets[0] = 0;
    for(i = 0; i < size; ++i){
        if(i == size-1 || i == 0){
            srccount[i] = grid_size_per_block - 2*N + 1;
        }
        else{
            srccount[i] = grid_size_per_block;
        }
        if(i > 0){
            offsets[i] = offsets[i-1] + srccount[i-1] - 4*N + 2;
        }
        if(rank == i){
            rank_reccount = srccount[i];
        }
    }
    MPI_Scatterv(&(g[0][0]), srccount, offsets, MPI_DOUBLE, &(g1[0][0]), rank_reccount, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(&(u0[0][0]), srccount, offsets, MPI_DOUBLE, &(u1[0][0]), rank_reccount, MPI_DOUBLE, 1, MPI_COMM_WORLD);
    }
    while(max_residue > tolerance){
        num_iter += 1;
        if(num_iter > iterations){
            break;
        }
        max_residue = 0.;
        // update red points
        j=0;
        for(i = 1; i < num_rows - 1; ++i){
            if((i+j)%2 != 0){
                residue = 0.25*(u1[i+1][j] + u1[i-1][j] + u1[i][j+1] - 3*u1[i][j] - dx*dx*g1[i][j]);
                max_residue = fabs(residue) > max_residue ?  fabs(residue) : max_residue;
                u1[i][j] += omega*residue;
            } 
        }

        for(i = 1; i < num_rows - 1; ++i){
            for(j = 1; j < 2*N - 2; ++j){
                if((i+j)%2 != 0){
                    residue = 0.25*(u1[i+1][j] + u1[i-1][j] + u1[i][j+1] + u1[i][j-1] - 4*u1[i][j] - dx*dx*g1[i][j]);
                    max_residue = fabs(residue) > max_residue ?  fabs(residue) : max_residue;
                    u1[i][j] += omega*residue;
                } 
            }
        }

        j = 2*N - 2;
        for(i = 1; i < num_rows - 1; ++i){
            if((i+j)%2 != 0){
                residue = 0.25*(u1[i+1][j] + u1[i-1][j] + u1[i][j-1] - 3*u1[i][j] - dx*dx*g1[i][j]);
                max_residue = fabs(residue) > max_residue ?  fabs(residue) : max_residue;
                u1[i][j] += omega*residue;
            } 
        }

        // send points to the right
        if(rank > 0){
            MPI_Recv(&(u1[0][0]), 2*N - 1, MPI_DOUBLE, rank - 1,  2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        index = num_rows - 2;
        if(rank != size-1){
            MPI_Send(&(u1[index][0]), 2*N - 1, MPI_DOUBLE, rank + 1, 2, MPI_COMM_WORLD);
        }

        // send points to the left
        if(rank != size - 1){
            MPI_Recv(&(u1[index + 1][0]), 2*N - 1, MPI_DOUBLE, rank + 1,  3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if(rank > 0){
            MPI_Send(&(u1[1][0]), 2*N - 1, MPI_DOUBLE, rank - 1, 3, MPI_COMM_WORLD);
        }

        // update black points
        j=0;
        for(i = 1; i < num_rows - 1; ++i){
            if((i+j)%2 == 0){
                residue = 0.25*(u1[i+1][j] + u1[i-1][j] + u1[i][j+1] - 3*u1[i][j] - dx*dx*g1[i][j]);
                max_residue = fabs(residue) > max_residue ?  fabs(residue) : max_residue;
                u1[i][j] += omega*residue;
            } 
        }

        for(i = 1; i < num_rows - 1; ++i){
            for(j = 1; j < 2*N - 2; ++j){
                if((i+j)%2 == 0){
                    residue = 0.25*(u1[i+1][j] + u1[i-1][j] + u1[i][j+1] + u1[i][j-1] - 4*u1[i][j] - dx*dx*g1[i][j]);
                    max_residue = fabs(residue) > max_residue ?  fabs(residue) : max_residue;
                    u1[i][j] += omega*residue;
                } 
            }
        }

        j = 2*N - 2;
        for(i = 1; i < num_rows - 1; ++i){
            if((i+j)%2 == 0){
                residue = 0.25*(u1[i+1][j] + u1[i-1][j] + u1[i][j-1] - 3*u1[i][j] - dx*dx*g1[i][j]);
                max_residue = fabs(residue) > max_residue ?  fabs(residue) : max_residue;
                u1[i][j] += omega*residue;
            } 
        }    

        // send points to the right
        if(rank > 0){
            MPI_Recv(&(u1[0][0]), 2*N - 1, MPI_DOUBLE, rank - 1,  2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        index = num_rows - 2;
        if(rank != size-1){
            MPI_Send(&(u1[index][0]), 2*N - 1, MPI_DOUBLE, rank + 1, 2, MPI_COMM_WORLD);
        }

        // send points to the left
        if(rank != size - 1){
            MPI_Recv(&(u1[index + 1][0]), 2*N - 1, MPI_DOUBLE, rank + 1,  3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if(rank > 0){
            MPI_Send(&(u1[1][0]), 2*N - 1, MPI_DOUBLE, rank - 1, 3, MPI_COMM_WORLD);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allreduce(&max_residue, &max_residue, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);


        // gather data
        int M = (2*N - 1)*(N/size);
        offsets[0] = 0;
        srccount[0] = (2*N-1)*(N/size);
        if(rank == 0){
            index = 0;
        }
        else{
            index = 1;
        }
        for(i = 1; i < size; ++i){
            srccount[i] = (2*N-1)*(N/size);
            offsets[i] = offsets[i-1] + srccount[i-1];
        }
        MPI_Gatherv(&(u1[index][0]), M, MPI_DOUBLE, &(u0[0][0]), srccount, offsets, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        // write to output file
    }

    double (*u_actual)[N] = malloc((2*N-1)*sizeof(*u_actual));
    if(rank == 0){
        for(i = 0; i < 2*N - 1; ++i){
            for(j = 0; j < N; ++j){
                u_actual[i][j] = u0[j][i];
            }
        }
        fwrite(&(u_actual[0][0]), sizeof(double), (2*N-1)*N, fileid);
    }
    
    double elapsedtime = MPI_Wtime()-starttime;
    if (rank == 0) {
    printf("Execution time = %le seconds\n", elapsedtime);
    printf("Precision of timing is %le seconds\n",  precision);
    printf("iterations: %d\n", num_iter);
    }
    MPI_Finalize();
    fclose(fileid);
    return 0;
}