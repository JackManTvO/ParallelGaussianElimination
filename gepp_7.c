/************************************************************************************
 * FILE: gepp_7.c
 * DESCRIPTION:
 * AUTHOR: Ruoxuan Liu
 * LAST REVISED: 29/05/2024
 *************************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <mpi.h>

void print_matrix(double **t, int rows, int cols);
int test(double **t1, double **t2, int rows);
int main(int argc, char *argv[])
{
    int myid;  // process id
    int procs; // total number of processes created

    double *a0;    // auxiliary 1D for 2D matrix a
    double **a;    // 2D matrix for sequential computation
    double *d0;    // auxiliary 1D for 2D matrix d
    double **d;    // 2D matrix, same initial data as a for computation
    int n;         // input size
    int blockSize; // block size

    int i, j, k, l;
    int indk;
    double amax;
    double temp, prodFactor;

    double di00, di01, di02, di03;
    double di10, di11, di12, di13;
    double di20, di21, di22, di23;
    double di30, di31, di32, di33;

    double dj00, dj01, dj02, dj03;
    double dj10, dj11, dj12, dj13;
    double dj20, dj21, dj22, dj23;
    double dj30, dj31, dj32, dj33;
    double aw00;

    int blocks;
    int blocksLocal;
    int cols;
    int ib, jb;
    int i0, j0, i1, j1, jg;
    int iproc;

    double *ak0;
    double **ak;
    double *aw0;
    double **aw;

    struct timeval start_time, end_time;
    long seconds, microseconds;
    double elapsed;

    MPI_Status status;
    MPI_Datatype blockType, blockLocalType, swapInfoType, lowerType, blueType;

    int info[blockSize];
    int disps[blockSize], blocklens[blockSize];

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    if (argc != 2)
    {
        if (myid == 0)
        {
            printf("Wrong number of arguments.\n");
            printf("Please enter the command in the following format:\n");
            printf("mpirun –np [proc num] main [matrix size N]\n");
            printf("SAMPLE: mpirun –np 3 main 40\n");
            printf("Usage: %s n\n\n"
                   " n: the matrix size\n",
                   argv[0]);
        }
        MPI_Finalize();
        return 0;
    }
    if (procs == 1)
    {
        printf("\n\nThe number of processes created is just 1 - a trivial problem!\n\n");
        MPI_Finalize();
        return 0;
    }

    n = atoi(argv[1]);
    blockSize = 8;

    if (myid == 0)
    {
        printf("The matrix size: %d * %d \n", n, n);
        printf("The block size: %d \n", blockSize);
        printf("Creating and initializing matrices...\n\n");
        /*** Allocate contiguous memory for 2D matrices ***/
        a0 = (double *)malloc(n * n * sizeof(double));
        a = (double **)malloc(n * sizeof(double *));
        for (i = 0; i < n; i++)
        {
            a[i] = a0 + i * n;
        }
        d0 = (double *)malloc(n * n * sizeof(double));
        d = (double **)malloc(n * sizeof(double *));
        for (i = 0; i < n; i++)
        {
            d[i] = d0 + i * n;
        }
        srand(time(0));
        for (i = 0; i < n; i++)
        {
            for (j = 0; j < n; j++)
            {
                a[i][j] = (double)rand() / RAND_MAX;
                d[i][j] = a[i][j];
            }
        }

        // printf("matrix a: \n");
        // print_matrix(a, n, n);

        printf("Starting sequential computation...\n\n");
        /**** Sequential computation *****/
        gettimeofday(&start_time, 0);
        for (i = 0; i < n - 1; i++)
        {
            // find and record k where |a(k,i)|= ax|a(j,i)|𝑚
            amax = a[i][i];
            indk = i;
            for (k = i + 1; k < n; k++)
            {
                if (fabs(a[k][i]) > fabs(amax))
                {
                    amax = a[k][i];
                    indk = k;
                }
            }
            // exit with a warning that a is singular
            if (amax == 0)
            {
                printf("matrix is singular!\n");
                exit(1);
            }
            else if (indk != i) // swap row i and row k
            {
                for (j = 0; j < n; j++)
                {
                    temp = a[i][j];
                    a[i][j] = a[indk][j];
                    a[indk][j] = temp;
                }
            }
            // store multiplier in place of A(k,i)
            for (k = i + 1; k < n; k++)
            {
                a[k][i] = a[k][i] / a[i][i];
            }
            // subtract multiple of row a(i,:) to zero out a(j,i)
            for (k = i + 1; k < n; k++)
            {
                prodFactor = a[k][i];
                for (j = i + 1; j < n; j++)
                {
                    a[k][j] -= prodFactor * a[i][j];
                }
            }
        }
        gettimeofday(&end_time, 0);
        // print the running time
        seconds = end_time.tv_sec - start_time.tv_sec;
        microseconds = end_time.tv_usec - start_time.tv_usec;
        elapsed = seconds + 1e-6 * microseconds;
        printf("sequential calculation time: %f\n\n", elapsed);
    }

    blocks = n / blockSize; // the total number of column blocks

    /* All processes create a submatrix ak of size N X K */
    if (myid < blocks % procs) // one more row block for each of the first r processes
        blocksLocal = blocks / procs + 1;
    else
        blocksLocal = blocks / procs;
    cols = blocksLocal * blockSize; // number of columns in submatrix and may be different for different processes

    ak0 = malloc(n * cols * sizeof(double));
    ak = malloc(n * sizeof(double *));
    aw0 = malloc(n * blockSize * sizeof(double));
    aw = malloc(n * sizeof(double *));
    if (ak == NULL)
    {
        fprintf(stderr, "**AK out of memory\n");
        exit(1);
    }
    if (aw == NULL)
    {
        fprintf(stderr, "**AW out of memory\n");
        exit(1);
    }
    for (i = 0; i < n; i++)
        ak[i] = ak0 + i * cols;
    for (i = 0; i < n; i++)
        aw[i] = aw0 + i * blockSize;

    /* initialize matrix AK */
    for (i = 0; i < n; i++)
        for (j = 0; j < cols; j++)
            ak[i][j] = 0;

    /* initialize matrix AW */
    for (i = 0; i < n; i++)
        for (j = 0; j < blockSize; j++)
            aw[i][j] = 0;

    MPI_Type_vector(n, blockSize, n, MPI_DOUBLE, &blockType);
    MPI_Type_commit(&blockType);

    MPI_Type_vector(n, blockSize, cols, MPI_DOUBLE, &blockLocalType);
    MPI_Type_commit(&blockLocalType);

    MPI_Type_contiguous(blockSize, MPI_INT, &swapInfoType);
    MPI_Type_commit(&swapInfoType);

    for (i = 0; i < blockSize; i++)
    {
        disps[i] = blockSize * i;
        blocklens[i] = i;
    }
    MPI_Type_indexed(blockSize, blocklens, disps, MPI_DOUBLE, &lowerType);
    MPI_Type_commit(&lowerType);

    /* send column blocks to other processes with block cyclic partitioning*/
    if (myid == 0)
    {
        for (ib = 0; ib < blocks; ib++)
        {
            iproc = ib % procs;
            jb = ib / procs;
            if (iproc == 0)
            {
                for (i = 0; i < n; i++)
                {
                    for (j = 0; j < blockSize; j++)
                    {
                        ak[i][jb * blockSize + j] = d[i][ib * blockSize + j];
                    }
                }
            }
            else
            {
                MPI_Send(&d[0][ib * blockSize], 1, blockType, iproc, 0, MPI_COMM_WORLD);
            }
        }
    }
    else
    {
        for (i = 0; i < blocksLocal; i++)
        {
            MPI_Recv(&ak[0][i * blockSize], 1, blockLocalType, 0, 0, MPI_COMM_WORLD, &status);
        }
    }

    if (myid == 0)
    {
        printf("Starting parallel computation with loop unrolling...\n\n");
        /***parallrl computation with loop unrolling***/
        gettimeofday(&start_time, 0);
    }

    for (ib = 0; ib < blocks; ib++)
    {
        iproc = ib % procs;
        i0 = ib * blockSize;
        i1 = i0 + blockSize;
        jb = ib / procs;
        j0 = jb * blockSize;
        j1 = j0 + blockSize;
        jg = myid > iproc ? j0 : j1;

        if (myid == iproc)
        {
            for (i = i0; i < i1; i++)
            {
                j = j0 + i - i0;
                amax = ak[i][j];
                indk = i;
                for (k = i + 1; k < n; k++)
                    if (fabs(ak[k][j]) > fabs(amax))
                    {
                        amax = ak[k][j];
                        indk = k;
                    }
                if (amax == 0.0)
                {
                    printf("the matrix is singular\n");
                    exit(1);
                }
                else if (indk != i) // swap row i and row indk in block
                {
                    for (l = 0; l < cols; l++)
                    {
                        temp = ak[i][l];
                        ak[i][l] = ak[indk][l];
                        ak[indk][l] = temp;
                    }
                }
                info[i - i0] = indk;

                // store multiplier in place of ak(k,j)
                for (k = i + 1; k < n; k++)
                    ak[k][j] = ak[k][j] / ak[i][j];

                // subtract multiple of row ak(i,:) to zero out ak(k,j)
                for (k = i + 1; k < n; k++)
                {
                    prodFactor = ak[k][j];
                    for (l = j + 1; l < j1; l++)
                    {
                        ak[k][l] -= prodFactor * ak[i][l];
                    }
                }
            }
        }
        // broadcast swap information
        MPI_Bcast(info, 1, swapInfoType, iproc, MPI_COMM_WORLD);
        for (i = i0; i < i1; i++)
        {
            indk = info[i - i0];
            if (indk != i)
            {
                for (l = 0; l < cols; l++)
                {
                    temp = ak[i][l];
                    ak[i][l] = ak[indk][l];
                    ak[indk][l] = temp;
                }
            }
        }

        /***update the main block of b rows***/
        if (myid == iproc)
        {
            for (i = i0; i < i1; i++)
            {
                for (j = 0; j < i - i0; j++)
                {
                    aw[i][j] = ak[i][j + j0];
                }
            }
        }
        // broadcast Lower triangular matrix right
        MPI_Bcast(&aw[i0][0], 1, lowerType, iproc, MPI_COMM_WORLD);

        // D(i0:i1,i1+1:n)
        for (i = i0; i < i1; i++)
        {
            for (k = 0; k < i - i0; k++)
            {
                aw00 = aw[i][k];
                for (j = jg; j < cols; j += 4)
                {
                    ak[i][j] -= aw00 * ak[i0 + k][j];
                    ak[i][j + 1] -= aw00 * ak[i0 + k][j + 1];
                    ak[i][j + 2] -= aw00 * ak[i0 + k][j + 2];
                    ak[i][j + 3] -= aw00 * ak[i0 + k][j + 3];
                }
            }
        }

        /*** broadcast blue right ***/
        MPI_Type_vector(n - i1, blockSize, blockSize, MPI_DOUBLE, &blueType);
        MPI_Type_commit(&blueType);

        if (myid == iproc && ib != blocks - 1)
        {
            for (i = i1; i < n; i++)
            {
                for (j = 0; j < blockSize; j++)
                {
                    aw[i][j] = ak[i][j + j0];
                }
            }
        }
        MPI_Bcast(&aw[i1][0], 1, blueType, iproc, MPI_COMM_WORLD);

        /***update the trailing submatrix (rank-b updating)***/
        for (k = 0; k < blockSize; k += 4)
        {
            for (i = i1; i < n; i += 4)
            {
                di00 = aw[i][k];
                di01 = aw[i][k + 1];
                di02 = aw[i][k + 2];
                di03 = aw[i][k + 3];
                di10 = aw[i + 1][k];
                di11 = aw[i + 1][k + 1];
                di12 = aw[i + 1][k + 2];
                di13 = aw[i + 1][k + 3];
                di20 = aw[i + 2][k];
                di21 = aw[i + 2][k + 1];
                di22 = aw[i + 2][k + 2];
                di23 = aw[i + 2][k + 3];
                di30 = aw[i + 3][k];
                di31 = aw[i + 3][k + 1];
                di32 = aw[i + 3][k + 2];
                di33 = aw[i + 3][k + 3];
                for (j = jg; j < cols; j += 4)
                {
                    dj00 = ak[i0 + k][j];
                    dj01 = ak[i0 + k][j + 1];
                    dj02 = ak[i0 + k][j + 2];
                    dj03 = ak[i0 + k][j + 3];
                    dj10 = ak[i0 + k + 1][j];
                    dj11 = ak[i0 + k + 1][j + 1];
                    dj12 = ak[i0 + k + 1][j + 2];
                    dj13 = ak[i0 + k + 1][j + 3];
                    dj20 = ak[i0 + k + 2][j];
                    dj21 = ak[i0 + k + 2][j + 1];
                    dj22 = ak[i0 + k + 2][j + 2];
                    dj23 = ak[i0 + k + 2][j + 3];
                    dj30 = ak[i0 + k + 3][j];
                    dj31 = ak[i0 + k + 3][j + 1];
                    dj32 = ak[i0 + k + 3][j + 2];
                    dj33 = ak[i0 + k + 3][j + 3];

                    ak[i][j] = ak[i][j] - di00 * dj00 - di01 * dj10 - di02 * dj20 - di03 * dj30;
                    ak[i + 1][j] = ak[i + 1][j] - di10 * dj00 - di11 * dj10 - di12 * dj20 - di13 * dj30;
                    ak[i + 2][j] = ak[i + 2][j] - di20 * dj00 - di21 * dj10 - di22 * dj20 - di23 * dj30;
                    ak[i + 3][j] = ak[i + 3][j] - di30 * dj00 - di31 * dj10 - di32 * dj20 - di33 * dj30;

                    ak[i][j + 1] = ak[i][j + 1] - di00 * dj01 - di01 * dj11 - di02 * dj21 - di03 * dj31;
                    ak[i + 1][j + 1] = ak[i + 1][j + 1] - di10 * dj01 - di11 * dj11 - di12 * dj21 - di13 * dj31;
                    ak[i + 2][j + 1] = ak[i + 2][j + 1] - di20 * dj01 - di21 * dj11 - di22 * dj21 - di23 * dj31;
                    ak[i + 3][j + 1] = ak[i + 3][j + 1] - di30 * dj01 - di31 * dj11 - di32 * dj21 - di33 * dj31;

                    ak[i][j + 2] = ak[i][j + 2] - di00 * dj02 - di01 * dj12 - di02 * dj22 - di03 * dj32;
                    ak[i + 1][j + 2] = ak[i + 1][j + 2] - di10 * dj02 - di11 * dj12 - di12 * dj22 - di13 * dj32;
                    ak[i + 2][j + 2] = ak[i + 2][j + 2] - di20 * dj02 - di21 * dj12 - di22 * dj22 - di23 * dj32;
                    ak[i + 3][j + 2] = ak[i + 3][j + 2] - di30 * dj02 - di31 * dj12 - di32 * dj22 - di33 * dj32;

                    ak[i][j + 3] = ak[i][j + 3] - di00 * dj03 - di01 * dj13 - di02 * dj23 - di03 * dj33;
                    ak[i + 1][j + 3] = ak[i + 1][j + 3] - di10 * dj03 - di11 * dj13 - di12 * dj23 - di13 * dj33;
                    ak[i + 2][j + 3] = ak[i + 2][j + 3] - di20 * dj03 - di21 * dj13 - di22 * dj23 - di23 * dj33;
                    ak[i + 3][j + 3] = ak[i + 3][j + 3] - di30 * dj03 - di31 * dj13 - di32 * dj23 - di33 * dj33;
                }
            }
        }
    }

    if (myid == 0)
    {
        for (ib = 0; ib < blocks; ib++)
        {
            iproc = ib % procs;
            i0 = ib * blockSize;
            jb = ib / procs;
            j0 = jb * blockSize;
            if (iproc == 0)
            {
                for (i = 0; i < n; i++)
                {
                    for (j = 0; j < blockSize; j++)
                    {
                        d[i][i0 + j] = ak[i][j0 + j];
                    }
                }
            }
            else
            {
                MPI_Recv(&d[0][i0], 1, blockType, iproc, 1, MPI_COMM_WORLD, &status);
            }
        }
    }
    else
    {
        for (ib = 0; ib < blocksLocal; ib++)
        {
            MPI_Send(&ak[0][ib * blockSize], 1, blockLocalType, 0, 1, MPI_COMM_WORLD);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (myid == 0)
    {
        gettimeofday(&end_time, 0);
        // print the running time
        seconds = end_time.tv_sec - start_time.tv_sec;
        microseconds = end_time.tv_usec - start_time.tv_usec;
        elapsed = seconds + 1e-6 * microseconds;

        printf("parallel calculation with loop unrolling time: %f\n\n", elapsed);
        printf("Starting comparison...\n\n");
    }
    free(ak0);
    free(ak);
    free(aw0);
    free(aw);

    MPI_Type_free(&blockType);
    MPI_Type_free(&blockLocalType);
    MPI_Type_free(&swapInfoType);
    MPI_Type_free(&lowerType);
    MPI_Type_free(&blueType);

    if (myid == 0)
    {
        int cnt;
        cnt = test(a, d, n);
        if (cnt == 0)
            printf("Done. There are no differences!\n");
        else
            printf("Results are incorrect! The number of different elements is %d\n",
                   cnt);
        free(a0);
        free(a);
        free(d0);
        free(d);
    }

    MPI_Finalize();
    return 1;
}

void print_matrix(double **t, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%.2f ", t[i][j]);
        }
        printf("\n");
    }
    printf("\n\n");
    return;
}

int test(double **t1, double **t2, int rows)
{
    int i, j;
    int cnt;
    cnt = 0;
    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < rows; j++)
        {
            if ((t1[i][j] - t2[i][j]) * (t1[i][j] - t2[i][j]) > 1.0e-16)
            {
                cnt += 1;
            }
        }
    }
    return cnt;
}