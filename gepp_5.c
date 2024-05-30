/************************************************************************************
 * FILE: gepp_5.c
 * DESCRIPTION:
 * AUTHOR: Ruoxuan Liu
 * LAST REVISED: 13/04/2024
 *************************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <omp.h>

void print_matrix(double **T, int rows, int cols);
int test(double **t1, double **t2, int rows);

int main(int argc, char *argv[])
{
    double *a0; // auxiliary 1D for 2D matrix a
    double **a; // 2D matrix for sequential computation
    double *d0;
    double **d;
    double *L0;
    double **LL;
    double *temp0;
    double **temp;

    int n; // input size
    int n0;
    int i, j, k;
    int indk;
    double c, amax;
    register double d00, d01, d02, d03, d10, d11, d12, d13, d20, d21, d22, d23, d30, d31, d32, d33;

    int nthreads;
    double tt;

    if (argc == 3)
    {
        n = atoi(argv[1]);
        nthreads = atoi(argv[2]);
        printf("The matrix size:  %d * %d \n", n, n);
        printf("nthreads = %d\n\n", nthreads);
    }
    else
    {
        printf("Usage: %s n\n\n"
               " n: the matrix size\n\n",
               argv[0]);
        return 1;
    }

    omp_set_num_threads(nthreads);

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
    L0 = (double *)malloc(n * n * sizeof(double));
    LL = (double **)malloc(n * sizeof(double *));
    for (i = 0; i < n; i++)
    {
        LL[i] = L0 + i * n;
    }
    temp0 = (double *)malloc(n * n * sizeof(double));
    temp = (double **)malloc(n * sizeof(double *));
    for (i = 0; i < n; i++)
    {
        temp[i] = temp0 + i * n;
    }

    srand(time(0));
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
        {
            a[i][j] = (double)rand() / RAND_MAX;
            d[i][j] = a[i][j];
        }

    //    printf("matrix a: \n");
    //    print_matrix(a, n, n);

    printf("Starting sequential computation...\n\n");
    /**** Sequential computation *****/
    tt = omp_get_wtime();
    for (i = 0; i < n - 1; i++)
    {
        // find and record k where |a(k,i)|=𝑚ax|a(j,i)|
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
                c = a[i][j];
                a[i][j] = a[indk][j];
                a[indk][j] = c;
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
            c = a[k][i];
            for (j = i + 1; j < n; j++)
            {
                a[k][j] -= c * a[i][j];
            }
        }
    }
    tt = omp_get_wtime() - tt;

    // print the running time
    printf("sequential calculation time: %f\n\n", tt);

    printf("Starting parallel computation with blocking and loop unrolling...\n\n");
    /**** parallel computation *****/
    tt = omp_get_wtime();
    for (int ib = 0; ib < n; ib += 4)
    {
        int end = ib + 4 > n ? n : ib + 4;
        for (i = ib; i < end; i++)
        {
            amax = d[i][i];
            indk = i;
            for (k = i + 1; k < n; k++)
                if (fabs(d[k][i]) > fabs(amax))
                {
                    amax = d[k][i];
                    indk = k;
                }

            if (amax == 0.0)
            {
                printf("the matrix is singular\n");
                exit(1);
            }
            else if (indk != i) // swap row i and row k
            {
                for (j = 0; j < n; j++)
                {
                    c = d[i][j];
                    d[i][j] = d[indk][j];
                    d[indk][j] = c;
                }
            }
            for (k = i + 1; k < n; k++)
                d[k][i] = d[k][i] / d[i][i];
            for (k = i + 1; k < n; k++)
                for (j = i + 1; j < end; j++)
                    d[k][j] -= d[k][i] * d[i][j];
        }

        // LL
        for (k = ib; k < end; k++)
            for (j = ib; j < end; j++)
            {
                if (k == j)
                    LL[k][j] = 1.0;
                else if (k > j)
                    LL[k][j] = d[k][j];
            }
        for (k = ib; k < end; k++)
            for (j = ib; j < k; j++)
            {
                double sum = 0.0;
                for (i = j; i < k; i++)
                    sum -= LL[k][i] * LL[i][j];
                LL[k][j] = sum / LL[k][k];
            }

        // A(ib:end,end+1:n)=LL\A(ib:end,end+1:n)
        for (k = ib; k < end; k++)
        {
            i = ib;
            d00 = LL[k][i];
            d01 = LL[k][i + 1];
            d02 = LL[k][i + 2];
            d03 = LL[k][i + 3];

            for (j = end; j < n; j++)
                temp[k][j] = d00 * d[i][j] + d01 * d[i + 1][j] + d02 * d[i + 2][j] + d03 * d[i + 3][j];
        }

        for (k = ib; k < end; k++)
            for (j = end; j < n; j++)
                d[k][j] = temp[k][j];

        // A(end+1:n,end+1:n)
        // =A(end+1:n,end+1:n)-A(end+1:n,ib:end)*A(ib:end,end+1:n)
        n0 = (n - end) / 4 * 4 + end;
#pragma omp parallel for schedule(dynamic)
        for (k = end; k < n0; k += 4)
        {
            i = ib;
            d00 = d[k][i];
            d01 = d[k][i + 1];
            d02 = d[k][i + 2];
            d03 = d[k][i + 3];
            d10 = d[k + 1][i];
            d11 = d[k + 1][i + 1];
            d12 = d[k + 1][i + 2];
            d13 = d[k + 1][i + 3];
            d20 = d[k + 2][i];
            d21 = d[k + 2][i + 1];
            d22 = d[k + 2][i + 2];
            d23 = d[k + 2][i + 3];
            d30 = d[k + 3][i];
            d31 = d[k + 3][i + 1];
            d32 = d[k + 3][i + 2];
            d33 = d[k + 3][i + 3];

            for (j = end; j < n0; j += 4)
            {
                d[k][j] -= d00 * d[i][j] + d01 * d[i + 1][j] + d02 * d[i + 2][j] + d03 * d[i + 3][j];
                d[k][j + 1] -= d00 * d[i][j + 1] + d01 * d[i + 1][j + 1] + d02 * d[i + 2][j + 1] + d03 * d[i + 3][j + 1];
                d[k][j + 2] -= d00 * d[i][j + 2] + d01 * d[i + 1][j + 2] + d02 * d[i + 2][j + 2] + d03 * d[i + 3][j + 2];
                d[k][j + 3] -= d00 * d[i][j + 3] + d01 * d[i + 1][j + 3] + d02 * d[i + 2][j + 3] + d03 * d[i + 3][j + 3];
                d[k + 1][j] -= d10 * d[i][j] + d11 * d[i + 1][j] + d12 * d[i + 2][j] + d13 * d[i + 3][j];
                d[k + 1][j + 1] -= d10 * d[i][j + 1] + d11 * d[i + 1][j + 1] + d12 * d[i + 2][j + 1] + d13 * d[i + 3][j + 1];
                d[k + 1][j + 2] -= d10 * d[i][j + 2] + d11 * d[i + 1][j + 2] + d12 * d[i + 2][j + 2] + d13 * d[i + 3][j + 2];
                d[k + 1][j + 3] -= d10 * d[i][j + 3] + d11 * d[i + 1][j + 3] + d12 * d[i + 2][j + 3] + d13 * d[i + 3][j + 3];
                d[k + 2][j] -= d20 * d[i][j] + d21 * d[i + 1][j] + d22 * d[i + 2][j] + d23 * d[i + 3][j];
                d[k + 2][j + 1] -= d20 * d[i][j + 1] + d21 * d[i + 1][j + 1] + d22 * d[i + 2][j + 1] + d23 * d[i + 3][j + 1];
                d[k + 2][j + 2] -= d20 * d[i][j + 2] + d21 * d[i + 1][j + 2] + d22 * d[i + 2][j + 2] + d23 * d[i + 3][j + 2];
                d[k + 2][j + 3] -= d20 * d[i][j + 3] + d21 * d[i + 1][j + 3] + d22 * d[i + 2][j + 3] + d23 * d[i + 3][j + 3];
                d[k + 3][j] -= d30 * d[i][j] + d31 * d[i + 1][j] + d32 * d[i + 2][j] + d33 * d[i + 3][j];
                d[k + 3][j + 1] -= d30 * d[i][j + 1] + d31 * d[i + 1][j + 1] + d32 * d[i + 2][j + 1] + d33 * d[i + 3][j + 1];
                d[k + 3][j + 2] -= d30 * d[i][j + 2] + d31 * d[i + 1][j + 2] + d32 * d[i + 2][j + 2] + d33 * d[i + 3][j + 2];
                d[k + 3][j + 3] -= d30 * d[i][j + 3] + d31 * d[i + 1][j + 3] + d32 * d[i + 2][j + 3] + d33 * d[i + 3][j + 3];
            }
            for (j = n0; j < n; j++)
            {
                d[k][j] -= d00 * d[i][j] + d01 * d[i + 1][j] + d02 * d[i + 2][j] + d03 * d[i + 3][j];
                d[k + 1][j] -= d10 * d[i][j] + d11 * d[i + 1][j] + d12 * d[i + 2][j] + d13 * d[i + 3][j];
                d[k + 2][j] -= d20 * d[i][j] + d21 * d[i + 1][j] + d22 * d[i + 2][j] + d23 * d[i + 3][j];
                d[k + 3][j] -= d30 * d[i][j] + d31 * d[i + 1][j] + d32 * d[i + 2][j] + d33 * d[i + 3][j];
            }
        }

        for (k = n0; k < n; k++)
        {
            i = ib;
            d00 = d[k][i];
            d01 = d[k][i + 1];
            d02 = d[k][i + 2];
            d03 = d[k][i + 3];
            for (j = end; j < n; j++)
                d[k][j] -= d00 * d[i][j] + d01 * d[i + 1][j] + d02 * d[i + 2][j] + d03 * d[i + 3][j];
        }
    }

    tt = omp_get_wtime() - tt;

    // print the running time
    printf("parallel calculation with blocking and loop unrolling time: %f\n\n", tt);

    printf("Starting comparison...\n\n");
    int cnt;
    cnt = test(a, d, n);
    if (cnt == 0)
        printf("Done. There are no differences!\n");
    else
        printf("Results are incorrect! The number of different elements is %d\n", cnt);
}

void print_matrix(double **T, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%.2f   ", T[i][j]);
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
