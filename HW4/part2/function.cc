#include <stdio.h>
#include <mpi.h>
// #include <time.h>
int *score;

int calculate(const int *a_ptr, const int *b_ptr, int row, int col, int m, int l)
{
    int val = 0;
    for (int i = 0; i < m; i++)
    {
        // int v1 = (*a_ptr + row*m + i);
        int v1 = a_ptr[row * m + i];
        int v2 = b_ptr[i * l + col];
        // int v2 = (*b_ptr + i*l + col);
        // printf("place %d %d = %d %d\n", row , col , *a_ptr + row*m + i , *b_ptr + i*l + col  );
        // val += (*a_ptr + row*m + i) * (*b_ptr + i*l + col);
        val += v1 * v2;
    }
    return val;
}

int check(int n, int l)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < l; j++)
        {
            if (score[i * l + j] == -1)
            {
                return 0;
            }
        }
    }
    return 1;
}

// void outer_product(int )

void construct_matrices(int *n_ptr, int *m_ptr, int *l_ptr,
                        int **a_mat_ptr, int **b_mat_ptr)
{
    int world_rank, world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int n, m, l, tmp;

    if (world_rank == 0)
    {
        scanf("%d %d %d", n_ptr, m_ptr, l_ptr);
        n = *n_ptr;
        m = *m_ptr;
        l = *l_ptr;
        *a_mat_ptr = (int *)malloc(sizeof(int) * n * m);
        *b_mat_ptr = (int *)malloc(sizeof(int) * m * l);

        for (int i = 0; i < *n_ptr; i++)
        {
            for (int j = 0; j < *m_ptr; j++)
            {
                scanf("%d", &tmp);
                *(*a_mat_ptr + i * m + j) = tmp;
                // printf("%d ",*(*a_mat_ptr + i*m + j));
            }
            // printf("\n");
        }

        for (int i = 0; i < *m_ptr; i++)
        {
            for (int j = 0; j < *l_ptr; j++)
            {

                scanf("%d", &tmp);
                *(*b_mat_ptr + i * l + j) = tmp;
                // printf("%d ",tmp);
            }
        }
        // printf("out of scanf\n");
    }
    // printf("exit4\n");
    MPI_Barrier(MPI_COMM_WORLD);
    // MPI_Bcast(n_ptr,1,MPI_INT,0,MPI_COMM_WORLD);
    // MPI_Bcast(m_ptr,1,MPI_INT,0,MPI_COMM_WORLD);
    // MPI_Bcast(l_ptr,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&l, 1, MPI_INT, 0, MPI_COMM_WORLD);
    *n_ptr = n;
    *m_ptr = m;
    *l_ptr = l;

    if (world_rank != 0)
    {
        *a_mat_ptr = (int *)malloc(sizeof(int) * n * m);
        *b_mat_ptr = (int *)malloc(sizeof(int) * m * l);
    }

    // printf("%d exit5\n",world_rank);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(*a_mat_ptr, n * m, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(*b_mat_ptr, l * m, MPI_INT, 0, MPI_COMM_WORLD);
    // printf("%d exit5----------------------\n",world_rank);
}

void matrix_multiply(const int n, const int m, const int l,
                     const int *a_mat, const int *b_mat)
{
    MPI_Win win;
    int world_rank, world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    // printf("%d exit6\n",world_rank);
    // MPI_Barrier(MPI_COMM_WORLD);

    // printf("%d exit7\n",world_rank);
    if (world_rank == 0)
    {
        // printf("in mul\n");
        MPI_Alloc_mem(n * l * sizeof(int), MPI_INFO_NULL, &score);
        // score = (int *)malloc(sizeof(int) * n * l);

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < l; j++)
            {
                score[i * l + j] = 0;
            }
        }

        MPI_Win_create(score, n * l * sizeof(int), sizeof(int), MPI_INFO_NULL,
                       MPI_COMM_WORLD, &win);
    }
    else
    {
        MPI_Win_create(NULL, 0, 1, MPI_INFO_NULL,
                       MPI_COMM_WORLD, &win);
    }

    int row, col;
    MPI_Barrier(MPI_COMM_WORLD);
    // printf("rank = %d \n",world_rank);
    int val[100*l+world_size];
    int offset = 0;
    int idx = 0;
    for (row = 0; row < n; row++)
    {
        // int val[l];
        // for (col = 0; col < l; col++ )
        for(col=0 ; col < l ; col+=world_size)
        {
            for(int z = 0 ; z<world_size ; z++){
                val[offset * l + col+z] = 0;
            }
            // val[(offset) * l + col] = 0;
            // val[col] = 0;
            // for(col=0 ; col < l ; col++){
            // printf("rank = %d ,world_size = %d ,mod val = %d\n", world_rank, world_size, col % (world_size));
            // if (((col % (world_size))) == world_rank && (col - world_size + world_rank < l))
            // if(col%world_size==world_rank)
            if((col+world_rank)%world_size==world_rank)
            {
                // printf("rank = %d , col = %d\n",world_rank,col+world_rank);
                // int val = calculate(a_mat, b_mat, row, col, m, l);
                int v1,v2;
                for (int i = 0; i < m; i++)
                {
                    v1 = a_mat[row * m + i];
                    // v1 = a_mat[row * m + i];
                    v2 = b_mat[i * l + col + world_rank];
                    val[(offset)*l + col + world_rank] += v1 * v2;
                    // val[col] += v1 * v2;
                }
                // MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win);
                // MPI_Put(&val[col], 1, MPI_INT, 0, row * l + col, 1, MPI_INT, win);
                // // MPI_Accumulate
                // MPI_Win_unlock(0, win);
            }
        }
        offset++;
        if(offset==100){
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win);
            // MPI_Put(&val, l, MPI_INT, 0, row * l , l, MPI_INT, win);
            MPI_Accumulate(&val, 100*l, MPI_INT, 0, idx*100*l , 100*l, MPI_INT,MPI_SUM, win);
            MPI_Win_unlock(0, win);
            idx++;
            offset = 0;
        // MPI_Barrier(MPI_COMM_WORLD);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win);
    // MPI_Put(&val, l, MPI_INT, 0, row * l , l, MPI_INT, win);
    MPI_Accumulate(&val, offset * l, MPI_INT, 0, idx * 100 *l, offset * l, MPI_INT,MPI_SUM, win);
    MPI_Win_unlock(0, win);


    MPI_Barrier(MPI_COMM_WORLD);

    if (world_rank == 0)
    {
        int ready = 0;
        // while(!ready){
        //     MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, win);
        //     ready = check(n,l);
        //     MPI_Win_unlock(0, win);
        // }
        for (int row = 0; row < n; row++)
        {
            for (int col = 0; col < l; col++)
            {
                printf("%d ", score[row * l + col]);
            }
            printf("\n");
        }
        // MPI_Win_free(&win);
        // MPI_Win_free(&win_a);
        // MPI_Win_free(&win_b);
        MPI_Free_mem(score);
        // MPI_Free_mem(matrix_a);
        // MPI_Free_mem(matrix_b);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Win_free(&win);
}

// Remember to release your allocated memory
void destruct_matrices(int *a_mat, int *b_mat)
{
    int world_rank, world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    free(a_mat);
    free(b_mat);
}