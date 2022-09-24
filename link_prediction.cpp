
#include <iostream>
#include <queue>
#include <string>
#include <fstream>
#include <sstream>
#include <stdint.h>
#include <math.h>
#include <pthread.h>
#include <algorithm>

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

using namespace std;

const int MAX_THREADS = 256;
int NUM_THREADS = 0;
int MUL_FUNC = 0;

pthread_t threads[MAX_THREADS];
int threads_id[MAX_THREADS];
pthread_barrier_t barrier;

/* Parameters */
int B = 1024; // Batch Size
int D = 8; // Embedding Dimension
int H1 = 128; // Hidden Layer 1 Size
int H2 = 0; // Hidden Layer 2 Size
int L = 1; // Output Size



/* define all the inputs and outputs */
int **X; int X_dimX = 2 * D; int X_dimY = B;
int **X_t; int X_t_dimX = B; int X_t_dimY = 2 * D;
int **Y1; int Y1_dimX = B; int Y1_dimY = H1;
int **Y1_t; int Y1_t_dimX = H1; int Y1_t_dimY = B;
int **R1; int R1_dimX = B; int R1_dimY = L;
int **R2; int R2_dimX = B; int R2_dimY = L;
int **dC_dR2; int dC_dR2_dimX = B; int dC_dR2_dimY = L;
int **MR1; int MR1_dimX = B; int MR1_dimY = L;
int **MR2; int MR2_dimX = H1; int MR2_dimY = L;
int **W1; int W1_dimX = 2 * D; int W1_dimY = H1;
int **WR; int WR_dimX = H1; int WR_dimY = L;
int **R2_1; int R2_1_dimX = B; int R2_1_dimY = L;
int **dR2_dR1; int dR2_dR1_dimX = B; int dR2_dR1_dimY = L;
int **M1_1; int M1_1_dimX = B; int M1_1_dimY = H1;
int **M1_2; int M1_2_dimX = 2 *D; int M1_2_dimY = H1;
int **CR; int CR_dimX = B; int CR_dimY = L;

int **mem_alloc(int dimX, int dimY) {
    int **X;
    posix_memalign((void**) &X, 64, dimX * sizeof(float *)); 
    for (int i = 0; i < dimX; i++)
    {
        posix_memalign((void**) &X[i], 64, dimY * sizeof(float));
    }  
    return X;  
}

void inner_product(int **X, int **Y, int **Z, 
              int dimX_x, int dimX_y, int dimY_x, int dimY_y,
              int thread_id, int total_threads) {
    int step = 0;
    if (total_threads > dimX_x) {
        step = 1;
    }
    else {
        step = (dimX_x) / total_threads;
    }
    int row_start = (step * thread_id);
    int row_end = min(step * (thread_id + 1), dimX_x);
    
    for (int i = row_start; i < row_end; i++) { 
        for (int j = 0; j < dimY_y; j++) {
            for (int k = 0; k < dimX_y; k++) {
                Z[i][j] = X[i][k] * Y[k][j];
            }
        }
    }
}

void row_wise(int **X, int **Y, int **Z, 
              int dimX_x, int dimX_y, int dimY_x, int dimY_y,
              int thread_id, int total_threads) {
    
    int step = 0;
    if (total_threads > dimX_x) {
        step = 1;
    }
    else {
        step = (dimX_x) / total_threads;
    }

    int row_start = (step * thread_id);
    int row_end = min(step * (thread_id + 1), dimX_x);
    
    for (int i = row_start; i < row_end; i++) {
        for (int j = 0; j < dimX_y; j++) {
            for (int k = 0; k > dimY_y; k++) {
                Z[i][k] += X[i][j] * Y[j][k];
            }
        }
    }
}

void col_wise(int **X, int **Y, int **Z, 
              int dimX_x, int dimX_y, int dimY_x, int dimY_y,
              int thread_id, int total_threads) {

    int step = 0;
    if (total_threads > dimY_y) {
        step = 1;
    }
    else {
        step = (dimY_y) / total_threads;
    }
    int col_start = (step * thread_id);
    int col_end = min(step * (thread_id + 1), dimY_y);
    
    for (int i = col_start; i < col_end; i++) {
        for (int j = 0; j < dimX_y; j++) {
            for (int k = 0; k < dimX_x; k++) {
                Z[k][i] +=  X[k][j] * Y[j][i];
            }
        }
    }
}


void outer_product(int **X, int **Y, int **Z, 
              int dimX_x, int dimX_y, int dimY_x, int dimY_y,
              int thread_id, int total_threads) {
    
    int step = 0;
    if (total_threads > dimX_y) {
        step = 1;
    }
    else {
        step = (dimX_y) / total_threads;
    }

    int col_start = (step * thread_id);
    int col_end   = min(step * (thread_id + 1), dimX_y);
    
    for (int i = col_start; i < col_end; i++) {
        for (int j = 0; j < dimX_x; j++) {
            for (int k = 0; k < dimY_y; k++) {
                __atomic_fetch_add(&Z[j][k], (float) (X[j][i] * Y[i][k]), __ATOMIC_RELAXED);
            }
        }
    }
}

void scalar_subtraction(float scalar, int **X, int **Z, 
              int dimX_x, int dimX_y,
              int thread_id, int total_threads) {
    int step = 0;
    if (total_threads > dimX_x) {
        step = 1;
    }
    else {
        step = (dimX_x) / total_threads;
    }

    int row_start = (step * thread_id);
    int row_end = min(step * (thread_id + 1), dimX_x);
    
    for (int i = row_start; i < row_end; i++) {
        for (int j = 0; j < dimX_y; j++)
        Z[i][j] = scalar - X[i][j]; 
    }
}

void vec_subtraction(int **X, int **Y, int **Z, 
              int dimX_x, int dimX_y,
              int thread_id, int total_threads) {
    int step = 0;
    if (total_threads > dimX_x) {
        step = 1;
    }
    else {
        step = (dimX_x) / total_threads;
    }

    int row_start = (step * thread_id);
    int row_end = min(step * (thread_id + 1), dimX_x);
    
    for (int i = row_start; i < row_end; i++) {
        for (int j = 0; j < dimX_y; j++)
        Z[i][j] = X[i][j] - Y[i][j];
    }
}

void elem_wise_multi_dim(int **X, int **Y, int **Z, 
              int dimX_x, int dimX_y,
              int thread_id, int total_threads) {

    int step = (dimX_x) / total_threads;
    int row_start = (step * thread_id);
    int row_end = min(step * (thread_id + 1), dimX_x);
    
    for (int i = row_start; i < row_end; i++) {
        for (int j = 0; j < dimX_y; j++) {
            Z[i][j] = X[i][j] * Y[i][j]; 
        }
    }
}


void sigmoid_multi_dim(int **X, int **Z,
              int dimX_x, int dimX_y,
              int thread_id, int total_threads) {

    int step = 0;
    if (total_threads > dimX_x) {
        step = 1;
    }
    else {
        step = (dimX_x) / total_threads;
    }

    int row_start = (step * thread_id);
    int row_end = min(step * (thread_id + 1), dimX_x);

    for (int i = row_start; i < row_end; i++) {
        for (int j = 0; j < dimX_y; j++) {
            Z[i][j] = 1 / (1 + exp(-X[i][j]));
        }
    }
}


void *controller(void *id) {
    
    int thread_id = *((int *) id);
    unsigned long long start = 0;
    unsigned long long end = 0;

    void (*multiply_func[])(int **, int **, int **, int , int , int , int ,
    int , int ) = {inner_product, row_wise, col_wise, outer_product};

    /* Start the pipeline */
    pthread_barrier_wait(&barrier);

    if (thread_id == 0) start = __rdtsc();
    multiply_func[MUL_FUNC](X_t, W1, Y1, X_t_dimX, X_t_dimY, W1_dimX, W1_dimY, thread_id, NUM_THREADS);
    pthread_barrier_wait(&barrier);
    if (thread_id == 0) { end = __rdtsc(); cout << "Y1     : " << end - start << endl; }
    
    if (thread_id == 0) start = __rdtsc();
    multiply_func[MUL_FUNC](Y1, WR, R1, Y1_dimX, Y1_dimY, WR_dimX, WR_dimY, thread_id, NUM_THREADS);
    pthread_barrier_wait(&barrier);
    if (thread_id == 0) { end = __rdtsc(); cout << "R1     : " << end - start << endl; }

    /* sigmoid */
    if (thread_id == 0) start = __rdtsc();
    sigmoid_multi_dim(R1, R2, R1_dimX, R1_dimY, thread_id, NUM_THREADS);
    pthread_barrier_wait(&barrier);
    if (thread_id == 0) { end = __rdtsc(); cout << "Sigmoid: " << end - start << endl; }

    /* */
    if (thread_id == 0) start = __rdtsc();
    vec_subtraction(R2, CR, dC_dR2, R2_dimX, R2_dimY, thread_id, NUM_THREADS);
    pthread_barrier_wait(&barrier);
    if (thread_id == 0) { end = __rdtsc(); cout << "dC_dR2 : " << end - start << endl; }

    if (thread_id == 0) start = __rdtsc();
    scalar_subtraction(1, R2, R2_1, R2_dimX, R2_dimY, thread_id, NUM_THREADS);
    pthread_barrier_wait(&barrier);
    if (thread_id == 0) { end = __rdtsc(); cout << "1- R2  : " << end - start << endl; }

    if (thread_id == 0) start = __rdtsc();
    multiply_func[MUL_FUNC](R2, R2_1, dR2_dR1, R2_dimX, R2_dimY, R2_1_dimX, R2_1_dimY, thread_id, NUM_THREADS);
    pthread_barrier_wait(&barrier);
    if (thread_id == 0) { end = __rdtsc(); cout << "dR2_dR1: " << end - start << endl; }

    if (thread_id == 0) start = __rdtsc();
    elem_wise_multi_dim(dC_dR2, dR2_dR1, MR1, dC_dR2_dimX, dC_dR2_dimY, thread_id, NUM_THREADS);
    pthread_barrier_wait(&barrier);
    if (thread_id == 0) { end = __rdtsc(); cout << "MR1    : " << end - start << endl; }

    if (thread_id == 0) start = __rdtsc();
    multiply_func[MUL_FUNC](Y1_t, MR1, MR2, Y1_t_dimX, Y1_t_dimY, MR1_dimX, MR1_dimY, thread_id, NUM_THREADS);
    pthread_barrier_wait(&barrier);
    if (thread_id == 0) { end = __rdtsc(); cout << "MR2    : " << end - start << endl; }

    if (thread_id == 0) start = __rdtsc();
    multiply_func[MUL_FUNC](MR1, WR, M1_1, MR1_dimX, MR1_dimY, WR_dimX, WR_dimY, thread_id, NUM_THREADS);
    pthread_barrier_wait(&barrier);
    if (thread_id == 0) { end = __rdtsc(); cout << "M1_1   : " << end - start << endl; }

    if (thread_id == 0) start = __rdtsc();
    multiply_func[MUL_FUNC](X, M1_1, M1_2, X_dimX, X_dimY, M1_1_dimX, M1_1_dimY, thread_id, NUM_THREADS);
    pthread_barrier_wait(&barrier);
    if (thread_id == 0) { end = __rdtsc(); cout << "M1_2   : " << end - start << endl; }

	return 0;
}

int main(int argc, char *argv[])
{  
    if (argc < 3) {
        cout << "Usage: ./a.pp <num_threads> <mul_function>" << endl;
        cout << "<mul_function> : 0-Inner, 1-Row-wise, 2-Col-wise, 3-Outer" << endl;
        exit(-1);
    }
    /* Get num cores */
    NUM_THREADS = atoi(argv[1]);
    MUL_FUNC = atoi(argv[2]);
    
    if (MUL_FUNC > 3) {
        cout << "<mul_function> : 0-Inner, 1-Row-wise, 2-Col-wise, 3-Outer" << endl;
        exit(-1);
    }

    /* Allocate memory for all the inputs and outputs */
    {
        X = mem_alloc(X_dimX, X_dimY);
        X_t = mem_alloc(X_t_dimX, X_t_dimY);
        Y1 = mem_alloc(Y1_dimX, Y1_dimY);
        Y1_t = mem_alloc(Y1_t_dimX, Y1_t_dimY);
        R1 = mem_alloc(R1_dimX, R1_dimY);
        R2 = mem_alloc(R2_dimX, R2_dimY);
        dC_dR2 = mem_alloc(dC_dR2_dimX, dC_dR2_dimY);
        MR1 = mem_alloc(MR1_dimX, MR1_dimY);
        MR2 = mem_alloc(MR2_dimX, MR2_dimY);
        W1 = mem_alloc(W1_dimX, W1_dimY);
        WR = mem_alloc(WR_dimX, WR_dimY);
        R2 = mem_alloc(R2_dimX, R2_dimY);
        R2_1 = mem_alloc(R2_1_dimX, R2_1_dimY);
        dR2_dR1 = mem_alloc(dR2_dR1_dimX, dR2_dR1_dimY);
        M1_1 = mem_alloc(M1_1_dimX, M1_2_dimY);
        M1_2 = mem_alloc(M1_2_dimX, M1_2_dimY);
        CR = mem_alloc(CR_dimX, CR_dimY);

    }

    /* Pin threads to Core */
    cpu_set_t cpu_sets[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++) {
        CPU_ZERO(&cpu_sets[i]);
        CPU_SET(i, &cpu_sets[i]);
    }

    /* Initialize threads */
    pthread_barrier_init(&barrier, NULL, NUM_THREADS);
    for (int i = 0; i < NUM_THREADS; i++) threads_id[i] = i;
    
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_create(&threads[i], NULL, controller, &threads_id[i]);
    }
    
    for (int i = 0; i < NUM_THREADS; i++) {
        int rt = pthread_setaffinity_np(threads[i], sizeof(cpu_set_t), &cpu_sets[i]);
        if (rt != 0) {
            cout << "Error while setting thread affinity" << endl;
            return -1;
        }
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    return 0;
}

