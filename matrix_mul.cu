#include <cstdio>          // Nagłówek do printf
#include <cstdlib>         // Nagłówek do funkcji rand()
#include <ctime>           // Nagłówek do inicjalizacji generatora liczb losowych
#include <cuda_runtime.h>  // Nagłówek do funkcji czasu CUDA

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;

// Thread block size
#define BLOCK_SIZE 16

// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size,
               cudaMemcpyHostToDevice);
    Matrix d_B;
    d_B.width = B.width; d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size,
               cudaMemcpyHostToDevice);

    // Allocate C in device memory
    Matrix d_C;
    d_C.width = C.width; d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

    // Read C from device memory
    cudaMemcpy(C.elements, d_C.elements, size,
               cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    // Each thread computes one element of C
    // by accumulating results into Cvalue
    float Cvalue = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    for (int e = 0; e < A.width; ++e)
        Cvalue += A.elements[row * A.width + e]
                * B.elements[e * B.width + col];
    C.elements[row * C.width + col] = Cvalue;
}

// Funkcja do wypełniania macierzy losowymi wartościami
void fillMatrixRandom(Matrix &matrix) {
    for (int i = 0; i < matrix.width * matrix.height; i++) {
        matrix.elements[i] = static_cast<float>(rand()) / RAND_MAX; // Losowe wartości między 0 a 1
    }
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        printf("Użycie: %s <n> <m> <k>\n", argv[0]);
        return 1;
    }

    // Wczytanie argumentów n, m, k
    int n = atoi(argv[1]);
    int m = atoi(argv[2]);
    int k = atoi(argv[3]);

    // Sprawdzenie poprawności wymiarów (muszą być wielokrotnościami BLOCK_SIZE)
    if (n % BLOCK_SIZE != 0 || m % BLOCK_SIZE != 0 || k % BLOCK_SIZE != 0) {
        printf("Wymiary muszą być wielokrotnościami %d\n", BLOCK_SIZE);
        return 1;
    }

    // Inicjalizacja macierzy hosta (CPU)
    Matrix A, B, C;
    A.width = m; A.height = n;
    B.width = k; B.height = m;
    C.width = k; C.height = n;

    // Alokacja pamięci na CPU
    A.elements = (float*)malloc(A.width * A.height * sizeof(float));
    B.elements = (float*)malloc(B.width * B.height * sizeof(float));
    C.elements = (float*)malloc(C.width * C.height * sizeof(float));

    // Inicjalizacja generatora liczb losowych
    srand(time(0));

    // Wypełnienie macierzy A i B losowymi wartościami
    fillMatrixRandom(A);
    fillMatrixRandom(B);

    // Tworzenie zdarzeń CUDA do mierzenia czasu
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start pomiaru czasu
    cudaEventRecord(start, 0);

    // Wywołanie mnożenia macierzy
    MatMul(A, B, C);

    // Zatrzymanie pomiaru czasu
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Obliczanie czasu wykonania
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Czas wykonania mnożenia macierzy na GPU: %f ms\n", milliseconds);

    // Zwolnienie pamięci
    free(A.elements);
    free(B.elements);
    free(C.elements);

    // Zwolnienie zdarzeń CUDA
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}