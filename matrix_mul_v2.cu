#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h> 

typedef struct {
    int width;
    int height;
    int stride;
    float* elements;
} Matrix;

#define FIXED_VALUE 2.0f
#define BLOCK_SIZE 16

__device__ float GetElement(const Matrix mat, int row, int col)
{
    return mat.elements[row * mat.stride + col];
}

__device__ void SetElement(Matrix mat, int row, int col, float value)
{
    mat.elements[row * mat.stride + col] = value;
}

 __device__ Matrix GetSubMatrix(const Matrix mat, int subRow, int subCol)
{
    Matrix subMat;
    subMat.width = BLOCK_SIZE;
    subMat.height = BLOCK_SIZE;
    subMat.stride = mat.stride;
    subMat.elements = &mat.elements[subRow * BLOCK_SIZE * mat.stride + subCol * BLOCK_SIZE];
    return subMat;
}


 __global__ void CalcMatMulKernel(Matrix matA, Matrix matB, Matrix matC)
{
    int blockRowIdx = blockIdx.y;
    int blockColIdx = blockIdx.x;

    Matrix subMatC = GetSubMatrix(matC, blockRowIdx, blockColIdx);

    float result = 0.0f;

    int localRowIdx = threadIdx.y;
    int localColIdx = threadIdx.x;

    for (int phase = 0; phase < (matA.width / BLOCK_SIZE); ++phase) {
        Matrix subMatA = GetSubMatrix(matA, blockRowIdx, phase);
        Matrix subMatB = GetSubMatrix(matB, phase, blockColIdx);

        __shared__ float sharedMatA[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float sharedMatB[BLOCK_SIZE][BLOCK_SIZE];

        sharedMatA[localRowIdx][localColIdx] = GetElement(subMatA, localRowIdx, localColIdx);
        sharedMatB[localRowIdx][localColIdx] = GetElement(subMatB, localRowIdx, localColIdx);

        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            result += sharedMatA[localRowIdx][k] * sharedMatB[k][localColIdx];
        }
        __syncthreads();
    }

    SetElement(subMatC, localRowIdx, localColIdx, result);
}

void MatMul(const Matrix matA, const Matrix matB, Matrix matC)
{
    Matrix devMatA;
    devMatA.width = devMatA.stride = matA.width;
    devMatA.height = matA.height;
    size_t bytesA = matA.width * matA.height * sizeof(float);
    cudaMalloc(&devMatA.elements, bytesA);
    cudaMemcpy(devMatA.elements, matA.elements, bytesA, cudaMemcpyHostToDevice);

    Matrix devMatB;
    devMatB.width = devMatB.stride = matB.width;
    devMatB.height = matB.height;
    size_t bytesB = matB.width * matB.height * sizeof(float);
    cudaMalloc(&devMatB.elements, bytesB);
    cudaMemcpy(devMatB.elements, matB.elements, bytesB, cudaMemcpyHostToDevice);

    Matrix devMatC;
    devMatC.width = devMatC.stride = matC.width;
    devMatC.height = matC.height;
    size_t bytesC = matC.width * matC.height * sizeof(float);
    cudaMalloc(&devMatC.elements, bytesC);

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((matB.width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (matA.height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    CalcMatMulKernel<<<blocksPerGrid, threadsPerBlock>>>(devMatA, devMatB, devMatC);

    cudaMemcpy(matC.elements, devMatC.elements, bytesC, cudaMemcpyDeviceToHost);

    cudaFree(devMatA.elements);
    cudaFree(devMatB.elements);
    cudaFree(devMatC.elements);
}




/***************
**TESTING CODE**
****************/
void fillMatrixRandom(Matrix &matrix) {
    for (int i = 0; i < matrix.width * matrix.height; i++) {
        matrix.elements[i] = static_cast<float>(rand()) / RAND_MAX; // Losowe wartości między 0 a 1
    }
}

void fillMatrixFixed(Matrix &matrix, float value) {
    for (int i = 0; i < matrix.width * matrix.height; i++) {
        matrix.elements[i] = value;
    }
}

void printMatrix(const Matrix &matrix) {
    for (int i = 0; i < matrix.height; i++) {
        for (int j = 0; j < matrix.width; j++) {
            printf("%f ", matrix.elements[i * matrix.width + j]);
        }
        printf("\n");
    }
}

int main(int argc, char* argv[]) {
    if (argc < 4 || argc > 6) {
        printf("Użycie: %s <n> <m> <k> [print] [fixed]\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    int m = atoi(argv[2]);
    int k = atoi(argv[3]);
    bool printResult = (argc >= 5 && strcmp(argv[4], "print") == 0);
    bool useFixedValues = (argc == 6 && strcmp(argv[5], "fixed") == 0);

    Matrix A, B, C;
    A.width = m; A.height = n;
    B.width = k; B.height = m;
    C.width = k; C.height = n;

    A.elements = (float*)malloc(A.width * A.height * sizeof(float));
    B.elements = (float*)malloc(B.width * B.height * sizeof(float));
    C.elements = (float*)malloc(C.width * C.height * sizeof(float));

    srand(time(0));
    if (useFixedValues) {
        fillMatrixFixed(A, FIXED_VALUE);
        fillMatrixFixed(B, FIXED_VALUE);
    } else {
        fillMatrixRandom(A);
        fillMatrixRandom(B);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    MatMul(A, B, C);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Czas wykonania mnożenia macierzy na GPU: %f ms\n", milliseconds);
    printf("Wymiary wynikowa macierz C: %d x %d\n", C.height, C.width);
    if (printResult) {
        printMatrix(C);
    }

    free(A.elements);
    free(B.elements);
    free(C.elements);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}