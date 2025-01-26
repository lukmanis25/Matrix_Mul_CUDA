#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <stdbool.h>
#include <string.h>

typedef struct {
    int width;
    int height;
    float** elements;
} Matrix;

#define FIXED_VALUE 2.0f
#define NUM_THREADS 8

void allocateMatrix(Matrix* matrix, int height, int width) {
    matrix->height = height;
    matrix->width = width;
    matrix->elements = (float**)malloc(height * sizeof(float*));
    for (int i = 0; i < height; i++) {
        matrix->elements[i] = (float*)malloc(width * sizeof(float));
    }
}

void freeMatrix(Matrix* matrix) {
    for (int i = 0; i < matrix->height; i++) {
        free(matrix->elements[i]);
    }
    free(matrix->elements);
}

void updateMatrixElement(Matrix* mat, int i, int j, float new_val) {
    mat->elements[i][j] = new_val;
}

float getMatrixElement(const Matrix* mat, int i, int j) {
    return mat->elements[i][j];
}

void MatMul(const Matrix matA, const Matrix matB, Matrix matC) {
    int i, j, k;
    omp_set_num_threads(NUM_THREADS);
    #pragma omp parallel for shared(matA, matB, matC) private(i, j, k) schedule(static) collapse(2)
    for (i = 0; i < matC.height; i++) {
        for (j = 0; j < matC.width; j++) {
            float sum = 0.0f;
            #pragma omp simd
            for (k = 0; k < matA.width; k++) {
                sum += matA.elements[i][k] * matB.elements[k][j];
            }
            matC.elements[i][j] = sum;
        }
    }
}

/***************
** TESTING CODE **
****************/
// Funkcja wypełniająca macierz losowymi wartościami
void fillMatrixRandom(Matrix* matrix) {
    for (int i = 0; i < matrix->height; i++) {
        for (int j = 0; j < matrix->width; j++) {
            matrix->elements[i][j] = (float)rand() / RAND_MAX; // Losowe wartości między 0 a 1
        }
    }
}

// Funkcja wypełniająca macierz stałą wartością
void fillMatrixFixed(Matrix* matrix, float value) {
    for (int i = 0; i < matrix->height; i++) {
        for (int j = 0; j < matrix->width; j++) {
            matrix->elements[i][j] = value;
        }
    }
}

// Funkcja wypisująca macierz
void printMatrix(const Matrix* matrix) {
    for (int i = 0; i < matrix->height; i++) {
        for (int j = 0; j < matrix->width; j++) {
            printf("%f ", matrix->elements[i][j]);
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
    allocateMatrix(&A, n, m);
    allocateMatrix(&B, m, k);
    allocateMatrix(&C, n, k);

    srand(time(0));
    if (useFixedValues) {
        fillMatrixFixed(&A, FIXED_VALUE);
        fillMatrixFixed(&B, FIXED_VALUE);
    } else {
        fillMatrixRandom(&A);
        fillMatrixRandom(&B);
    }

    double start = omp_get_wtime();

    MatMul(A, B, C);

    double end = omp_get_wtime();
    double time_spent = end - start;

    printf("Czas wykonania mnożenia macierzy na CPU: %f s\n", time_spent);
    printf("Wymiary wynikowa macierz C: %d x %d\n", C.height, C.width);
    if (printResult) {
        printMatrix(&C);
    }

    freeMatrix(&A);
    freeMatrix(&B);
    freeMatrix(&C);
    return 0;
}
