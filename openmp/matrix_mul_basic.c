#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <stdbool.h>
#include <string.h>

typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;

#define FIXED_VALUE 2.0f
#define NUM_THREADS 8 //24 najwięcej

void updateMatrixElement(Matrix mat, int i, int j, float new_val){
    mat.elements[i * mat.width + j] = new_val;
}

float getMatixElement(Matrix mat, int i, int j){
    return mat.elements[i * mat.width + j];
}

void MatMul(const Matrix matA, const Matrix matB, Matrix matC) {
    int i,j,k;
    omp_set_num_threads(NUM_THREADS);
    #pragma omp parallel for shared(matA, matB, matC) private(i,j,k) schedule(static) collapse(2)
    for(i=0; i< matC.height; i++){
        for(j=0; j< matC.width; j++){
            matC.elements[i * matC.width + j] = 0.0f;
            for(k=0; k < matA.width; k++){
                matC.elements[i * matC.width + j] += matA.elements[i * matA.width + k] * matB.elements[k * matB.width + j];
            }
        }
    }
}

/***************
**TESTING CODE**
****************/
void fillMatrixRandom(Matrix* matrix) {
    for (int i = 0; i < matrix->width * matrix->height; i++) {
        matrix->elements[i] = (float)rand() / RAND_MAX; // Losowe wartości między 0 a 1
    }
}

void fillMatrixFixed(Matrix* matrix, float value) {
    for (int i = 0; i < matrix->width * matrix->height; i++) {
        matrix->elements[i] = value;
    }
}

void printMatrix(const Matrix* matrix) {
    for (int i = 0; i < matrix->height; i++) {
        for (int j = 0; j < matrix->width; j++) {
            printf("%f ", matrix->elements[i * matrix->width + j]);
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

    free(A.elements);
    free(B.elements);
    free(C.elements);
    return 0;
}
