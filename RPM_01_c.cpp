// COdigo para inferencia de valores de respiraciones por minuto, RR/RPM
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// NUmero de ejemplos y atributos
#define NUM_SAMPLES 50
#define NUM_FEATURES 4
#define LEARNING_RATE 0.0001  // modificar dependiendo la magnitud de los valores
#define ITERATIONS 40

// FunciOn para leer los datos
void read_data(double X[NUM_SAMPLES][NUM_FEATURES], double Y[NUM_SAMPLES], const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("No se pudo abrir el archivo");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < NUM_SAMPLES; i++) {
        fscanf(file, "%lf %lf %lf %lf %lf", &X[i][0], &X[i][1], &X[i][2], &X[i][3], &Y[i]);
    }
    fclose(file);
}

// FunciOn para predecir
double predict(double X[NUM_FEATURES], double weights[NUM_FEATURES + 1]) {
    double result = weights[0]; // Intercepto
    for (int i = 0; i < NUM_FEATURES; i++) {
        result += weights[i + 1] * X[i];
    }
    return result;
}

// Gradiente descendente
void train(double X[NUM_SAMPLES][NUM_FEATURES], double Y[NUM_SAMPLES], double weights[NUM_FEATURES + 1]) {
    for (int iter = 0; iter < ITERATIONS; iter++) {
        double gradients[NUM_FEATURES + 1] = {0};

        // CAlculo de gradientes
        for (int i = 0; i < NUM_SAMPLES; i++) {
            double y_pred = predict(X[i], weights);
            double error = y_pred - Y[i];
            gradients[0] += error; 
            for (int j = 0; j < NUM_FEATURES; j++) {
                gradients[j + 1] += error * X[i][j];
            }
        }

        // Actualizacion de pesos
        for (int j = 0; j <= NUM_FEATURES; j++) {
            weights[j] -= LEARNING_RATE * gradients[j] / NUM_SAMPLES;
        }

        // Error actual
        if (iter % 100 == 0) {
            double total_error = 0;
            for (int i = 0; i < NUM_SAMPLES; i++) {
                double y_pred = predict(X[i], weights);
                total_error += pow(y_pred - Y[i], 2);
            }
            printf("Iteration %d, Error: %.4f\n", iter, total_error / NUM_SAMPLES);
        }
    }
}

int main() {
    double X[NUM_SAMPLES][NUM_FEATURES];
    double Y[NUM_SAMPLES];
    double weights[NUM_FEATURES + 1] = {0}; // Pesos iniciales

    // Lectura de datos con valores de HR, Age, Male, Female y RR, en este orden estAn en data
    read_data(X, Y, "data.txt");

    // Entrenamiento modelo
    train(X, Y, weights);

    // ImpresiOn de pesos finales, SON LOS QUE PASO AL CODIGO DE INFERENCIA
    printf("Final weights:\n");
    for (int i = 0; i <= NUM_FEATURES; i++) {
        printf("w%d: %.4f\n", i, weights[i]);
    }

    // Probar predicciOn
    double test_sample[NUM_FEATURES] = {73.0, 26.0, 1.0, 0.0};
    double test_sample2[NUM_FEATURES] = {73.0, 56.0, 0.0, 1.0};
    double prediction = predict(test_sample, weights);
    double prediction2 = predict(test_sample2, weights);
    // printf("PredicciOn para mi [73.0, 26.0, 1.0, 0.0]: %.4f\n", prediction);
    printf("Prediction for someone with [73.0 (HR), 56.0 (years), 0.0, 1.0 (male)]: %.4f rpm\n ", prediction2);

    return 0;
}


