// Code for inferring values ​​of breaths per minute, RR/RPM
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Number of examples and attributes
#define NUM_SAMPLES 50
#define NUM_FEATURES 4
#define LEARNING_RATE 0.0001  // modificar dependiendo la magnitud de los valores
#define ITERATIONS 40

// Reading data function
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

// Prediction function
double predict(double X[NUM_FEATURES], double weights[NUM_FEATURES + 1]) {
    double result = weights[0]; // Intercept
    for (int i = 0; i < NUM_FEATURES; i++) {
        result += weights[i + 1] * X[i];
    }
    return result;
}

// Gradiente-descendente algorithm
void train(double X[NUM_SAMPLES][NUM_FEATURES], double Y[NUM_SAMPLES], double weights[NUM_FEATURES + 1]) {
    for (int iter = 0; iter < ITERATIONS; iter++) {
        double gradients[NUM_FEATURES + 1] = {0};

        // Gradient calculation
        for (int i = 0; i < NUM_SAMPLES; i++) {
            double y_pred = predict(X[i], weights);
            double error = y_pred - Y[i];
            gradients[0] += error; 
            for (int j = 0; j < NUM_FEATURES; j++) {
                gradients[j + 1] += error * X[i][j];
            }
        }

        // Weight update
        for (int j = 0; j <= NUM_FEATURES; j++) {
            weights[j] -= LEARNING_RATE * gradients[j] / NUM_SAMPLES;
        }

        // Actual error
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
    double weights[NUM_FEATURES + 1] = {0}; // Initial weights

    // Reading data with values ​​of HR, Age, Male, Female and RR, in this order are in data
    read_data(X, Y, "data.txt");

    train(X, Y, weights);

    // Final weight printing, THESE ARE THE ONES I PASS TO THE INFERENCE CODE
    printf("Final weights:\n");
    for (int i = 0; i <= NUM_FEATURES; i++) {
        printf("w%d: %.4f\n", i, weights[i]);
    }

    // Test prediction
    double test_sample[NUM_FEATURES] = {73.0, 26.0, 1.0, 0.0};
    double test_sample2[NUM_FEATURES] = {73.0, 56.0, 0.0, 1.0};
    double prediction = predict(test_sample, weights);
    double prediction2 = predict(test_sample2, weights);
    // printf("Prediction for me [73.0, 26.0, 1.0, 0.0]: %.4f\n", prediction);
    printf("Prediction for someone with [73.0 (HR), 56.0 (years), 0.0, 1.0 (male)]: %.4f rpm\n ", prediction2);

    return 0;
}


