#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#define NUM_EXAMPLES 8

#define NUM_INPUTS 3
#define NUM_HIDDEN_LAYERS 2
#define NEURONS_PER_HIDDEN_LAYER 4
#define NUM_OUTPUTS 2

#define LEARNING_RATE 10.0
#define NUM_EPOCHS 10000

void adjustParameters(double parameters[], double gradients[], size_t numParameters)
{
	for (size_t i = 0; i < numParameters; i++) {
		double gradient = gradients[i]/NUM_EXAMPLES;
		parameters[i] -= LEARNING_RATE*gradient;
	}
}
void clearParameters(double parameters[], size_t numParameters)
{
	for (size_t i = 0; i < numParameters; i++) {
		parameters[i] = 0.0;
	}
}

double sigmoid(double inputs[], double weights[], double bias, size_t numInputs)
{
	double sum = 0;
	for (size_t i = 0; i < numInputs; i++) {
		sum += weights[i]*inputs[i];
	}
	return 1.0/(1.0 + exp(-(sum + bias)));
}

int main()
{
	// Import Training/Test Data
	FILE *trainingImages = fopen("train-images-idx3-ubyte", "rb");
	FILE *trainingLabels = fopen("train-labels-idx1-ubyte", "rb");
	FILE *testImages = fopen("t10k-images-idx3-ubyte", "rb");
	FILE *testLabels = fopen("t10k-labels-idx1-ubyte", "rb");

	// Check Files
	if (trainingImages == NULL || trainingLabels == NULL || testImages == NULL || testLabels == NULL) {
		printf("Error reading MNIST data\n");
		exit(-1);
	}

	// Read Training Images
	uint8_t image1[784] = {0};
	fseek(trainingImages, 16, SEEK_SET);
	fread(image1, 1, 784, trainingImages);
	uint8_t image1Label;
	fseek(trainingLabels, 8, SEEK_SET);
	fread(&image1Label, 1, 1, trainingLabels);

	// Close Files
	fclose(trainingImages);
	fclose(trainingLabels);
	fclose(testImages);
	fclose(testLabels);

	double trainingInputs[NUM_EXAMPLES][NUM_INPUTS] = {
		// A,   B,   C
		{0.0, 0.0, 0.0},
		{0.0, 0.0, 1.0},
		{0.0, 1.0, 0.0},
		{0.0, 1.0, 1.0},
		{1.0, 0.0, 0.0},
		{1.0, 0.0, 1.0},
		{1.0, 1.0, 0.0},
		{1.0, 1.0, 1.0},
	};
	double trainingOutputs[NUM_EXAMPLES][NUM_OUTPUTS] = {
		// C,   S
		{0.0, 0.0},
		{0.0, 1.0},
		{0.0, 1.0},
		{1.0, 0.0},
		{0.0, 1.0},
		{1.0, 0.0},
		{1.0, 0.0},
		{1.0, 1.0},
	};

	// Temp Activations
	double finalLayerActivations[NUM_OUTPUTS] = {0.0};
	double hiddenLayerActivations[NUM_HIDDEN_LAYERS][NEURONS_PER_HIDDEN_LAYER] = {0.0};

	// Parameters
	double firstLayerBiases[NEURONS_PER_HIDDEN_LAYER] = {0.0};
	double firstLayerWeights[NEURONS_PER_HIDDEN_LAYER][NUM_INPUTS];
	double hiddenLayerBiases[NUM_HIDDEN_LAYERS-1][NEURONS_PER_HIDDEN_LAYER] = {0.0};
	double hiddenLayerWeights[NUM_HIDDEN_LAYERS-1][NEURONS_PER_HIDDEN_LAYER][NEURONS_PER_HIDDEN_LAYER];
	double finalLayerBiases[NUM_OUTPUTS] = {0.0};
	double finalLayerWeights[NUM_OUTPUTS][NEURONS_PER_HIDDEN_LAYER];
	for (size_t i = 0; i < NEURONS_PER_HIDDEN_LAYER; i++) {
		double x = sqrt(6.0/(NUM_INPUTS+NEURONS_PER_HIDDEN_LAYER));
		for (size_t j = 0; j < NUM_INPUTS; j++) {
			firstLayerWeights[i][j] = x*(((double)rand()/RAND_MAX)*2.0-1.0);
		}
		x = sqrt(6.0/(NEURONS_PER_HIDDEN_LAYER*2.0));
		for (size_t j = 0; j < NUM_HIDDEN_LAYERS-1; j++) {
			for (size_t k = 0; k < NEURONS_PER_HIDDEN_LAYER; k++) {
				hiddenLayerWeights[j][i][k] = x*(((double)rand()/RAND_MAX)*2.0-1.0);
			}
		}
		x = sqrt(6.0/(NEURONS_PER_HIDDEN_LAYER+NUM_OUTPUTS));
		for (size_t j = 0; j < NUM_OUTPUTS; j++) {
			finalLayerWeights[j][i] = x*(((double)rand()/RAND_MAX)*2.0-1.0);
		}
	}

	// Gradient
	double firstLayerBiasGradients[NEURONS_PER_HIDDEN_LAYER];
	double firstLayerWeightGradients[NEURONS_PER_HIDDEN_LAYER][NUM_INPUTS];
	double hiddenLayerBiasGradients[NUM_HIDDEN_LAYERS-1][NEURONS_PER_HIDDEN_LAYER];
	double hiddenLayerWeightGradients[NUM_HIDDEN_LAYERS-1][NEURONS_PER_HIDDEN_LAYER][NEURONS_PER_HIDDEN_LAYER];
	double finalLayerBiasGradients[NUM_OUTPUTS];
	double finalLayerWeightGradients[NUM_OUTPUTS][NEURONS_PER_HIDDEN_LAYER];

	// Check Number of Layers
	if (NUM_HIDDEN_LAYERS <= 1) {
		printf("Must have more than 1 hidden layer\n");
		exit(-1);
	}

	for (size_t epoch = 0; epoch < NUM_EPOCHS; epoch++) {

		double cost = 0.0;
		// Reset Gradients to 0.0
		clearParameters(&firstLayerBiasGradients[0], NEURONS_PER_HIDDEN_LAYER);
		clearParameters(&firstLayerWeightGradients[0][0], NEURONS_PER_HIDDEN_LAYER*NUM_INPUTS);
		clearParameters(&hiddenLayerBiasGradients[0][0], (NUM_HIDDEN_LAYERS-1)*NEURONS_PER_HIDDEN_LAYER);
		clearParameters(&hiddenLayerWeightGradients[0][0][0], (NUM_HIDDEN_LAYERS-1)*NEURONS_PER_HIDDEN_LAYER*NEURONS_PER_HIDDEN_LAYER);
		clearParameters(&finalLayerBiasGradients[0], NUM_OUTPUTS);
		clearParameters(&finalLayerWeightGradients[0][0], NUM_OUTPUTS*NEURONS_PER_HIDDEN_LAYER);

		for (size_t exampleNum = 0; exampleNum < NUM_EXAMPLES; exampleNum++) {
			// Calculate Activations

			// First Hidden Layer
			double *inputs = trainingInputs[exampleNum];
			for (size_t neuron = 0; neuron < NEURONS_PER_HIDDEN_LAYER; neuron++) {
				double *weights = firstLayerWeights[neuron];
				double bias = firstLayerBiases[neuron];
				double activation = sigmoid(inputs, weights, bias, NUM_INPUTS);
				hiddenLayerActivations[0][neuron] = activation;
			}

			// Other Hidden Layers
			for (size_t layer = 1; layer < NUM_HIDDEN_LAYERS; layer++) {
				double *inputs = hiddenLayerActivations[layer-1];
				for (size_t neuron = 0; neuron < NEURONS_PER_HIDDEN_LAYER; neuron++) {
					double *weights = hiddenLayerWeights[layer-1][neuron];
					double bias = hiddenLayerBiases[layer-1][neuron];
					double activation = sigmoid(inputs, weights, bias, NEURONS_PER_HIDDEN_LAYER);
					hiddenLayerActivations[layer][neuron] = activation;
				}
			}

			// Final Layer
			inputs = hiddenLayerActivations[NUM_HIDDEN_LAYERS-1];
			for (size_t neuron = 0; neuron < NUM_OUTPUTS; neuron++) {
				double *weights = finalLayerWeights[neuron];
				double bias = finalLayerBiases[neuron];
				double activation = sigmoid(inputs, weights, bias, NEURONS_PER_HIDDEN_LAYER);
				finalLayerActivations[neuron] = activation;
			}

			// Calculate Cost Gradient

			// Final Layer
			double dCdZs[NUM_OUTPUTS] = {0.0};
			for (size_t neuron = 0; neuron < NUM_OUTPUTS; neuron++) {
				double activation = finalLayerActivations[neuron];
				double goal = trainingOutputs[exampleNum][neuron];
				double error = activation - goal;
				cost += error*error;

				double dCdA = 2*error;
				double dAdZ = activation*(1 - activation);
				double dCdZ = dCdA*dAdZ;
				dCdZs[neuron] = dCdZ;

				// Calculate Partial Derivative of Cost wrt Bias
				double dZdB = 1;
				double dCdB = dCdZ*dZdB;
				finalLayerBiasGradients[neuron] += dCdB;

				// Calculate Partial Derivative of Cost wrt Weights
				for (size_t i = 0; i < NEURONS_PER_HIDDEN_LAYER; i++) {
					double lastActivation = hiddenLayerActivations[NUM_HIDDEN_LAYERS-1][i];
					double dZdW = lastActivation;
					double dCdW = dCdZ*dZdW;
					finalLayerWeightGradients[neuron][i] += dCdW;
				}
			}

			// Last Hidden Layer
			double dCdZs2[NEURONS_PER_HIDDEN_LAYER] = {0.0};
			double dCdZs2last[NEURONS_PER_HIDDEN_LAYER] = {0.0};
			for (size_t neuron = 0; neuron < NEURONS_PER_HIDDEN_LAYER; neuron++) {
				double activation = hiddenLayerActivations[NUM_HIDDEN_LAYERS-1][neuron];

				double dCdA = 0.0;
				for (size_t i = 0; i < NUM_OUTPUTS; i++) {
					double dZdA = finalLayerWeights[i][neuron];
					dCdA += dZdA*dCdZs[i];
				}
				double dAdZ = activation*(1 - activation);
				double dCdZ = dCdA*dAdZ;
				dCdZs2[neuron] = dCdZ;

				// Calculate Partial Derivative of Cost wrt Bias
				double dZdB = 1;
				double dCdB = dCdZ*dZdB;
				hiddenLayerBiasGradients[NUM_HIDDEN_LAYERS-2][neuron] += dCdB;

				// Calculate Partial Derivative of Cost wrt Weights
				for (size_t i = 0; i < NEURONS_PER_HIDDEN_LAYER; i++) {
					double lastActivation = hiddenLayerActivations[NUM_HIDDEN_LAYERS-2][i];
					double dZdW = lastActivation;
					double dCdW = dCdZ*dZdW;
					hiddenLayerWeightGradients[NUM_HIDDEN_LAYERS-2][neuron][i] += dCdW;
				}
			}
			for (size_t i = 0; i < NEURONS_PER_HIDDEN_LAYER; i++) {
				dCdZs2last[i] = dCdZs2[i];
			}

			// Middle Hidden Layers
			for (size_t layer = NUM_HIDDEN_LAYERS-2; layer > 0; layer--) {
				for (size_t neuron = 0; neuron < NEURONS_PER_HIDDEN_LAYER; neuron++) {
					double activation = hiddenLayerActivations[layer][neuron];

					double dCdA = 0.0;
					for (size_t i = 0; i < NEURONS_PER_HIDDEN_LAYER; i++) {
						double dZdA = hiddenLayerWeights[layer][i][neuron];
						dCdA += dZdA*dCdZs2last[i];
					}
					double dAdZ = activation*(1 - activation);
					double dCdZ = dCdA*dAdZ;
					dCdZs2[neuron] = dCdZ;

					// Calculate Partial Derivative of Cost wrt Bias
					double dZdB = 1;
					double dCdB = dCdZ*dZdB;
					hiddenLayerBiasGradients[layer-1][neuron] += dCdB;

					// Calculate Partial Derivative of Cost wrt Weights
					for (size_t i = 0; i < NEURONS_PER_HIDDEN_LAYER; i++) {
						double lastActivation = hiddenLayerActivations[layer-1][i];
						double dZdW = lastActivation;
						double dCdW = dCdZ*dZdW;
						hiddenLayerWeightGradients[layer-1][neuron][i] += dCdW;
					}
				}
				for (size_t i = 0; i < NEURONS_PER_HIDDEN_LAYER; i++) {
					dCdZs2last[i] = dCdZs2[i];
				}
			}

			// First Hidden Layer
			double *firstLayerActivations = trainingInputs[exampleNum];
			for (size_t neuron = 0; neuron < NEURONS_PER_HIDDEN_LAYER; neuron++) {
				double activation = hiddenLayerActivations[0][neuron];

				double dCdA = 0.0;
				for (size_t i = 0; i < NEURONS_PER_HIDDEN_LAYER; i++) {
					double dZdA = hiddenLayerWeights[0][i][neuron];
					dCdA += dZdA*dCdZs2last[i];
				}
				double dAdZ = activation*(1 - activation);
				double dCdZ = dCdA*dAdZ;

				// Calculate Partial Derivative of Cost wrt Bias
				double dZdB = 1;
				double dCdB = dCdZ*dZdB;
				firstLayerBiasGradients[neuron] += dCdB;

				// Calculate Partial Derivative of Cost wrt Weights
				for (size_t i = 0; i < NUM_INPUTS; i++) {
					double lastActivation = firstLayerActivations[i];
					double dZdW = lastActivation;
					double dCdW = dCdZ*dZdW;
					firstLayerWeightGradients[neuron][i] += dCdW;
				}
			}
		}
		cost /= NUM_EXAMPLES;

		// Adjust Parameters
		adjustParameters(&firstLayerBiases[0], &firstLayerBiasGradients[0], NEURONS_PER_HIDDEN_LAYER);
		adjustParameters(&firstLayerWeights[0][0], &firstLayerWeightGradients[0][0], NEURONS_PER_HIDDEN_LAYER*NUM_INPUTS);
		adjustParameters(&hiddenLayerBiases[0][0], &hiddenLayerBiasGradients[0][0], (NUM_HIDDEN_LAYERS-1)*NEURONS_PER_HIDDEN_LAYER);
		adjustParameters(&hiddenLayerWeights[0][0][0], &hiddenLayerWeightGradients[0][0][0], (NUM_HIDDEN_LAYERS-1)*NEURONS_PER_HIDDEN_LAYER*NEURONS_PER_HIDDEN_LAYER);
		adjustParameters(&finalLayerBiases[0], &finalLayerBiasGradients[0], NUM_OUTPUTS);
		adjustParameters(&finalLayerWeights[0][0], &finalLayerWeightGradients[0][0], NUM_OUTPUTS*NEURONS_PER_HIDDEN_LAYER);

		printf("Epoch #%zu | cost = %f\n", epoch, cost);
		printf("-------------------------------------------------\n");
	}

	printf("First Layer:\n");
	for (size_t i = 0; i < NEURONS_PER_HIDDEN_LAYER; i++) {
		printf("Neuron #%zu\n", i);
		printf("Bias: %f\n", firstLayerBiases[i]);
		printf("Weights: {");
		for (size_t j = 0; j < NUM_INPUTS; j++) {
			printf("%f", firstLayerWeights[i][j]);
			if (j != NUM_INPUTS-1) printf(", ");
		}
		printf("}\n------------------------------\n");
	}

	printf("Image 1: Label: %d\n", image1Label);
	for (size_t y = 0; y < 28; y++) {
		for (size_t x = 0; x < 28; x++) {
			char *grayscale = " .:-=+*#%@";
			uint8_t pixelValue = image1[y*28 + x];
			char character = grayscale[pixelValue/26];
			printf("%c%c", character, character);
		}
		printf("\n");
	}

	return 0;
}
