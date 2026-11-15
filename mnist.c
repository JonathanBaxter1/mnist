#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

// Hyperparameters
#define NUM_EXAMPLES 8
#define NUM_LAYERS 4
const size_t layers[NUM_LAYERS] = {3, 4, 4, 2};
const NUM_INPUTS layers[0];
const NUM_OUTPUTS layers[NUM_LAYERS-1];
#define LEARNING_RATE 10.0
#define NUM_EPOCHS 10000

/* ------------------- Helper Functions ------------------- */

// Use Xavier Initilization on Weights
void initWeightsXavier(double *weights, size_t numWeights, size_t fanIn, size_t fanOut)
{
	double x = sqrt(6.0/(fanIn + fanOut));
	for (size_t i = 0; i < numWeights; i++) {
		weights[i] = x*(((double)rand()/RAND_MAX)*2.0 - 1.0);
	}
}

// Print an error to the console then exit
void error(char* string)
{
	printf("ERROR: %s\n", string);
	exit(-1);
}

// Adjust Weights/Biases according to the gradient
void adjustParameters(double parameters[], double gradients[], size_t numParameters)
{
	for (size_t i = 0; i < numParameters; i++) {
		double gradient = gradients[i]/NUM_EXAMPLES;
		parameters[i] -= LEARNING_RATE*gradient;
	}
}

// Set array of doubles to 0.0
void clearParameters(double parameters[], size_t numParameters)
{
	for (size_t i = 0; i < numParameters; i++) {
		parameters[i] = 0.0;
	}
}

// Perform sigmoid activation function
double sigmoid(double inputs[], double weights[], double bias, size_t numInputs)
{
	double sum = 0;
	for (size_t i = 0; i < numInputs; i++) {
		sum += weights[i]*inputs[i];
	}
	return 1.0/(1.0 + exp(-(sum + bias)));
}

/* ------------------- Main ------------------- */

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

	// Temporary training data
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

	// Temporary Activations
	double **activations = malloc(NUM_LAYERS*sizeof(*activations));
	if (activations == NULL) error("Memory");
	for (size_t layer = 0; layer < NUM_LAYERS; layer++) {
		size_t numNeurons = layers[layer];
		activations[layer] = malloc(numNeurons*sizeof(*activations[layer]));
		if (activations[layer] == NULL) error("Memory");
		clearParameters(activations[layer], numNeurons);
	}

	// Biases
	double **biases = malloc((NUM_LAYERS - 1)*sizeof(*biases));
	if (biases == NULL) error("Memory");
	for (size_t layer = 0; layer < NUM_LAYERS-1; layer++) {
		size_t numNeurons = layers[layer + 1];
		biases[layer] = malloc(numNeurons*sizeof(*biases[layer]));
		if (biases[layer] == NULL) error("Memory");
		clearParameters(biases[layer], numNeurons);
	}

	// Weights
	double ***weights = malloc((NUM_LAYERS - 1)*sizeof(*weights));
	if (weights == NULL) error("Memory");
	for (size_t layer = 0; layer < NUM_LAYERS-1; layer++) {
		size_t numNeurons = layers[layer + 1];
		size_t numWeights = layers[layer];
		weights[layer] = malloc(numNeurons*sizeof(*weights[layer]));
		if (weights[layer] == NULL) error("Memory");
		for (size_t neuron = 0; neuron < numNeurons; neuron++) {
			weights[layer][neuron] = malloc(numWeights*sizeof(*weights[layer][neuron]));
			if (weights[layer][neuron] == NULL) error("Memory");
			size_t fanIn = numWeights;
			size_t fanOut = numNeurons;
			initWeightsXavier(weights[layer][neuron], numWeights, fanIn, fanOut);
		}
	}

	// Bias Gradients
	double **biasGradients = malloc((NUM_LAYERS - 1)*sizeof(*biasGradients));
	if (biasGradients == NULL) error("Memory");
	for (size_t layer = 0; layer < NUM_LAYERS-1; layer++) {
		size_t numNeurons = layers[layer + 1];
		biasGradients[layer] = malloc(numNeurons*sizeof(*biasGradients[layer]));
		if (biasGradients[layer] == NULL) error("Memory");
		clearParameters(biasGradients[layer], numNeurons);
	}

	// Weight Gradients
	double ***weightGradients = malloc((NUM_LAYERS - 1)*sizeof(*weightGradients));
	if (weightGradients == NULL) error("Memory");
	for (size_t layer = 0; layer < NUM_LAYERS-1; layer++) {
		size_t numNeurons = layers[layer + 1];
		size_t numWeights = layers[layer];
		weightGradients[layer] = malloc(numNeurons*sizeof(*weightGradients[layer]));
		if (weightGradients[layer] == NULL) error("Memory");
		for (size_t neuron = 0; neuron < numNeurons; neuron++) {
			weightGradients[layer][neuron] = malloc(numWeights*sizeof(*weightGradients[layer][neuron]));
			if (weightGradients[layer][neuron] == NULL) error("Memory");
			clearParameters(weightGradients[layer][neuron], numWeights);
		}
	}

	for (size_t epoch = 0; epoch < NUM_EPOCHS; epoch++) {

		// Reset Gradients to 0.0
		for (size_t layer = 0; layer < NUM_LAYERS-1; layer++) {
			size_t numNeurons = layers[layer + 1];
			clearParameters(biasGradients[layer], numNeurons);
		}
		for (size_t layer = 0; layer < NUM_LAYERS-1; layer++) {
			size_t numNeurons = layers[layer + 1];
			size_t numWeights = layers[layer];
			for (size_t neuron = 0; neuron < numNeurons; neuron++) {
				clearParameters(weightGradients[layer][neuron], numWeights);
			}
		}

		double cost = 0.0;

		for (size_t exampleNum = 0; exampleNum < NUM_EXAMPLES; exampleNum++) {

			// Copy inputs to first layer
			for (size_t i = 0; i < NUM_INPUTS; i++) {
				activations[0][i] = trainingInputs[exampleNum][i];
			}

			// Calculate Activations
			for (size_t layer = 0; layer < NUM_LAYERS-1; layer++) {
				size_t numNeurons = layers[layer + 1];
				size_t numWeights = layers[layer];
				for (size_t neuron = 0; neuron < numNeurons; neuron++) {
					double activation = sigmoid(activations[layer], weights[layer][neuron], biases[layer][neuron], numWeights);
					activations[layer + 1][neuron] = activation;
				}
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

	// Free Activation Array
	for (size_t layer = 0; layer < NUM_LAYERS; layer++) {
		free(activations[layer]);
	}
	free(activations);

	// Free Bias and Bias Gradient Arrays
	for (size_t layer = 0; layer < NUM_LAYERS-1; layer++) {
		free(biases[layer]);
		free(biasGradients[layer]);
	}
	free(biases);
	free(biasGradients);

	// Free Weight and Weight Gradient Arrays
	for (size_t layer = 0; layer < NUM_LAYERS-1; layer++) {
		size_t numNeurons = layers[layer + 1];
		for (size_t neuron = 0; neuron < numNeurons; neuron++) {
			free(weights[layer][neuron]);
			free(weightGradients[layer][neuron]);
		}
		free(weights[layer]);
		free(weightGradients[layer]);
	}
	free(weights);
	free(weightGradients);

	return 0;
}
