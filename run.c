#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#define NUM_EXAMPLES 10000
#define NUM_INPUTS 784
#define NUM_OUTPUTS 10
#define RELU_NEGATIVE_SLOPE 0.01

/* ------------------- Helper Functions ------------------- */

// Print an error to the console then exit
void error(const char* string)
{
	printf("ERROR: %s\n", string);
	exit(EXIT_FAILURE);
}

// Find weighted sum
double weightedSum(double inputs[], double weights[], double bias, size_t numInputs)
{
        double sum = 0;
        double sum1 = 0;
        double sum2 = 0;
        double sum3 = 0;
        for (size_t i = 0; i < numInputs; i+=4) {
                sum += weights[i]*inputs[i];
                sum1 += weights[i+1]*inputs[i+1];
                sum2 += weights[i+2]*inputs[i+2];
                sum3 += weights[i+3]*inputs[i+3];
        }
        sum += sum1 + sum2 + sum3 + bias;

        return sum;
}

// Calculate Activations
void calculateActivations(double **activations, double **biases, double ***weights, size_t *layers, size_t numLayers)
{
        for (size_t layer = 0; layer < numLayers-1; layer++) {
                size_t numNeurons = layers[layer + 1];
                size_t numWeights = layers[layer];
                for (size_t neuron = 0; neuron < numNeurons; neuron++) {
                        double z = weightedSum(activations[layer], weights[layer][neuron], biases[layer][neuron], numWeights);
                        double activation;
                        if (layer == numLayers-2) {
                                activation = z; // softmax will occur later
                        } else {
                                activation = z > 0 ? z : z*RELU_NEGATIVE_SLOPE;
                        }
                        activations[layer + 1][neuron] = activation;
                }
        }

        // Softmax on Output Layer
        double *out = activations[numLayers-1];
        double max = out[0];
        for (size_t i = 1; i < NUM_OUTPUTS; i++) {
                max = out[i] > max ? out[i] : max;
        }
        double sum = 0.0;
        for (size_t i = 0; i < NUM_OUTPUTS; i++) {
                out[i] = exp(out[i] - max);
                sum += out[i];
        }
        for (size_t i = 0; i < NUM_OUTPUTS; i++) {
                out[i] /= sum;
        }
}

/* ------------------- Main ------------------- */

int main()
{
	// Import Test Data
	FILE *testImages = fopen("t10k-images-idx3-ubyte", "rb");
	FILE *testLabels = fopen("t10k-labels-idx1-ubyte", "rb");

	// Check Files
	if (testImages == NULL || testLabels == NULL) error("Error reading MNIST data");

	// Read Test Images
	double **testInputs = malloc(NUM_EXAMPLES*sizeof(*testInputs));
	double **testOutputs = malloc(NUM_EXAMPLES*sizeof(*testOutputs));
	if (testInputs == NULL || testOutputs == NULL) error("Memory");
	for (size_t example = 0; example < NUM_EXAMPLES; example++) {
		testInputs[example] = malloc(NUM_INPUTS*sizeof(*testInputs[example]));
		testOutputs[example] = malloc(NUM_OUTPUTS*sizeof(*testOutputs[example]));
		if (testInputs[example] == NULL || testOutputs[example] == NULL) error("Memory");
	}

	uint8_t image[784] = {0};
	uint8_t imageLabel;
	for (size_t example = 0; example < NUM_EXAMPLES; example++) {
		fseek(testImages, 16 + 784*example, SEEK_SET);
		fread(image, 1, 784, testImages);
		fseek(testLabels, 8 + example, SEEK_SET);
		fread(&imageLabel, 1, 1, testLabels);

		for (size_t i = 0; i < NUM_INPUTS; i++) {
			testInputs[example][i] = (double)image[i]/255.0;
		}
		for (size_t i = 0; i < NUM_OUTPUTS; i++) {
			if (i == imageLabel) {
				testOutputs[example][i] = 1.0;
			} else {
				testOutputs[example][i] = 0.0;
			}
		}
	}
	fclose(testImages);
	fclose(testLabels);

	FILE *modelFile = fopen("mnist.nn", "rb");

	// Header
	uint8_t numLayers;
	fread(&numLayers, sizeof(uint8_t), 1, modelFile);
	size_t *layers = malloc(sizeof(size_t)*numLayers);
	for (size_t layer = 0; layer < numLayers; layer++) {
		uint16_t numNeurons;
		fread(&numNeurons, sizeof(numNeurons), 1, modelFile);
		layers[layer] = (size_t)numNeurons;
	}

	// Biases
	double **biases = malloc((numLayers - 1)*sizeof(*biases));
	if (biases == NULL) error("Memory");
	for (size_t layer = 0; layer < numLayers-1; layer++) {
		size_t numNeurons = layers[layer + 1];
		biases[layer] = malloc(numNeurons*sizeof(*biases[layer]));
		if (biases[layer] == NULL) error("Memory");
		fread(biases[layer], sizeof(double), numNeurons, modelFile);
	}

	// Weights
	double ***weights = malloc((numLayers - 1)*sizeof(*weights));
	if (weights == NULL) error("Memory");
	for (size_t layer = 0; layer < numLayers-1; layer++) {
		size_t numNeurons = layers[layer + 1];
		size_t numWeights = layers[layer];
		weights[layer] = malloc(numNeurons*sizeof(*weights[layer]));
		if (weights[layer] == NULL) error("Memory");
		for (size_t neuron = 0; neuron < numNeurons; neuron++) {
			weights[layer][neuron] = malloc(numWeights*sizeof(*weights[layer][neuron]));
			if (weights[layer][neuron] == NULL) error("Memory");
			fread(weights[layer][neuron], sizeof(double), numWeights, modelFile);
		}
	}
	fclose(modelFile);


	// Activations
	double **activations = malloc(numLayers*sizeof(*activations));
	if (activations == NULL) error("Memory");
	for (size_t layer = 0; layer < numLayers; layer++) {
		size_t numNeurons = layers[layer];
		activations[layer] = malloc(numNeurons*sizeof(*activations[layer]));
		if (activations[layer] == NULL) error("Memory");
	}

	// Copy inputs to first layer
	size_t imageIndex = 4;
	for (size_t i = 0; i < NUM_INPUTS; i++) {
		activations[0][i] = testInputs[imageIndex][i];
	}

	// Calculate Activations
	calculateActivations(activations, biases, weights, layers, numLayers);

	// Display Results
	uint32_t correctNum;
	for (size_t i = 0; i < NUM_OUTPUTS; i++) {
		if (testOutputs[imageIndex][i] == 1.0) correctNum = i;
	}

	char *grayscale = "-.:-=+*#%@";
	for (size_t y = 0; y < 28; y++) {
		for (size_t x = 0; x < 28; x++) {
			double pixelValue = activations[0][y*28 + x];
			char character = grayscale[(size_t)(pixelValue*9.999)];
			printf("%c%c", character, character);
		}
		if (y == 0) {
			printf("\tCorrect Answer: %d\n", correctNum);
		} else if (y == 1) {
			printf("\tNum:\tConfidence\n");
		} else if (y-2 < NUM_OUTPUTS) {
			size_t i = y-2;
			printf("\t%zu:\t%.3f%%\n", i, 100.0*activations[numLayers-1][i]);
		} else {
			printf("\n");
		}
	}

	// Free Test Input and Output Arrays
	for (size_t example = 0; example < NUM_EXAMPLES; example++) {
		free(testInputs[example]);
		free(testOutputs[example]);
	}
	free(testInputs);
	free(testOutputs);

	// Free Activation Array
	for (size_t layer = 0; layer < numLayers; layer++) {
		free(activations[layer]);
	}
	free(activations);

	// Free Bias Array
	for (size_t layer = 0; layer < numLayers-1; layer++) {
		free(biases[layer]);
	}
	free(biases);

	// Free Weight Array
	for (size_t layer = 0; layer < numLayers-1; layer++) {
		size_t numNeurons = layers[layer + 1];
		for (size_t neuron = 0; neuron < numNeurons; neuron++) {
			free(weights[layer][neuron]);
		}
		free(weights[layer]);
	}
	free(weights);

	return 0;
}
