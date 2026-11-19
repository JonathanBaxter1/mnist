#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

// Hyperparameters
#define NUM_EXAMPLES 60000
#define NUM_TEST_EXAMPLES 10000
#define BATCH_SIZE 60
#define BATCHES_PER_EPOCH (NUM_EXAMPLES/BATCH_SIZE)
#define NUM_LAYERS 4
#define NUM_INPUTS 784
#define NUM_OUTPUTS 10
const size_t layers[NUM_LAYERS] = {NUM_INPUTS, 128, 64, NUM_OUTPUTS};
#define LEARNING_RATE 0.5
#define NUM_EPOCHS 30

/* ------------------- Helper Functions ------------------- */

// Set array of doubles to 0.0
void clearParameters(double parameters[], size_t numParameters)
{
	for (size_t i = 0; i < numParameters; i++) {
		parameters[i] = 0.0;
	}
}

// Set gradient elements to 0.0
void resetGradients(double **biasGradients, double ***weightGradients)
{
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
}

// Use Xavier Initilization on Weights
void initWeightsXavier(double *weights, size_t numWeights, size_t fanIn, size_t fanOut)
{
	double x = sqrt(6.0/(fanIn + fanOut));
	for (size_t i = 0; i < numWeights; i++) {
		weights[i] = x*(((double)rand()/RAND_MAX)*2.0 - 1.0);
	}
}

// Print an error to the console then exit
void error(const char* string)
{
	printf("ERROR: %s\n", string);
	exit(EXIT_FAILURE);
}

// Adjust Weights/Biases according to the gradient
void adjustParameters(double parameters[], double gradients[], size_t numParameters)
{
	for (size_t i = 0; i < numParameters; i++) {
		double gradient = gradients[i]/BATCH_SIZE;
		parameters[i] -= LEARNING_RATE*gradient;
	}
}

// Perform sigmoid activation function
double sigmoid(double inputs[], double weights[], double bias, size_t numInputs)
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
	sum += sum1 + sum2 + sum3;

	return 1.0/(1.0 + exp(-(sum + bias)));
}

// Calculate Activations
void calculateActivations(double **activations, double **biases, double ***weights)
{
	for (size_t layer = 0; layer < NUM_LAYERS-1; layer++) {
		size_t numNeurons = layers[layer + 1];
		size_t numWeights = layers[layer];
		for (size_t neuron = 0; neuron < numNeurons; neuron++) {
			double activation = sigmoid(activations[layer], weights[layer][neuron], biases[layer][neuron], numWeights);
			activations[layer + 1][neuron] = activation;
		}
	}
}

// For calculating accuracy
bool checkOutput(double **activations, double **trainingOutputs, size_t imageIndex)
{
	uint32_t correctNum = 0;
	uint32_t predictedNum = 0;
	double maxConfidence = 0.0;
	for (size_t i = 0; i < NUM_OUTPUTS; i++) {
		if (trainingOutputs[imageIndex][i] == 1.0) {
			correctNum = i;
			break;
		}
	}
	for (size_t i = 0; i < NUM_OUTPUTS; i++) {
		double confidence = activations[NUM_LAYERS-1][i];
		if (confidence > maxConfidence) {
			maxConfidence = confidence;
			predictedNum = i;
		}
	}
	return predictedNum == correctNum;
}

// Calculate Cost Gradient (Backpropagation)
double backpropogate(double **activations, double **dCdAs, double ***weights, double **biasGradients, double ***weightGradients, double **trainingOutputs, size_t imageIndex)
{
	double cost = 0.0;
	for (size_t layer = NUM_LAYERS-1; layer > 0; layer--) {
		size_t numNeurons = layers[layer];
		size_t numWeights = layers[layer-1];
		for (size_t neuron = 0; neuron < numNeurons; neuron++) {
			double dCdA;
			if (layer == NUM_LAYERS-1) {
				double activation = activations[layer][neuron];
				double goal = trainingOutputs[imageIndex][neuron];
				double diff = activation - goal;
				dCdA = 2.0*diff;
				cost += diff*diff;
			} else {
				dCdA = 0.0;
				size_t numConnections = layers[layer+1];
				for (size_t i = 0; i < numConnections; i++) {
					double activation = activations[layer+1][i];
					double dZdAm1 = weights[layer][i][neuron];
					double dAdZ = activation*(1 - activation);
					double dCdAold = dCdAs[layer+1][i];
					dCdA += dZdAm1*dAdZ*dCdAold;
				}
			}
			dCdAs[layer][neuron] = dCdA;
			double activation = activations[layer][neuron];
			double dAdZ = activation*(1 - activation);
			double dZdB = 1.0;
			double dCdB = dCdA*dAdZ*dZdB;
			biasGradients[layer-1][neuron] += dCdB;
			for (size_t i = 0; i < numWeights; i++) {
				double lastActivation = activations[layer-1][i];
				double dZdW = lastActivation;
				double dCdW = dCdA*dAdZ*dZdW;
				weightGradients[layer-1][neuron][i] += dCdW;
			}
		}
	}
	return cost;
}

void adjustAllParameters(double **biases, double **biasGradients, double ***weights, double ***weightGradients)
{
	for (size_t layer = 0; layer < NUM_LAYERS-1; layer++) {
		size_t numNeurons = layers[layer + 1];
		size_t numWeights = layers[layer];
		adjustParameters(biases[layer], biasGradients[layer], numNeurons);
		for (size_t neuron = 0; neuron < numNeurons; neuron++) {
			adjustParameters(weights[layer][neuron], weightGradients[layer][neuron], numWeights);
		}
	}
}


/* ------------------- Main ------------------- */

int main()
{
	// Check Layers
	for (size_t i = 0; i < NUM_LAYERS-1; i++) {
		if (layers[i]%4 != 0) error("All layers execpt output must have multiple of 4 neurons");
	}

	// Import Training/Test Data
	FILE *trainingImages = fopen("train-images-idx3-ubyte", "rb");
	FILE *trainingLabels = fopen("train-labels-idx1-ubyte", "rb");
	FILE *testImages = fopen("t10k-images-idx3-ubyte", "rb");
	FILE *testLabels = fopen("t10k-labels-idx1-ubyte", "rb");

	// Check Files
	if (trainingImages == NULL || trainingLabels == NULL || testImages == NULL || testLabels == NULL) error("Error reading MNIST data\n");

	// Shuffle Images
	size_t imageIndices[NUM_EXAMPLES];
	for (size_t i = 0; i < NUM_EXAMPLES; i++) {
		imageIndices[i] = i;
	}
	for (size_t i = NUM_EXAMPLES-1; i > 0; i--) {
		size_t randIndex = (size_t)((i + 1)*((double)rand()/(RAND_MAX + 1.0)));
		size_t tempIndex = imageIndices[i];
		imageIndices[i] = imageIndices[randIndex];
		imageIndices[randIndex] = tempIndex;
	}


	// Read Training Images
	uint8_t imageLabel;
	uint8_t **trainingInputs = malloc(NUM_EXAMPLES*sizeof(*trainingInputs));
	double **trainingOutputs = malloc(NUM_EXAMPLES*sizeof(*trainingOutputs));
	if (trainingInputs == NULL || trainingOutputs == NULL) error("Memory");
	for (size_t example = 0; example < NUM_EXAMPLES; example++) {
		trainingInputs[example] = malloc(NUM_INPUTS*sizeof(*trainingInputs[example]));
		trainingOutputs[example] = malloc(NUM_OUTPUTS*sizeof(*trainingOutputs[example]));
		if (trainingInputs[example] == NULL || trainingOutputs[example] == NULL) error("Memory");
	}
	for (size_t example = 0; example < NUM_EXAMPLES; example++) {
		fseek(trainingImages, 16 + 784*example, SEEK_SET);
		fread(trainingInputs[example], 1, 784, trainingImages);
		fseek(trainingLabels, 8 + example, SEEK_SET);
		fread(&imageLabel, 1, 1, trainingLabels);

		for (size_t i = 0; i < NUM_OUTPUTS; i++) {
			if (i == imageLabel) {
				trainingOutputs[example][i] = 1.0;
			} else {
				trainingOutputs[example][i] = 0.0;
			}
		}
	}
	// Read Test Images
	uint8_t **testInputs = malloc(NUM_EXAMPLES*sizeof(*testInputs));
	double **testOutputs = malloc(NUM_EXAMPLES*sizeof(*testOutputs));
	if (testInputs == NULL || testOutputs == NULL) error("Memory");
	for (size_t example = 0; example < NUM_EXAMPLES; example++) {
		testInputs[example] = malloc(NUM_INPUTS*sizeof(*testInputs[example]));
		testOutputs[example] = malloc(NUM_OUTPUTS*sizeof(*testOutputs[example]));
		if (testInputs[example] == NULL || testOutputs[example] == NULL) error("Memory");
	}
	for (size_t example = 0; example < NUM_EXAMPLES; example++) {
		fseek(testImages, 16 + 784*example, SEEK_SET);
		fread(testInputs[example], 1, 784, testImages);
		fseek(testLabels, 8 + example, SEEK_SET);
		fread(&imageLabel, 1, 1, testLabels);

		for (size_t i = 0; i < NUM_OUTPUTS; i++) {
			if (i == imageLabel) {
				testOutputs[example][i] = 1.0;
			} else {
				testOutputs[example][i] = 0.0;
			}
		}
	}

	// Close Files
	fclose(trainingImages);
	fclose(trainingLabels);
	fclose(testImages);
	fclose(testLabels);


	// Activations and Activation Gradients
	double **activations = malloc(NUM_LAYERS*sizeof(*activations));
	double **dCdAs = malloc(NUM_LAYERS*sizeof(*dCdAs));
	if (activations == NULL || dCdAs == NULL) error("Memory");
	for (size_t layer = 0; layer < NUM_LAYERS; layer++) {
		size_t numNeurons = layers[layer];
		activations[layer] = malloc(numNeurons*sizeof(*activations[layer]));
		dCdAs[layer] = malloc(numNeurons*sizeof(*dCdAs[layer]));
		if (activations[layer] == NULL || dCdAs[layer] == NULL) error("Memory");
		clearParameters(activations[layer], numNeurons);
		clearParameters(dCdAs[layer], numNeurons);
	}

	// Biases and Bias Gradients
	double **biases = malloc((NUM_LAYERS - 1)*sizeof(*biases));
	double **biasGradients = malloc((NUM_LAYERS - 1)*sizeof(*biasGradients));
	if (biases == NULL || biasGradients == NULL) error("Memory");
	for (size_t layer = 0; layer < NUM_LAYERS-1; layer++) {
		size_t numNeurons = layers[layer + 1];
		biases[layer] = malloc(numNeurons*sizeof(*biases[layer]));
		biasGradients[layer] = malloc(numNeurons*sizeof(*biasGradients[layer]));
		if (biases[layer] == NULL || biasGradients[layer] == NULL) error("Memory");
		clearParameters(biases[layer], numNeurons);
		clearParameters(biasGradients[layer], numNeurons);
	}

	// Weights and Weight Gradients
	double ***weights = malloc((NUM_LAYERS - 1)*sizeof(*weights));
	double ***weightGradients = malloc((NUM_LAYERS - 1)*sizeof(*weightGradients));
	if (weights == NULL || weightGradients == NULL) error("Memory");
	for (size_t layer = 0; layer < NUM_LAYERS-1; layer++) {
		size_t numNeurons = layers[layer + 1];
		size_t numWeights = layers[layer];
		weights[layer] = malloc(numNeurons*sizeof(*weights[layer]));
		weightGradients[layer] = malloc(numNeurons*sizeof(*weightGradients[layer]));
		if (weights[layer] == NULL || weightGradients[layer] == NULL) error("Memory");
		for (size_t neuron = 0; neuron < numNeurons; neuron++) {
			weights[layer][neuron] = malloc(numWeights*sizeof(*weights[layer][neuron]));
			weightGradients[layer][neuron] = malloc(numWeights*sizeof(*weightGradients[layer][neuron]));
			if (weights[layer][neuron] == NULL || weightGradients[layer][neuron] == NULL) error("Memory");
			size_t fanIn = numWeights;
			size_t fanOut = numNeurons;
			initWeightsXavier(weights[layer][neuron], numWeights, fanIn, fanOut);
			clearParameters(weightGradients[layer][neuron], numWeights);
		}
	}

	// Training Loop
	for (size_t epoch = 0; epoch < NUM_EPOCHS; epoch++) {

		double cost = 0.0;
		uint32_t numCorrect = 0;

		for (size_t batch = 0; batch < BATCHES_PER_EPOCH; batch++) {

			resetGradients(biasGradients, weightGradients);

			// Iterate through training examples
			for (size_t exampleNum = 0; exampleNum < BATCH_SIZE; exampleNum++) {

				// Copy inputs to first layer
				size_t imageIndex = imageIndices[batch*BATCH_SIZE + exampleNum];
				for (size_t i = 0; i < NUM_INPUTS; i++) {
					activations[0][i] = (double)trainingInputs[imageIndex][i]/255.0;
				}

				// Feedforward
				calculateActivations(activations, biases, weights);

				// Backpropogate
				cost += backpropogate(activations, dCdAs, weights, biasGradients, weightGradients, trainingOutputs, imageIndex);
			}

			adjustAllParameters(biases, biasGradients, weights, weightGradients);

			if (batch%20 == 0) {
				printf("-");
				fflush(stdout);
			}
		}

		// Find Accuracy for Test Examples
		for (size_t exampleNum = 0; exampleNum < NUM_TEST_EXAMPLES; exampleNum++) {
			// Copy inputs to first layer
			for (size_t i = 0; i < NUM_INPUTS; i++) {
				activations[0][i] = (double)testInputs[exampleNum][i]/255.0;
			}

			// Feedforward
			calculateActivations(activations, biases, weights);
			numCorrect += (uint32_t)checkOutput(activations, testOutputs, exampleNum);
		}

		cost /= (NUM_EXAMPLES*NUM_OUTPUTS);
		double accuracy = (double)numCorrect/NUM_TEST_EXAMPLES*100.0;
		printf("\n");
		printf("Epoch #%zu | Cost = %f | Accuracy = %.3f%%\n", epoch, cost, accuracy);
	}

	// Output Neural Network to File
	FILE *outputFile = fopen("mnist.nn", "wb");
	if (outputFile == NULL) error("Could not create output file");
	uint8_t numLayers = NUM_LAYERS;
	fwrite(&numLayers, sizeof(numLayers), 1, outputFile);

	// Header
	for (size_t layer = 0; layer < NUM_LAYERS; layer++) {
		uint16_t numNeurons = (uint16_t)layers[layer];
		fwrite(&numNeurons, sizeof(numNeurons), 1, outputFile);
	}

	// Biases
	for (size_t layer = 0; layer < NUM_LAYERS-1; layer++) {
		size_t numNeurons = layers[layer + 1];
		fwrite(biases[layer], sizeof(double), numNeurons, outputFile);
	}

	// Weights
	for (size_t layer = 0; layer < NUM_LAYERS-1; layer++) {
		size_t numNeurons = layers[layer + 1];
		size_t numWeights = layers[layer];
		for (size_t neuron = 0; neuron < numNeurons; neuron++) {
			fwrite(weights[layer][neuron], sizeof(double), numWeights, outputFile);
		}
	}
	fclose(outputFile);

	// Free Training Input and Output Arrays
	for (size_t example = 0; example < NUM_EXAMPLES; example++) {
		free(trainingInputs[example]);
		free(trainingOutputs[example]);
	}
	free(trainingInputs);
	free(trainingOutputs);

	// Free Activation and Activation Gradient Arrays
	for (size_t layer = 0; layer < NUM_LAYERS; layer++) {
		free(activations[layer]);
		free(dCdAs[layer]);
	}
	free(activations);
	free(dCdAs);

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
