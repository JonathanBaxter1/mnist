#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

// Hyperparameters
#define NUM_EXAMPLES 60000
#define BATCH_SIZE 60
#define BATCHES_PER_EPOCH NUM_EXAMPLES/BATCH_SIZE
#define NUM_LAYERS 4
#define NUM_INPUTS 784
#define NUM_OUTPUTS 10
const size_t layers[NUM_LAYERS] = {NUM_INPUTS, 128, 64, NUM_OUTPUTS};
#define LEARNING_RATE 0.05
#define NUM_EPOCHS 30

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
	double **trainingInputs = malloc(NUM_EXAMPLES*sizeof(*trainingInputs));
	double **trainingOutputs = malloc(NUM_EXAMPLES*sizeof(*trainingOutputs));
	if (trainingInputs == NULL || trainingOutputs == NULL) error("Memory");
	for (size_t example = 0; example < NUM_EXAMPLES; example++) {
		trainingInputs[example] = malloc(NUM_INPUTS*sizeof(*trainingInputs[example]));
		trainingOutputs[example] = malloc(NUM_OUTPUTS*sizeof(*trainingOutputs[example]));
		if (trainingInputs[example] == NULL || trainingOutputs[example] == NULL) error("Memory");
		clearParameters(trainingInputs[example], NUM_INPUTS);
		clearParameters(trainingOutputs[example], NUM_OUTPUTS);
	}

	uint8_t image[784] = {0};
	uint8_t imageLabel;
	for (size_t example = 0; example < NUM_EXAMPLES; example++) {
		fseek(trainingImages, 16 + 784*example, SEEK_SET);
		fread(image, 1, 784, trainingImages);
		fseek(trainingLabels, 8 + example, SEEK_SET);
		fread(&imageLabel, 1, 1, trainingLabels);

		for (size_t i = 0; i < NUM_INPUTS; i++) {
			trainingInputs[example][i] = (double)image[i]/255.0;
		}
		for (size_t i = 0; i < NUM_OUTPUTS; i++) {
			if (i == imageLabel) {
				trainingOutputs[example][i] = 1.0;
			} else {
				trainingOutputs[example][i] = 0.0;
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
	for (size_t batch = 0; batch < BATCHES_PER_EPOCH; batch++) {

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
		uint32_t numCorrect = 0;

		// Iterate through training examples
		for (size_t exampleNum = 0; exampleNum < BATCH_SIZE; exampleNum++) {

			// Copy inputs to first layer
			size_t imageIndex = imageIndices[batch*BATCH_SIZE + exampleNum];
			for (size_t i = 0; i < NUM_INPUTS; i++) {
				activations[0][i] = trainingInputs[imageIndex][i];
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

			// Calculate Accuracy
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
			if (predictedNum == correctNum) numCorrect++;

			// Calculate Cost Gradient (Backpropagation)
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
		}
		cost /= (BATCH_SIZE*NUM_OUTPUTS);

		// Adjust Parameters
		for (size_t layer = 0; layer < NUM_LAYERS-1; layer++) {
			size_t numNeurons = layers[layer + 1];
			size_t numWeights = layers[layer];
			adjustParameters(biases[layer], biasGradients[layer], numNeurons);
			for (size_t neuron = 0; neuron < numNeurons; neuron++) {
				adjustParameters(weights[layer][neuron], weightGradients[layer][neuron], numWeights);
			}
		}

		double accuracy = (double)numCorrect/BATCH_SIZE*100.0;
		printf("Batch #%zu | Cost = %f | Accuracy = %.3f%%\n", batch, cost, accuracy);
		printf("-------------------------------------------------\n");
	}
	printf("+++++++++++++++++++++++++++++++++++++++++++++++++\n");
	printf("Epoch #%zu\n", epoch);
	printf("+++++++++++++++++++++++++++++++++++++++++++++++++\n");
	printf("-------------------------------------------------\n");
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
