#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <pthread.h>

// Hyperparameters
#define NUM_EXAMPLES 60000
#define NUM_TEST_EXAMPLES 10000
#define MAX_THREADS 16
#define BATCH_SIZE 75
#define BATCHES_PER_EPOCH (NUM_EXAMPLES/BATCH_SIZE)
#define RELU_NEGATIVE_SLOPE 0.01
#define NUM_LAYERS 4
#define NUM_INPUTS 784
#define NUM_OUTPUTS 10
const size_t layers[NUM_LAYERS] = {NUM_INPUTS, 128, 64, NUM_OUTPUTS};
#define LEARNING_RATE 0.01
#define NUM_EPOCHS 10

typedef struct {
	size_t *imageIndices;
	uint8_t **trainingInputs;
	double **trainingOutputs;
	uint8_t **testInputs;
	double **testOutputs;
	double **biases;
	double ***weights;

	size_t batchStart;
	size_t batchEnd;
	size_t testStart;
	size_t testEnd;

	double cost;
	uint32_t numCorrect;
} threadArgs;

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
void calculateActivations(double **activations, double **biases, double ***weights)
{
	for (size_t layer = 0; layer < NUM_LAYERS-1; layer++) {
		size_t numNeurons = layers[layer + 1];
		size_t numWeights = layers[layer];
		for (size_t neuron = 0; neuron < numNeurons; neuron++) {
			double z = weightedSum(activations[layer], weights[layer][neuron], biases[layer][neuron], numWeights);
			double activation;
			if (layer == NUM_LAYERS-2) {
				activation = z; // softmax will occur later
			} else {
				activation = z > 0 ? z : z*RELU_NEGATIVE_SLOPE;
			}
			activations[layer + 1][neuron] = activation;
		}
	}

	// Softmax on Output Layer
	double *out = activations[NUM_LAYERS-1];
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

// For calculating accuracy
bool checkOutput(double **activations, double **trainingOutputs, size_t imageIndex)
{
	uint32_t correctNum = 0;
	uint32_t predictedNum = 0;
	double maxConfidence = -1.0;
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
double backpropogate(double **activations, double **dCdZs, double ***weights, double **biasGradients, double ***weightGradients, double **trainingOutputs, size_t imageIndex)
{
	double cost = 0.0;
	for (size_t layer = NUM_LAYERS-1; layer > 0; layer--) {
		size_t numNeurons = layers[layer];
		size_t numWeights = layers[layer-1];
		for (size_t neuron = 0; neuron < numNeurons; neuron++) {
			double activation = activations[layer][neuron];
			double dCdZ;
			if (layer == NUM_LAYERS-1) {
				double goal = trainingOutputs[imageIndex][neuron];
				dCdZ = activation - goal;
				if (goal == 1.0) cost -= log(activation + 1e-12);
			} else {
				double dCdA = 0.0;
				size_t numConnections = layers[layer+1];
				for (size_t i = 0; i < numConnections; i++) {
					dCdA += weights[layer][i][neuron]*dCdZs[layer+1][i];
				}
				double dAdZ = activation > 0 ? 1 : RELU_NEGATIVE_SLOPE;
				dCdZ = dCdA*dAdZ;
			}
			dCdZs[layer][neuron] = dCdZ;
			biasGradients[layer-1][neuron] += dCdZ;
			for (size_t i = 0; i < numWeights; i++) {
				weightGradients[layer-1][neuron][i] += dCdZ*activations[layer-1][i];
			}
		}
	}
	return cost;
}

// Add gradients to all biases and weights
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

void shuffleImages(size_t *imageIndices) {
	for (size_t i = 0; i < NUM_EXAMPLES; i++) {
		imageIndices[i] = i;
	}
	for (size_t i = NUM_EXAMPLES-1; i > 0; i--) {
		size_t randIndex = (size_t)((i + 1)*((double)rand()/(RAND_MAX + 1.0)));
		size_t tempIndex = imageIndices[i];
		imageIndices[i] = imageIndices[randIndex];
		imageIndices[randIndex] = tempIndex;
	}
}

// checks if integer is a power of two
bool isPowerOfTwo(int num)
{
	return (num > 0) && (num & (num - 1)) == 0;
}

// Train Network (threaded)
void *train(void *args)
{
	threadArgs *data = (threadArgs*)args;

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

	// Activations and Activation Gradients
	double **activations = malloc(NUM_LAYERS*sizeof(*activations));
	double **dCdZs = malloc(NUM_LAYERS*sizeof(*dCdZs));
	if (activations == NULL || dCdZs == NULL) error("Memory");
	for (size_t layer = 0; layer < NUM_LAYERS; layer++) {
		size_t numNeurons = layers[layer];
		activations[layer] = malloc(numNeurons*sizeof(*activations[layer]));
		dCdZs[layer] = malloc(numNeurons*sizeof(*dCdZs[layer]));
		if (activations[layer] == NULL || dCdZs[layer] == NULL) error("Memory");
		clearParameters(activations[layer], numNeurons);
		clearParameters(dCdZs[layer], numNeurons);
	}

	data->cost = 0.0;
	for (size_t batch = data->batchStart; batch < data->batchEnd; batch++) {

		resetGradients(biasGradients, weightGradients);

		// Iterate through training examples
		for (size_t exampleNum = 0; exampleNum < BATCH_SIZE; exampleNum++) {

			// Copy inputs to first layer
			size_t imageIndex = data->imageIndices[batch*BATCH_SIZE + exampleNum];
			for (size_t i = 0; i < NUM_INPUTS; i++) {
				activations[0][i] = (double)(data->trainingInputs[imageIndex][i])/255.0;
			}

			// Feedforward
			calculateActivations(activations, data->biases, data->weights);

			// Backpropogate
			data->cost += backpropogate(activations, dCdZs, data->weights, biasGradients, weightGradients, data->trainingOutputs, imageIndex);
		}

		adjustAllParameters(data->biases, biasGradients, data->weights, weightGradients);

	}

	// Free Bias Gradient Arrays
	for (size_t layer = 0; layer < NUM_LAYERS-1; layer++) {
		free(biasGradients[layer]);
	}
	free(biasGradients);

	// Free Weight Gradient Arrays
	for (size_t layer = 0; layer < NUM_LAYERS-1; layer++) {
		size_t numNeurons = layers[layer + 1];
		for (size_t neuron = 0; neuron < numNeurons; neuron++) {
			free(weightGradients[layer][neuron]);
		}
		free(weightGradients[layer]);
	}
	free(weightGradients);

	// Free Activation and Activation Gradient Arrays
	for (size_t layer = 0; layer < NUM_LAYERS; layer++) {
		free(activations[layer]);
		free(dCdZs[layer]);
	}
	free(activations);
	free(dCdZs);

}

// Find Accuracy for Test Examples
void *test(void *args)
{
	threadArgs *data = (threadArgs*)args;

	// Activations and Activation Gradients
	double **activations = malloc(NUM_LAYERS*sizeof(*activations));
	if (activations == NULL) error("Memory");
	for (size_t layer = 0; layer < NUM_LAYERS; layer++) {
		size_t numNeurons = layers[layer];
		activations[layer] = malloc(numNeurons*sizeof(*activations[layer]));
		if (activations[layer] == NULL) error("Memory");
		clearParameters(activations[layer], numNeurons);
	}

	for (size_t exampleNum = data->testStart; exampleNum < data->testEnd; exampleNum++) {
		// Copy inputs to first layer
		for (size_t i = 0; i < NUM_INPUTS; i++) {
			activations[0][i] = (double)data->testInputs[exampleNum][i]/255.0;
		}

		// Feedforward
		calculateActivations(activations, data->biases, data->weights);
		data->numCorrect += (uint32_t)checkOutput(activations, data->testOutputs, exampleNum);
	}

	// Free Activation Array
	for (size_t layer = 0; layer < NUM_LAYERS; layer++) {
		free(activations[layer]);
	}
	free(activations);

}

/* ------------------- Main ------------------- */

int main(int argc, char *argv[])
{
	if (argc != 2) error("Usage: train num_threads");
	size_t numThreads = atoi(argv[1]);
	if (numThreads > MAX_THREADS) error("Max threads exceeded");
	if (numThreads < 1) error("Thread count must be 1 or greater");
	if (!isPowerOfTwo(numThreads)) error("Thread count must be power of 2");

	pthread_t thread_ids[numThreads];
	threadArgs thread_args[numThreads];

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
	uint8_t **testInputs = malloc(NUM_TEST_EXAMPLES*sizeof(*testInputs));
	double **testOutputs = malloc(NUM_TEST_EXAMPLES*sizeof(*testOutputs));
	if (testInputs == NULL || testOutputs == NULL) error("Memory");
	for (size_t example = 0; example < NUM_TEST_EXAMPLES; example++) {
		testInputs[example] = malloc(NUM_INPUTS*sizeof(*testInputs[example]));
		testOutputs[example] = malloc(NUM_OUTPUTS*sizeof(*testOutputs[example]));
		if (testInputs[example] == NULL || testOutputs[example] == NULL) error("Memory");
	}
	for (size_t example = 0; example < NUM_TEST_EXAMPLES; example++) {
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

	// Initialize Thread Global Args
	for (size_t i = 0; i < numThreads; i++) {
		thread_args[i].imageIndices = imageIndices;
		thread_args[i].trainingInputs = trainingInputs;
		thread_args[i].trainingOutputs = trainingOutputs;
		thread_args[i].testInputs = testInputs;
		thread_args[i].testOutputs = testOutputs;
		thread_args[i].biases = biases;
		thread_args[i].weights = weights;
	}

	// Training Loop
	for (size_t epoch = 0; epoch < NUM_EPOCHS; epoch++) {

		double cost = 0.0;
		uint32_t numCorrect = 0;

		shuffleImages(imageIndices);

		// Initialize Thread Local Args
		for (size_t i = 0; i < numThreads; i++) {
			thread_args[i].batchStart = i*BATCHES_PER_EPOCH/numThreads;
			thread_args[i].batchEnd = (i+1)*BATCHES_PER_EPOCH/numThreads;
			thread_args[i].testStart = i*NUM_TEST_EXAMPLES/numThreads;
			thread_args[i].testEnd = (i+1)*NUM_TEST_EXAMPLES/numThreads;
			thread_args[i].cost = 0.0;
			thread_args[i].numCorrect = 0;
		}

		// Train (multi-threaded)
		for (size_t i = 0; i < numThreads; i++) {
			pthread_create(&thread_ids[i], NULL, train, (void*)(&thread_args[i]));
		}
		for (size_t i = 0; i < numThreads; i++) {
			pthread_join(thread_ids[i], NULL);
			cost += thread_args[i].cost;
		}

		// Test (multi-threaded)
		for (size_t i = 0; i < numThreads; i++) {
			pthread_create(&thread_ids[i], NULL, test, (void*)(&thread_args[i]));
		}
		for (size_t i = 0; i < numThreads; i++) {
			pthread_join(thread_ids[i], NULL);
			numCorrect += thread_args[i].numCorrect;
		}

		cost /= NUM_EXAMPLES;
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

	// Free Bias Array
	for (size_t layer = 0; layer < NUM_LAYERS-1; layer++) {
		free(biases[layer]);
	}
	free(biases);

	// Free Weight Array
	for (size_t layer = 0; layer < NUM_LAYERS-1; layer++) {
		size_t numNeurons = layers[layer + 1];
		for (size_t neuron = 0; neuron < numNeurons; neuron++) {
			free(weights[layer][neuron]);
		}
		free(weights[layer]);
	}
	free(weights);

	return 0;
}
