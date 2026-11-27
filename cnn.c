#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define LEARNING_RATE 0.01
#define NUM_EPOCHS 100

#define IMAGE_SIZE 28
#define IMAGE_INDEX 4
#define NUM_FILTERS1 8
#define NUM_FILTERS2 16
#define KERNEL_SIZE 5
#define LAYER1_SIZE (IMAGE_SIZE - KERNEL_SIZE + 1)
#define LAYER2_SIZE (LAYER1_SIZE/2)
#define LAYER3_SIZE (LAYER2_SIZE - KERNEL_SIZE + 1)
#define LAYER4_SIZE (LAYER3_SIZE/2)
#define LAYER5_SIZE (NUM_FILTERS2*LAYER4_SIZE*LAYER4_SIZE)
#define OUTPUT_SIZE 10

// Model Parameters
typedef struct {
	float w1[NUM_FILTERS1][KERNEL_SIZE][KERNEL_SIZE]; // kernel 1
	float b1[NUM_FILTERS1];
	float w2[NUM_FILTERS2][NUM_FILTERS1][KERNEL_SIZE][KERNEL_SIZE]; // kernel 2
	float b2[NUM_FILTERS2];
	float w3[OUTPUT_SIZE][LAYER5_SIZE];
	float b3[OUTPUT_SIZE];
} Parameters;

// Activations
typedef struct {
	float image[IMAGE_SIZE][IMAGE_SIZE];
	float layer1[NUM_FILTERS1][LAYER1_SIZE][LAYER1_SIZE]; // conv layer 1
	float layer2[NUM_FILTERS1][LAYER2_SIZE][LAYER2_SIZE]; // pool layer 1
	float layer3[NUM_FILTERS2][LAYER3_SIZE][LAYER3_SIZE]; // conv layer 2
	float layer4[NUM_FILTERS2][LAYER4_SIZE][LAYER4_SIZE]; // pool layer 2
	float output[OUTPUT_SIZE]; // output
	uint8_t argmax1[NUM_FILTERS1][LAYER2_SIZE][LAYER2_SIZE];
	uint8_t argmax2[NUM_FILTERS2][LAYER4_SIZE][LAYER4_SIZE];
} Activations;

// Print Error and exit program
void error(const char *string)
{
	printf("Error: %s", string);
	exit(-1);
}

// Read an image file into an array of floats
uint8_t readImageFile(float imageData[IMAGE_SIZE][IMAGE_SIZE])
{
	// Image Data
	FILE *imageFile = fopen("train-images-idx3-ubyte", "rb");
	if (imageFile == NULL) error("Image fopen didn't work");
	fseek(imageFile, 16 + IMAGE_SIZE*IMAGE_SIZE*IMAGE_INDEX, SEEK_SET); // Skip to nth image
	size_t numPixels = IMAGE_SIZE*IMAGE_SIZE;
	uint8_t imageDataInt[IMAGE_SIZE][IMAGE_SIZE];
	size_t numPixelsRead = fread(imageDataInt, 1, numPixels, imageFile);
	if (numPixelsRead != numPixels) error("Image fread didn't work");
	for (size_t i = 0; i < IMAGE_SIZE; i++) {
		for (size_t j = 0; j < IMAGE_SIZE; j++) {
			imageData[i][j] = (float)imageDataInt[i][j]/255.0f;
		}
	}
	fclose(imageFile);

	// Label Data
	FILE *labelFile = fopen("train-labels-idx1-ubyte", "rb");
	if (labelFile == NULL) error("Labels fopen didn't work");
	fseek(labelFile, 8 + IMAGE_INDEX, SEEK_SET); // Skip to nth image
	uint8_t label;
	size_t numBytesRead = fread(&label, 1, 1, labelFile);
	if (numBytesRead != 1) error("Labels fread didn't work");
	fclose(labelFile);

	return label;
}

// Relu
float relu(float num)
{
	return num > 0.0f ? num : 0.0f;
}

void runModel(const Parameters *model, Activations *data)
{
	size_t i, j, k, x, y, f;

	for (f = 0; f < NUM_FILTERS1; f++) for (y = 0; y < LAYER1_SIZE; y++) for (x = 0; x < LAYER1_SIZE; x++) { // conv layer 1 + relu
		float sum = model->b1[f];
		for (i = 0; i < KERNEL_SIZE; i++) for (j = 0; j < KERNEL_SIZE; j++) {
			sum += data->image[i + y][j + x]*model->w1[f][i][j];
		}
		data->layer1[f][y][x] = relu(sum);
	}
	for (f = 0; f < NUM_FILTERS1; f++) for (y = 0; y < LAYER2_SIZE; y++) for (x = 0; x < LAYER2_SIZE; x++) { // pool layer 1
		float num1 = data->layer1[f][y*2][x*2];
		float num2 = data->layer1[f][y*2][x*2 + 1];
		float num3 = data->layer1[f][y*2 + 1][x*2];
		float num4 = data->layer1[f][y*2 + 1][x*2 + 1];
		float max;
		uint8_t argmax;
		if (num1 > num2 && num1 > num3 && num1 > num4) {max = num1; argmax = 0;
		} else if (num2 > num3 && num2 > num4) {	max = num2; argmax = 1;
		} else if (num3 > num4) {			max = num3; argmax = 2;
		} else {					max = num4; argmax = 3;
		}
		data->layer2[f][y][x] = max;
		data->argmax1[f][y][x] = argmax;
	}
	for (f = 0; f < NUM_FILTERS2; f++) for (y = 0; y < LAYER3_SIZE; y++) for (x = 0; x < LAYER3_SIZE; x++) { // conv layer 2 + relu
		float sum = model->b2[f];
		for (k = 0; k < NUM_FILTERS1; k++) for (i = 0; i < KERNEL_SIZE; i++) for (j = 0; j < KERNEL_SIZE; j++) {
			sum += data->layer2[k][i + y][j + x]*model->w2[f][k][i][j];
		}
		data->layer3[f][y][x] = relu(sum);
	}
	for (f = 0; f < NUM_FILTERS2; f++) for (y = 0; y < LAYER4_SIZE; y++) for (x = 0; x < LAYER4_SIZE; x++) { // pool layer 2
		float num1 = data->layer3[f][y*2][x*2];
		float num2 = data->layer3[f][y*2][x*2 + 1];
		float num3 = data->layer3[f][y*2 + 1][x*2];
		float num4 = data->layer3[f][y*2 + 1][x*2 + 1];
		float max;
		uint8_t argmax;
		if (num1 > num2 && num1 > num3 && num1 > num4) {max = num1; argmax = 0;
		} else if (num2 > num3 && num2 > num4) {	max = num2; argmax = 1;
		} else if (num3 > num4) {			max = num3; argmax = 2;
		} else {					max = num4; argmax = 3;
		}
		data->layer4[f][y][x] = max;
		data->argmax2[f][y][x] = argmax;
	}
	float *layer5Data = &data->layer4[0][0][0]; // "flatten" layer 4
	for (size_t outputIndex = 0; outputIndex < OUTPUT_SIZE; outputIndex++) { // fully connected layer
		float sum = model->b3[outputIndex];
		for (size_t layer5Index = 0; layer5Index < LAYER5_SIZE; layer5Index++) {
			sum += model->w3[outputIndex][layer5Index]*layer5Data[layer5Index];
		}
		data->output[outputIndex] = sum;
	}
	float max = data->output[0]; // softmax
	for (i = 1; i < OUTPUT_SIZE; i++) {
		max = data->output[i] > max ? data->output[i] : max;
	}
	float sum = 0.0f;
	for (i = 0; i < OUTPUT_SIZE; i++) {
		data->output[i] = expf(data->output[i] - max);
		sum += data->output[i];
	}
	for (i = 0; i < OUTPUT_SIZE; i++) {
		data->output[i] /= sum;
	}
}

// Initialize weights to random numbers, and biases to 0
void initParams(Parameters *model)
{
	size_t i, j, k, f;

	// Weights (He initialization)
	uint32_t fanIn;
	float x;
	fanIn = KERNEL_SIZE*KERNEL_SIZE; // kernel 1
	x = sqrtf(6.0f/(float)fanIn);
	for (f = 0; f < NUM_FILTERS1; f++) for (i = 0; i < KERNEL_SIZE; i++) for (j = 0; j < KERNEL_SIZE; j++) {
		model->w1[f][i][j] = x*(((float)rand()/RAND_MAX)*2.0f - 1.0f);
	}

	fanIn = NUM_FILTERS1*KERNEL_SIZE*KERNEL_SIZE; // kernel 2
	x = sqrtf(6.0f/(float)fanIn);
	for (f = 0; f < NUM_FILTERS2; f++) for (k = 0; k < NUM_FILTERS1; k++) for (i = 0; i < KERNEL_SIZE; i++) for (j = 0; j < KERNEL_SIZE; j++) {
		model->w2[f][k][i][j] = x*(((float)rand()/RAND_MAX)*2.0f - 1.0f);
	}

	fanIn = LAYER5_SIZE; // Fully connected weights
	x = sqrtf(6.0f/(float)fanIn);
	for (i = 0; i < OUTPUT_SIZE; i++) for (j = 0; j < LAYER5_SIZE; j++) {
		model->w3[i][j] = x*(((float)rand()/RAND_MAX)*2.0f - 1.0f);
	}

	// Biases
	for (i = 0; i < NUM_FILTERS1; i++) {
		model->b1[i] = 0.0f;
	}
	for (i = 0; i < NUM_FILTERS2; i++) {
		model->b2[i] = 0.0f;
	}
	for (i = 0; i < OUTPUT_SIZE; i++) {
		model->b3[i] = 0.0f;
	}
}

// Calculate Gradients (backpropagation)
void findGradients(const Parameters *model, const Activations *data, Parameters *delta, const float labelArray[OUTPUT_SIZE])
{
	float layer1Delta[NUM_FILTERS1][LAYER1_SIZE][LAYER1_SIZE] = {0.0f};
	float layer2Delta[NUM_FILTERS1][LAYER2_SIZE][LAYER2_SIZE] = {0.0f};
	float layer3Delta[NUM_FILTERS2][LAYER3_SIZE][LAYER3_SIZE] = {0.0f};
	float layer4Delta[NUM_FILTERS2][LAYER4_SIZE][LAYER4_SIZE] = {0.0f};
	const float *layer5Data = &data->layer4[0][0][0];
	float *layer5Delta = &layer4Delta[0][0][0];

	size_t i, j, k, x, y, f;

	for (i = 0; i < OUTPUT_SIZE; i++) { // fully connected layer
		delta->b3[i] = data->output[i] - labelArray[i];
		for (j = 0; j < LAYER5_SIZE; j++) {
			delta->w3[i][j] = delta->b3[i]*layer5Data[j];
		}
	}
	for (j = 0; j < LAYER5_SIZE; j++) {
		float sum = 0.0f;
		for (i = 0; i < OUTPUT_SIZE; i++) {
			sum += model->w3[i][j]*delta->b3[i];
		}
		layer5Delta[j] = sum;
	}
	for (f = 0; f < NUM_FILTERS2; f++) for (i = 0; i < LAYER4_SIZE; i++) for (j = 0; j < LAYER4_SIZE; j++) { // conv layer 2
		uint8_t argmax = data->argmax2[f][i][j];
		size_t x = j*2;
		size_t y = i*2;
		if (argmax >= 2) y++;
		if (argmax%2 == 1) x++;
		layer3Delta[f][y][x] = data->layer3[f][y][x] > 0.0f ? layer4Delta[f][i][j] : 0.0f;
	}
	for (f = 0; f < NUM_FILTERS2; f++) for (k = 0; k < NUM_FILTERS1; k++) for (i = 0; i < KERNEL_SIZE; i++) for (j = 0; j < KERNEL_SIZE; j++) {
		float sum = 0.0f;
		for (y = 0; y < LAYER3_SIZE; y++) for (x = 0; x < LAYER3_SIZE; x++) {
			sum += layer3Delta[f][y][x]*data->layer2[k][y + i][x + j];
			layer2Delta[k][y + i][x + j] += layer3Delta[f][y][x]*model->w2[f][k][i][j];
		}
		delta->w2[f][k][i][j] = sum;
	}
	for (f = 0; f < NUM_FILTERS2; f++) {
		float sum = 0.0f;
		for (i = 0; i < LAYER3_SIZE; i++) for (j = 0; j < LAYER3_SIZE; j++) {
			sum += layer3Delta[f][i][j];
		}
		delta->b2[f] = sum;
	}

	for (size_t f = 0; f < NUM_FILTERS1; f++) for (size_t i = 0; i < LAYER2_SIZE; i++) for (size_t j = 0; j < LAYER2_SIZE; j++) { // conv layer 1
		uint8_t argmax = data->argmax1[f][i][j];
		size_t x = j*2;
		size_t y = i*2;
		if (argmax >= 2) y++;
		if (argmax%2 == 1) x++;
		layer1Delta[f][y][x] = data->layer1[f][y][x] > 0.0f ? layer2Delta[f][i][j] : 0.0f;
	}
	for (f = 0; f < NUM_FILTERS1; f++) for (i = 0; i < KERNEL_SIZE; i++) for (j = 0; j < KERNEL_SIZE; j++) {
		float sum = 0.0f;
		for (y = 0; y < LAYER1_SIZE; y++) for (x = 0; x < LAYER1_SIZE; x++) {
			sum += layer1Delta[f][y][x]*data->image[y + i][x + j];
		}
		delta->w1[f][i][j] = sum;
	}
	for (f = 0; f < NUM_FILTERS1; f++) {
		float sum = 0.0f;
		for (i = 0; i < LAYER1_SIZE; i++) for (j = 0; j < LAYER1_SIZE; j++) {
			sum += layer1Delta[f][i][j];
		}
		delta->b1[f] = sum;
	}
}

// Update Weights and Biases
void updateParameters(Parameters *model, const Parameters *delta)
{
	size_t i, j, k, l;
	for (i = 0; i < NUM_FILTERS1; i++) {
		for (j = 0; j < KERNEL_SIZE; j++) for (k = 0; k < KERNEL_SIZE; k++) {
			model->w1[i][j][k] -= LEARNING_RATE*delta->w1[i][j][k];
		}
		model->b1[i] -= LEARNING_RATE*delta->b1[i];
	}
	for (i = 0; i < NUM_FILTERS2; i++) {
		for (j = 0; j < NUM_FILTERS1; j++) for (k = 0; k < KERNEL_SIZE; k++) for (l = 0; l < KERNEL_SIZE; l++) {
			model->w2[i][j][k][l] -= LEARNING_RATE*delta->w2[i][j][k][l];
		}
		model->b2[i] -= LEARNING_RATE*delta->b2[i];
	}
	for (i = 0; i < OUTPUT_SIZE; i++) {
		for (j = 0; j < LAYER5_SIZE; j++) {
			model->w3[i][j] -= LEARNING_RATE*delta->w3[i][j];
		}
		model->b3[i] -= LEARNING_RATE*delta->b3[i];
	}
}

// Main
int main(void)
{
	srand(time(NULL));

	Parameters model;
	Parameters delta;
	Activations activations;

	// Read Image Data
	uint8_t label = readImageFile(activations.image);
	float labelArray[OUTPUT_SIZE] = {0.0f};
	labelArray[(size_t)label] = 1.0f;

	// Training Loop
	initParams(&model);
	for (size_t epoch = 0; epoch < NUM_EPOCHS; epoch++) {
		runModel(&model, &activations);
		findGradients(&model, &activations, &delta, labelArray);
		updateParameters(&model, &delta);
	}

	// Print Result
	float max = -1.0f;
	uint8_t prediction = 0;
	for (size_t i = 0; i < OUTPUT_SIZE; i++) {
		float confidence = activations.output[i]*100.0f;
		if (confidence > max) {
			max = confidence;
			prediction = i;
		}
		printf("%zu: %.3f%%\n", i, confidence);
	}
	printf("Prediction: %d, Actual: %d\n", prediction, label);

	return 0;
}
