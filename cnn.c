#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#define IMAGE_SIZE 28
#define IMAGE_INDEX 4
#define NUM_FILTERS 8
#define KERNEL_SIZE 5
#define LAYER1_SIZE (IMAGE_SIZE - KERNEL_SIZE + 1)
#define LAYER2_SIZE (LAYER1_SIZE/2)

// Print Error and exit program
void error(const char *string)
{
	printf("Error: %s", string);
	exit(-1);
}

// Calculate Dot Product of kernel on a single pixel, plus bias
float convolution(const float *matrix, const float *kernel, float bias, size_t startX, size_t startY, size_t matrixSize, size_t kernelSize)
{
	float sum = bias;
	for (size_t i = 0; i < kernelSize; i++) { // Row
		for (size_t j = 0; j < kernelSize; j++) { // Column
			size_t x = startX + j;
			size_t y = startY + i;
			float matrixVal = matrix[y*matrixSize + x];
			sum += matrixVal*kernel[i*kernelSize + j];
		}
	}
	return sum;
}

// Read an image file into an array of floats
void readImageFile(float *imageData, const char *fileName)
{
	FILE *imageFile = fopen(fileName, "rb");
	if (imageFile == NULL) error("Image fopen didn't work");
	fseek(imageFile, 16 + IMAGE_SIZE*IMAGE_SIZE*IMAGE_INDEX, SEEK_SET); // Skip to nth image
	size_t numPixels = IMAGE_SIZE*IMAGE_SIZE;
	uint8_t imageDataInt[IMAGE_SIZE][IMAGE_SIZE];
	size_t numPixelsRead = fread(imageDataInt, 1, numPixels, imageFile);
	if (numPixelsRead != numPixels) error("Image fread didn't work");
	for (size_t i = 0; i < IMAGE_SIZE; i++) {
		for (size_t j = 0; j < IMAGE_SIZE; j++) {
			imageData[i*IMAGE_SIZE + j] = (float)imageDataInt[i][j]/255.0f;
		}
	}
	fclose(imageFile);
}

// Convert float to uint8_t
uint8_t float2uint8(float num)
{
	if (num < 0.0f) num = 0.0f;
	if (num > 1.0f) num = 1.0f;
	return (uint8_t)(num*255.0f + 0.5f);
}

// Relu
float relu(float num)
{
	return num > 0.0f ? num : 0.0f;
}

// Max of 4 floats
float max4(float num1, float num2, float num3, float num4)
{
	float max1 = num1 > num2 ? num1 : num2;
	float max2 = num3 > num4 ? num3 : num4;
	return max1 > max2 ? max1 : max2;
}

// Main
int main(void)
{
	// Vars
	float imageData[IMAGE_SIZE][IMAGE_SIZE];
	float kernels[NUM_FILTERS][KERNEL_SIZE][KERNEL_SIZE] = {
		{{0,0,0,0,0}, {0,0,0,0,0}, {0,0,1.0f,0,0}, {0,0,0,0,0}, {0,0,0,0,0}},
		{{0,0,0,0,0}, {0,0,-1,0,0}, {0,-1,5,-1,0}, {0,0,-1,0,0}, {0,0,0,0,0}},
		{{ 1/256.0f,  4/256.0f,  6/256.0f,  4/256.0f,  1/256.0f},
		 { 4/256.0f, 16/256.0f, 24/256.0f, 16/256.0f,  4/256.0f},
		 { 6/256.0f, 24/256.0f, 36/256.0f, 24/256.0f,  6/256.0f},
		 { 4/256.0f, 16/256.0f, 24/256.0f, 16/256.0f,  4/256.0f},
		 { 1/256.0f,  4/256.0f,  6/256.0f,  4/256.0f,  1/256.0f}},
		{{0,0,0,0,0}, {0,-1,-1,-1,0}, {0,-1,8,-1,0}, {0,-1,-1,-1,0}, {0,0,0,0,0}},
		{{0,0,0,0,0}, {0,-1,0,1,0}, {0,-2,0,2,0}, {0,-1,0,1,0}, {0,0,0,0,0}},
		{{0,0,0,0,0}, {0,-1,-2,-1,0}, {0,0,0,0,0}, {0,1,2,1,0}, {0,0,0,0,0}},
		{{0,0,0,0,0}, {0,-2,-1,0,0}, {0,-1,1,1,0}, {0,0,1,2,0}, {0,0,0,0,0}},
		{{-1,-1,4,-1,-1}, {-1,-1,4,-1,-1}, {-1,-1,4,-1,-1}, {-1,-1,4,-1,-1}, {-1,-1,4,-1,-1}},
	};
	float biases[NUM_FILTERS] = {0.0f};
	float layer1Data[NUM_FILTERS][LAYER1_SIZE][LAYER1_SIZE]; // conv layer 1
	float layer2Data[NUM_FILTERS][LAYER2_SIZE][LAYER2_SIZE]; // pool layer 1

	// Read Image File
	readImageFile(&imageData[0][0], "t10k-images-idx3-ubyte");

	// Apply Filters
	for (size_t filter = 0; filter < NUM_FILTERS; filter++) { // conv layer 1 + relu
		for (size_t y = 0; y < LAYER1_SIZE; y++) {
			for (size_t x = 0; x < LAYER1_SIZE; x++) {
				float dot = convolution(&imageData[0][0], &kernels[filter][0][0], biases[filter], x, y, IMAGE_SIZE, KERNEL_SIZE);
				layer1Data[filter][y][x] = relu(dot);
			}
		}
	}
	for (size_t filter = 0; filter < NUM_FILTERS; filter++) { // pool layer 1
		for (size_t y = 0; y < LAYER2_SIZE; y++) {
			for (size_t x = 0; x < LAYER2_SIZE; x++) {
				float num1 = layer1Data[filter][y*2][x*2];
				float num2 = layer1Data[filter][y*2][x*2 + 1];
				float num3 = layer1Data[filter][y*2 + 1][x*2];
				float num4 = layer1Data[filter][y*2 + 1][x*2 + 1];
				layer2Data[filter][y][x] = max4(num1, num2, num3, num4);
			}
		}
	}

	// Print Filtered Images to Terminal
	const char *grayscale = " .:-=+*#%@";
	for (size_t filter = 0; filter < NUM_FILTERS; filter++) {
		for (size_t i = 0; i < LAYER2_SIZE; i++) {
			for (size_t j = 0; j < LAYER2_SIZE; j++) {
				char character = grayscale[float2uint8(layer2Data[filter][i][j])*10/256];
				printf("%c%c", character, character);
			}
			printf("\n");
		}
		printf("\n");
	}

	return 0;
}
