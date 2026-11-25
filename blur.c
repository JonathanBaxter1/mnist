#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#define IMAGE_SIZE 28
#define IMAGE_INDEX 4
#define KERNEL_SIZE 5
#define OUTPUT_SIZE (IMAGE_SIZE - KERNEL_SIZE + 1)

// Print Error and exit program
void error(const char *string)
{
	printf("Error: %s", string);
	exit(-1);
}

// Calculate Dot Product of a kernel on a single pixel
uint32_t kernelDot(const uint8_t *matrix, const uint8_t *kernel, size_t startX, size_t startY, size_t matrixSize, size_t kernelSize)
{
	uint32_t sum = 0;
	for (size_t i = 0; i < kernelSize; i++) { // Row
		for (size_t j = 0; j < kernelSize; j++) { // Column
			size_t x = startX + j;
			size_t y = startY + i;
			sum += (uint32_t)(matrix[y*matrixSize + x]*kernel[i*kernelSize + j]);
		}
	}
	return sum;
}

// Main
int main(void)
{
	// Vars
	uint8_t imageData[IMAGE_SIZE][IMAGE_SIZE];
	uint8_t kernel[KERNEL_SIZE][KERNEL_SIZE] = {
		{1,  4,  6,  4, 1},
		{4, 16, 24, 16, 4},
		{6, 24, 36, 24, 6},
		{4, 16, 24, 16, 4},
		{1,  4,  6,  4, 1},
	};
	uint8_t outputData[OUTPUT_SIZE][OUTPUT_SIZE];

	// Read Image File
	FILE *imageFile = fopen("t10k-images-idx3-ubyte", "rb");
	if (imageFile == NULL) error("Image fopen didn't work");
	fseek(imageFile, 16 + IMAGE_SIZE*IMAGE_SIZE*IMAGE_INDEX, SEEK_SET); // Skip to nth image
	size_t numPixels = IMAGE_SIZE*IMAGE_SIZE;
	size_t numPixelsRead = fread(imageData, 1, numPixels, imageFile);
	if (numPixelsRead != numPixels) error("Image fread didn't work");
	fclose(imageFile);

	// Blur Image
	for (size_t x = 0; x < OUTPUT_SIZE; x++) {
		for (size_t y = 0; y < OUTPUT_SIZE; y++) {
			uint32_t dot = kernelDot(&imageData[0][0], &kernel[0][0], x, y, IMAGE_SIZE, KERNEL_SIZE);
			outputData[y][x] = (uint8_t)(dot >> 8); // Kernel is divided by 256
		}
	}

	// Print Image to Terminal
	const char *grayscale = " .:-=+*#%@";
	for (size_t i = 0; i < OUTPUT_SIZE; i++) {
		for (size_t j = 0; j < OUTPUT_SIZE; j++) {
			char character = grayscale[outputData[i][j]*10/256];
			printf("%c%c", character, character);
		}
		printf("\n");
	}

	return 0;
}
