/*
The MIT License(MIT)
Copyright(c) 2016 Fan Wen Jie

Permission is hereby granted, free of charge, to any person obtaining a copy
of this softwareand associated documentation files(the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions :

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
// Source: https://github.com/fan-wenjie/LeNet-5

#include "lenetGPU.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define FILE_TRAIN_IMAGE		"train-images-idx3-ubyte"
#define FILE_TRAIN_LABEL		"train-labels-idx1-ubyte"
#define FILE_TEST_IMAGE		"t10k-images-idx3-ubyte"
#define FILE_TEST_LABEL		"t10k-labels-idx1-ubyte"
#define LENET_FILE 		"model.dat"
#define COUNT_TRAIN		60000
#define COUNT_TEST		10000


int read_data(unsigned char(*data)[28][28], unsigned char label[], const int count, const char data_file[], const char label_file[])
{
    FILE *fp_image = fopen(data_file, "rb");
    FILE *fp_label = fopen(label_file, "rb");
    if (!fp_image||!fp_label) return 1;
	fseek(fp_image, 16, SEEK_SET);
	fseek(fp_label, 8, SEEK_SET);
	fread(data, sizeof(*data)*count, 1, fp_image);
	fread(label,count, 1, fp_label);
	fclose(fp_image);
	fclose(fp_label);
	return 0;
}

void training(LeNet5 *lenet, image *train_data, uint8 *train_label, int batch_size, int total_size)
{
	printf("Total images in training set: %d\n", total_size);
	printf("Batchsize: %d\n", batch_size);
	
	double* buffer = (double*)malloc(sizeof(LeNet5));

	//Allocate the device feature array.
	Feature* deviceFeatureArray;
	gpuErrchk(cudaMalloc((void**)&deviceFeatureArray, sizeof(Feature) * batch_size))
	//Allocate the device neural net.
	LeNet5* deviceLenet;
    gpuErrchk(cudaMalloc((void**)&deviceLenet, sizeof(LeNet5)));

	image* deviceInputs;
	gpuErrchk(cudaMalloc((void**)&deviceInputs, sizeof(image) * batch_size));

	Feature* deviceErrors;
	gpuErrchk(cudaMalloc((void**)&deviceErrors, sizeof(Feature) * batch_size));

	LeNet5* deviceDeltas;
	gpuErrchk(cudaMalloc((void**)&deviceDeltas, sizeof(LeNet5) * batch_size));

	uint8* deviceLabels;
	gpuErrchk(cudaMalloc((void**)&deviceLabels, sizeof(uint8) * batch_size));

	double* deviceBuffer;
	gpuErrchk(cudaMalloc((void**)&deviceBuffer, sizeof(LeNet5)));
	//gpuErrchk(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 50000000));
	//gpuErrchk(cudaDeviceSetLimit(cudaLimitStackSize, 250000));
	//For every batch we train on.
	for (int i = 0, percent = 0; i <= total_size - batch_size; i += batch_size)
	{
		printf("\nTraining on images: %d-%d\t", i, i + batch_size);
		TrainBatch(
			lenet,
			&(train_data[i]),
			deviceInputs,
			&(train_label[i]),
			deviceLabels,
			batch_size,
			deviceLenet,
			deviceFeatureArray,
			deviceErrors,
			deviceDeltas,
			buffer,
			deviceBuffer
		);
		if (i * 100 / total_size > percent)
			printf("Training %2d%% complete", percent = i * 100 / total_size);
	}

	gpuErrchk(cudaFree(deviceLenet));
	gpuErrchk(cudaFree(deviceFeatureArray));
	gpuErrchk(cudaFree(deviceInputs));
	gpuErrchk(cudaFree(deviceErrors));
	gpuErrchk(cudaFree(deviceDeltas));
	gpuErrchk(cudaFree(deviceLabels));
	gpuErrchk(cudaFree(deviceBuffer));

	free(buffer);

	printf("\n");
}

int testing(LeNet5 *lenet, image *test_data, uint8 *test_label, int total_size)
{
	int batch_size = 300;
	//Allocate the host feature array.
	Feature* featureArray = (Feature*)malloc(sizeof(Feature) * batch_size);

	//Allocate the device feature array.
	Feature* deviceFeatureArray;
	gpuErrchk(cudaMalloc((void**)&deviceFeatureArray, sizeof(Feature) * batch_size))
	//Allocate the device neural net.
	LeNet5* deviceLenet;
    gpuErrchk(cudaMalloc((void**)&deviceLenet, sizeof(LeNet5)));

	image* deviceInputs;
    gpuErrchk(cudaMalloc((void**)&deviceInputs, sizeof(image) * batch_size));

	int confusion_matrix[10][10] = { 0 }; // For our specific problem, we have a 10x10 confusion matrix 
	int right = 0;
	
	for (int b = 0; b <= total_size - batch_size; b += batch_size)
	{
		uint8* p = PredictBatch(lenet, featureArray, &(test_data[b]), deviceInputs, batch_size, deviceLenet, deviceFeatureArray, 10);

		for (int i = 0; i < batch_size; i++)
		{
			uint8 l = test_label[b + i];
			confusion_matrix[l][p[i]] += 1;
			right += (l == p[i]) ? 1 : 0; // If the prediction is correct, increment our counter
		}
		free(p);
	}
	if (total_size % batch_size != 0)
	{
		int leftOver = (total_size % batch_size);
		int b = total_size - leftOver;

		uint8* p = PredictBatch(lenet, featureArray, &(test_data[b]), deviceInputs, leftOver, deviceLenet, deviceFeatureArray, 10);

		for (int i = 0; i < leftOver; i++)
		{
			uint8 l = test_label[b + i];
			confusion_matrix[l][p[i]] += 1;
			right += (l == p[i]) ? 1 : 0; // If the prediction is correct, increment our counter
		}
		free(p);
	}
	PrintResult(confusion_matrix);

	gpuErrchk(cudaFree(deviceLenet));
	gpuErrchk(cudaFree(deviceFeatureArray));
	gpuErrchk(cudaFree(deviceInputs));

	free(featureArray);
	return right;
}

int save(LeNet5 *lenet, char filename[])
{
	FILE *fp = fopen(filename, "wb");
	if (!fp) return 1;
	fwrite(lenet, sizeof(LeNet5), 1, fp);
	fclose(fp);
	return 0;
}

int load(LeNet5 *lenet, char filename[])
{
	FILE *fp = fopen(filename, "rb");
	if (!fp) return 1;
	fread(lenet, sizeof(LeNet5), 1, fp);
	fclose(fp);
	return 0;
}

int main()
{
	image *train_data = (image *)calloc(COUNT_TRAIN, sizeof(image));
	uint8 *train_label = (uint8 *)calloc(COUNT_TRAIN, sizeof(uint8));
	image *test_data = (image *)calloc(COUNT_TEST, sizeof(image));
	uint8 *test_label = (uint8 *)calloc(COUNT_TEST, sizeof(uint8));
	if (read_data(train_data, train_label, COUNT_TRAIN, FILE_TRAIN_IMAGE, FILE_TRAIN_LABEL))
	{
		printf("ERROR!!!\nDataset File Not Find!Please Copy Dataset to the Floder Included the exe\n");
		free(train_data);
		free(train_label);
		system("pause");
	}
	if (read_data(test_data, test_label, COUNT_TEST, FILE_TEST_IMAGE, FILE_TEST_LABEL))
	{
		printf("ERROR!!!\nDataset File Not Find!Please Copy Dataset to the Floder Included the exe\n");
		free(test_data);
		free(test_label);
		system("pause");
	}

	LeNet5 *lenet = (LeNet5 *)malloc(sizeof(LeNet5));
	if (load(lenet, LENET_FILE))
		Initial(lenet);
	clock_t start = clock();

	//----------------------------------------------------------------------------------------
	// Sai: When the entire training dataset is passed forward and backward through 
	// the neural network one time, it is called one "epoch".
	// batch size = the number of training examples in one forward/backward pass. 
	// Different values of batch size can significantly affect the training and test accuracies
	int batches[] = { 300 };
	// We are using only one epoch, even though multiple epochs have their benefits.
	for (int i = 0; i < sizeof(batches) / sizeof(*batches); ++i)
	{
		training(lenet, train_data, train_label, batches[i], COUNT_TRAIN);
	}
	//printf("Calculating training accuracy...\n");
	//int training_right = testing(lenet, train_data, train_label, COUNT_TRAIN);
	//printf("Training accuracy: %f%%\n", training_right * 100.0 / COUNT_TRAIN);

	printf("Calculating test accuracy...\n");
	int right = testing(lenet, test_data, test_label, COUNT_TEST);
	printf("Testing: Correct predictions = %d (%.2f%%)\n", right, right/100.0);
	int wrong = COUNT_TEST - right;
	printf("Testing: Wrong predictions = %d (%.2f%%)\n", wrong, wrong/100.0);
	//----------------------------------------------------------------------------------------

	printf("Time taken: %u sec\n", (unsigned)(clock() - start)/CLOCKS_PER_SEC);
	//save(lenet, LENET_FILE);
	free(lenet);
	free(train_data);
	free(train_label);
	free(test_data);
	free(test_label);
	system("pause");
	return 0;
}