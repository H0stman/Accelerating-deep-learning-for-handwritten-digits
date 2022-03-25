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
#include <memory.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#define GETLENGTH(array) (sizeof(array)/sizeof(*(array)))

#define GETCOUNT(array)  (sizeof(array)/sizeof(double))

#define FOREACH(i,count) for (int i = 0; i < count; ++i)								

#define CONVOLUTE_VALID(input,output,weight)											\
{																						\
	FOREACH(o0,GETLENGTH(output))														\
		FOREACH(o1,GETLENGTH(*(output)))												\
			FOREACH(w0,GETLENGTH(weight))												\
				FOREACH(w1,GETLENGTH(*(weight)))										\
					(output)[o0][o1] += (input)[o0 + w0][o1 + w1] * (weight)[w0][w1];	\
}

#define CONVOLUTE_VALID_GPU1(input, output, weight)														\
{																										\
	for (unsigned int i = 0; i < LENGTH_KERNEL; i++)													\
	{																									\
		for (unsigned int j = 0; j < LENGTH_KERNEL; j++)												\
		{																								\
			output[threadIdx.x][threadIdx.y] = __dadd_rn(output[threadIdx.x][threadIdx.y], __dmul_rn(input[threadIdx.x + i][threadIdx.y + j], weight[i][j]));	\
		}																								\
	}																									\
}

#define CONVOLUTE_VALID_GPU2(input, output, weight)														\
{																										\
	for (unsigned int i = 0; i < LENGTH_KERNEL; i++)													\
	{																									\
		for (unsigned int j = 0; j < LENGTH_KERNEL; j++)												\
		{																								\
			output[threadIdx.x][threadIdx.y] = __dadd_rn(output[threadIdx.x][threadIdx.y], __dmul_rn(input[threadIdx.x + i][threadIdx.y + j], weight[i][j]));	\
		}																								\
	}																									\
}

#define CONVOLUTE_VALID_GPU3(input, output, weight)														\
{																										\
	for (unsigned int i = 0; i < LENGTH_KERNEL; i++)													\
	{																									\
		for (unsigned int j = 0; j < LENGTH_KERNEL; j++)												\
		{																								\
			output[0][0] = __dadd_rn(output[0][0] , __dmul_rn(input[0 + i][0 + j], weight[i][j]));	\
		}																								\
	}																									\
}

#define CONVOLUTE_FULL(input,output,weight)												\
{																						\
	FOREACH(i0,GETLENGTH(input))														\
		FOREACH(i1,GETLENGTH(*(input)))													\
			FOREACH(w0,GETLENGTH(weight))												\
				FOREACH(w1,GETLENGTH(*(weight)))										\
					(output)[i0 + w0][i1 + w1] += (input)[i0][i1] * (weight)[w0][w1];	\
}																						

// Similar functionality as the code in Figure 16.4 of the textbook
#define CONVOLUTION_FORWARD(input,output,weight,bias,action)					\
{																				\
	for (int x = 0; x < GETLENGTH(weight); ++x)									\
		for (int y = 0; y < GETLENGTH(*weight); ++y)							\
			CONVOLUTE_VALID(input[x], output[y], weight[x][y]);					\
	FOREACH(j, GETLENGTH(output))												\
		FOREACH(i, GETCOUNT(output[j]))											\
		((double *)output[j])[i] = action(((double *)output[j])[i] + bias[j]);	\
}

#define CONVOLUTION_FORWARD_GPU1(input, output, weight, bias)																		\
{																																	\
	CONVOLUTE_VALID_GPU1(input[0], output[blockIdx.y], weight[0][blockIdx.y]);																		\
	((double *)output[blockIdx.y])[threadIdx.x * LENGTH_FEATURE1 + threadIdx.y] = relu(__dadd_rn(((double *)output[blockIdx.y])[threadIdx.x * LENGTH_FEATURE1 + threadIdx.y], bias[blockIdx.y]));	\
}

#define CONVOLUTION_FORWARD_GPU2(input, output, weight, bias)																		\
{																																	\
	for (unsigned int q = 0; q < LAYER2; q++)																						\
	{																																\
		CONVOLUTE_VALID_GPU2(input[q], output[blockIdx.y], weight[q][blockIdx.y]);																	\
	}																																\
	((double *)output[blockIdx.y])[threadIdx.x * LENGTH_FEATURE3 + threadIdx.y] = relu(__dadd_rn(((double *)output[blockIdx.y])[threadIdx.x * LENGTH_FEATURE3 + threadIdx.y], bias[blockIdx.y]));			\
}

#define CONVOLUTION_FORWARD_GPU3(input, output, weight, bias)																		\
{																																	\
	for (unsigned int q = 0; q < LAYER4; q++)																						\
	{																																\
		CONVOLUTE_VALID_GPU3(input[q], output[blockIdx.y], weight[q][blockIdx.y]);																	\
	}																																\
	((double *)output[blockIdx.y])[0] = relu(__dadd_rn(((double *)output[blockIdx.y])[0], bias[blockIdx.y]));								\
}

#define CONVOLUTION_BACKWARD(input,inerror,outerror,weight,wd,bd)\
{																			\
	for (int x = 0; x < GETLENGTH(weight); ++x)								\
		for (int y = 0; y < GETLENGTH(*weight); ++y)						\
			CONVOLUTE_FULL(outerror[y], inerror[x], weight[x][y]);			\
	FOREACH(i, GETCOUNT(inerror))											\
		((double *)inerror)[i] *= relugrad(((double *)input)[i]);			\
	FOREACH(j, GETLENGTH(outerror))											\
		FOREACH(i, GETCOUNT(outerror[j]))									\
		bd[j] += ((double *)outerror[j])[i];								\
	for (int x = 0; x < GETLENGTH(weight); ++x)								\
		for (int y = 0; y < GETLENGTH(*weight); ++y)						\
			CONVOLUTE_VALID(input[x], wd[x][y], outerror[y]);				\
}																						

// Similar functionality as the code in Figure 16.5 of the textbook
#define SUBSAMP_MAX_FORWARD(input,output)														\
{																								\
	const int len0 = GETLENGTH(*(input)) / GETLENGTH(*(output));								\
	const int len1 = GETLENGTH(**(input)) / GETLENGTH(**(output));								\
	FOREACH(i, GETLENGTH(output))																\
	FOREACH(o0, GETLENGTH(*(output)))															\
	FOREACH(o1, GETLENGTH(**(output)))															\
	{																							\
		int x0 = 0, x1 = 0, ismax;																\
		FOREACH(l0, len0)																		\
			FOREACH(l1, len1)																	\
		{																						\
			ismax = input[i][o0*len0 + l0][o1*len1 + l1] > input[i][o0*len0 + x0][o1*len1 + x1];\
			x0 += ismax * (l0 - x0);															\
			x1 += ismax * (l1 - x1);															\
		}																						\
		output[i][o0][o1] = input[i][o0*len0 + x0][o1*len1 + x1];								\
	}																							\
}

#define SUBSAMP_MAX_FORWARD_GPU(input, output)																													\
{																																								\
	const int len0 = GETLENGTH(*input) / GETLENGTH(*output);																									\
	const int len1 = GETLENGTH(**input) / GETLENGTH(**output);																									\
	int x0 = 0, x1 = 0, ismax;																																	\
	FOREACH(l0, len0)																																			\
    {																																							\
		FOREACH(l1, len1)																																		\
		{																																						\
			ismax = input[blockIdx.y][threadIdx.x * len0 + l0][threadIdx.y * len1 + l1] > input[blockIdx.y][threadIdx.x * len0 + x0][threadIdx.y * len1 + x1];	\
			x0 += ismax * (l0 - x0);																															\
			x1 += ismax * (l1 - x1);																															\
		}																																						\
    }																																							\
	output[blockIdx.y][threadIdx.x][threadIdx.y] = input[blockIdx.y][threadIdx.x * len0 + x0][threadIdx.y * len1 + x1];											\
}

#define SUBSAMP_MAX_BACKWARD(input,inerror,outerror)											\
{																								\
	const int len0 = GETLENGTH(*(inerror)) / GETLENGTH(*(outerror));							\
	const int len1 = GETLENGTH(**(inerror)) / GETLENGTH(**(outerror));							\
	FOREACH(i, GETLENGTH(outerror))																\
	FOREACH(o0, GETLENGTH(*(outerror)))															\
	FOREACH(o1, GETLENGTH(**(outerror)))														\
	{																							\
		int x0 = 0, x1 = 0, ismax;																\
		FOREACH(l0, len0)																		\
			FOREACH(l1, len1)																	\
		{																						\
			ismax = input[i][o0*len0 + l0][o1*len1 + l1] > input[i][o0*len0 + x0][o1*len1 + x1];\
			x0 += ismax * (l0 - x0);															\
			x1 += ismax * (l1 - x1);															\
		}																						\
		inerror[i][o0*len0 + x0][o1*len1 + x1] = outerror[i][o0][o1];							\
	}																							\
}																								

#define DOT_PRODUCT_FORWARD(input,output,weight,bias,action)				\
{																			\
	for (int x = 0; x < GETLENGTH(weight); ++x)								\
		for (int y = 0; y < GETLENGTH(*weight); ++y)						\
			((double *)output)[y] += ((double *)input)[x] * weight[x][y];	\
	FOREACH(j, GETLENGTH(bias))												\
		((double *)output)[j] = action(((double *)output)[j] + bias[j]);	\
}

#define DOT_PRODUCT_FORWARD_GPU(input, output, weight, bias)											\
{																										\
	for (unsigned int i = 0; i < LAYER5; i++)															\
    {																									\
		((double*)output)[threadIdx.x] = __dadd_rn(((double*)output)[threadIdx.x], __dmul_rn(((double *)input)[i], weight[i][threadIdx.x]));		\
    }																									\
	((double*)output)[threadIdx.x] = relu(__dadd_rn(((double*)output)[threadIdx.x], bias[threadIdx.x]));			\
}																								

#define DOT_PRODUCT_BACKWARD(input,inerror,outerror,weight,wd,bd)	\
{																				\
	for (int x = 0; x < GETLENGTH(weight); ++x)									\
		for (int y = 0; y < GETLENGTH(*weight); ++y)							\
			((double *)inerror)[x] += ((double *)outerror)[y] * weight[x][y];	\
	FOREACH(i, GETCOUNT(inerror))												\
		((double *)inerror)[i] *= relugrad(((double *)input)[i]);				\
	FOREACH(j, GETLENGTH(outerror))												\
		bd[j] += ((double *)outerror)[j];										\
	for (int x = 0; x < GETLENGTH(weight); ++x)									\
		for (int y = 0; y < GETLENGTH(*weight); ++y)							\
			wd[x][y] += ((double *)input)[x] * ((double *)outerror)[y];			\
}																							

__device__ double relu(double x)
{
	return __dmul_rn(x, (double)(x > 0.0));
}

double relugrad(double y)
{
	return y > 0;
}

static void forward(LeNet5 *lenet, Feature *features, double(*action)(double))
{
	/*
	CONVOLUTION_FORWARD(features->input, features->layer1, lenet->weight0_1, lenet->bias0_1, action);
	SUBSAMP_MAX_FORWARD(features->layer1, features->layer2);
	CONVOLUTION_FORWARD(features->layer2, features->layer3, lenet->weight2_3, lenet->bias2_3, action);
	SUBSAMP_MAX_FORWARD(features->layer3, features->layer4);
	CONVOLUTION_FORWARD(features->layer4, features->layer5, lenet->weight4_5, lenet->bias4_5, action);
	DOT_PRODUCT_FORWARD(features->layer5, features->output, lenet->weight5_6, lenet->bias5_6, action);
	*/
}
/*
static void backward(LeNet5 *lenet, LeNet5 *deltas, Feature *errors, Feature *features)
{
	DOT_PRODUCT_BACKWARD(features->layer5, errors->layer5, errors->output, lenet->weight5_6, deltas->weight5_6, deltas->bias5_6);
	CONVOLUTION_BACKWARD(features->layer4, errors->layer4, errors->layer5, lenet->weight4_5, deltas->weight4_5, deltas->bias4_5);
	SUBSAMP_MAX_BACKWARD(features->layer3, errors->layer3, errors->layer4);
	CONVOLUTION_BACKWARD(features->layer2, errors->layer2, errors->layer3, lenet->weight2_3, deltas->weight2_3, deltas->bias2_3);
	SUBSAMP_MAX_BACKWARD(features->layer1, errors->layer1, errors->layer2);
	CONVOLUTION_BACKWARD(features->input, errors->input, errors->layer1, lenet->weight0_1, deltas->weight0_1, deltas->bias0_1);
}
*/
__device__ static inline void load_input(Feature *features, image input)
{
	double (*layer0)[LENGTH_FEATURE0][LENGTH_FEATURE0] = features->input;
	const long sz = sizeof(image) / sizeof(**input);
	double mean = 0, std = 0;
	FOREACH(j, sizeof(image) / sizeof(*input))
		FOREACH(k, sizeof(*input) / sizeof(**input))
	{
		mean += input[j][k];
		std += input[j][k] * input[j][k];
	}
	mean /= sz;
	std = sqrt(std / sz - mean*mean);
	FOREACH(j, sizeof(image) / sizeof(*input))
		FOREACH(k, sizeof(*input) / sizeof(**input))
	{
		layer0[0][j + PADDING][k + PADDING] = (input[j][k] - mean) / std;
	}
}

__device__ static inline void softmax(double input[OUTPUT], double loss[OUTPUT], int label, int count)
{
	double inner = 0;
	for (int i = 0; i < count; ++i)
	{
		double res = 0;
		for (int j = 0; j < count; ++j)
		{
			res += exp(input[j] - input[i]);
		}
		loss[i] = 1. / res;
		inner -= loss[i] * loss[i];
	}
	inner += loss[label];
	for (int i = 0; i < count; ++i)
	{
		loss[i] *= (i == label) - loss[i] - inner;
	}
}

__device__ static void load_target(Feature *features, Feature *errors, int label)
{
	double *output = (double *)features->output;
	double *error = (double *)errors->output;
	softmax(output, error, label, GETCOUNT(features->output));
}

static uint8 get_result(Feature *features, uint8 count)
{
	double *output = (double *)features->output; 
	const int outlen = GETCOUNT(features->output);
	uint8 result = 0;
	double maxvalue = *output;
	for (uint8 i = 1; i < count; ++i)
	{
		if (output[i] > maxvalue)
		{
			maxvalue = output[i];
			result = i;
		}
	}
	return result;
}
//Randomizes between -1 and 1.
static double f64rand()
{
	static int randbit = 0;
	if (!randbit)
	{
		srand((unsigned)time(0));
		for (int i = RAND_MAX; i; i >>= 1, ++randbit);
	}
	unsigned long long lvalue = 0x4000000000000000L;
	int i = 52 - randbit;
	for (; i > 0; i -= randbit)
		lvalue |= (unsigned long long)rand() << i;
	lvalue |= (unsigned long long)rand() >> -i;
	return *(double *)&lvalue - 3;
}

__global__ void forwardKernelLoadInput(
	Feature* featureArray,
	image* inputs
)
{
	load_input(&(featureArray[threadIdx.x]), inputs[threadIdx.x]);
	return;
}

__global__ void forwardKernelFirst(
	void** lenet,
	void** featureArray
)
{
	CONVOLUTION_FORWARD_GPU1(((Feature*)featureArray)[blockIdx.z].input, ((Feature*)featureArray)[blockIdx.z].layer1, ((LeNet5*)lenet)->weight0_1, ((LeNet5*)lenet)->bias0_1);
    return;
}

__global__ void forwardKernelFirstSubsamp(
	void** featureArray
)
{
	SUBSAMP_MAX_FORWARD_GPU(((Feature*)featureArray)[blockIdx.z].layer1, ((Feature*)featureArray)[blockIdx.z].layer2);
	return;
}

__global__ void forwardKernelSecond(
	void** lenet,
	void** featureArray
)
{
	CONVOLUTION_FORWARD_GPU2(((Feature*)featureArray)[blockIdx.z].layer2, ((Feature*)featureArray)[blockIdx.z].layer3, ((LeNet5*)lenet)->weight2_3, ((LeNet5*)lenet)->bias2_3);
    return;
}

__global__ void forwardKernelSecondSubsamp(
	void** featureArray
)
{
	SUBSAMP_MAX_FORWARD_GPU(((Feature*)featureArray)[blockIdx.z].layer3, ((Feature*)featureArray)[blockIdx.z].layer4);
    return;
}

//Kernel function run for the output.
__global__ void forwardKernelLast(
	void** lenet,
	void** featureArray
)
{
	CONVOLUTION_FORWARD_GPU3(((Feature*)featureArray)[blockIdx.z].layer4, ((Feature*)featureArray)[blockIdx.z].layer5, ((LeNet5*)lenet)->weight4_5, ((LeNet5*)lenet)->bias4_5);
    return;
}

__global__ void forwardKernelDot(
	void** lenet,
	void** featureArray
)
{
	DOT_PRODUCT_FORWARD_GPU(((Feature*)featureArray)[blockIdx.z].layer5, ((Feature*)featureArray)[blockIdx.z].output, ((LeNet5*)lenet)->weight5_6, ((LeNet5*)lenet)->bias5_6);
	return;
}

//Backward stuff
__global__ void backwardKernelLoadTarget(
	Feature* featureArray,
	Feature* errors,
	uint8* labels
)
{
	load_target(&(featureArray[threadIdx.x]), &(errors[threadIdx.x]), labels[threadIdx.x]);
	return;
}

void TrainBatch(
	LeNet5* lenet,
	Feature* featureArray,
	image* inputs,
	image* deviceInputs,
	uint8* labels,
	uint8* deviceLabels,
	int batchSize,
	LeNet5* deviceLenet,
	Feature* deviceFeatureArray,
	Feature* errors,
	Feature* deviceErrors
	
)
{
	//Set the allocated memory to 0 in the host & device.
	//Normal lenet is NOT set to 0 as they are our input.
	memset(featureArray, 0.0, sizeof(Feature) * batchSize);
	memset(errors, 0.0, sizeof(Feature) * batchSize);

	gpuErrchk(cudaMemset(deviceLenet, 0.0, sizeof(LeNet5)));
	gpuErrchk(cudaMemset(deviceFeatureArray, 0.0, sizeof(Feature) * batchSize));
	gpuErrchk(cudaMemset(deviceInputs, 0.0, sizeof(image) * batchSize));
	gpuErrchk(cudaMemset(deviceErrors, 0.0, sizeof(Feature) * batchSize));
	gpuErrchk(cudaMemset(deviceLabels, 0.0, sizeof(uint8) * batchSize));

    //For each training image load input into feature host array.
	//for (int i = 0; i < batchSize; ++i)
	//{
		//load_input(&(featureArray[i]), inputs[i]);
    //}
	gpuErrchk(cudaMemcpy(deviceInputs, inputs, sizeof(image) * batchSize, cudaMemcpyHostToDevice));
	//Copy the feature array to device.
	//gpuErrchk(cudaMemcpy(deviceFeatureArray, featureArray, sizeof(Feature) * batchSize, cudaMemcpyHostToDevice));
	//Copy the lenet input to device.
    gpuErrchk(cudaMemcpy(deviceLenet, lenet, sizeof(LeNet5), cudaMemcpyHostToDevice));
    
	gpuErrchk(cudaMemcpy(deviceLabels, labels, sizeof(uint8) * batchSize, cudaMemcpyHostToDevice));

	dim3 gridDims(1, 1, 1);
    dim3 blockDims(batchSize, 1, 1);
	forwardKernelLoadInput<<<gridDims, blockDims>>>(
		deviceFeatureArray,
		deviceInputs
	);

	gridDims.x = 1;
	gridDims.y = 6;
	gridDims.z = batchSize;
	blockDims.x = 28;
	blockDims.y = 28;
	blockDims.z = 1;
    //First forward propagation kernel calls
	//Third configuration parameter is for the dynamic array allocation within the kernel
    forwardKernelFirst<<<gridDims, blockDims>>>(
		(void**)deviceLenet,
		(void**)deviceFeatureArray
	);

	blockDims.x = 14;
	blockDims.y = 14;
	forwardKernelFirstSubsamp<<<gridDims, blockDims>>>(
		(void**)deviceFeatureArray
	);

	gridDims.y = 16;
	blockDims.x = 10;
	blockDims.y = 10;

	forwardKernelSecond<<<gridDims, blockDims>>>(
		(void**)deviceLenet,
		(void**)deviceFeatureArray
	);

	blockDims.x = 5;
	blockDims.y = 5;
	forwardKernelSecondSubsamp<<<gridDims, blockDims>>>(
		(void**)deviceFeatureArray
	);

	gridDims.y = 120;
	blockDims.x = 1;
	blockDims.y = 1;
	//deviceOutput does not the same dimensions, have to make a separate kernel call here.
	forwardKernelLast<<<gridDims, blockDims>>>(
		(void**)deviceLenet,
		(void**)deviceFeatureArray
	);

	gridDims.y = 1;
	blockDims.x = 10;
	blockDims.y = 1;
	forwardKernelDot<<<gridDims, blockDims>>>(
		(void**)deviceLenet,
		(void**)deviceFeatureArray
	);
	
	gpuErrchk(cudaDeviceSynchronize());
	//Copy output back.
	//gpuErrchk(cudaMemcpy(featureArray, deviceFeatureArray, sizeof(Feature) * batchSize, cudaMemcpyDeviceToHost));
	
	//Copy lenet back.
	//Not needed since lenet was not changed at all on the gpu.
	//cudaMemcpy(lenet, deviceLenet, sizeof(LeNet5), cudaMemcpyDeviceToHost);

	//Sequential backward propagation.
	double buffer[GETCOUNT(LeNet5)] = { 0 };
	int z = 0;

	gridDims.x = 1;
	gridDims.y = 1;
	gridDims.z = 1;
	blockDims.x = batchSize;
	blockDims.y = 1;
	blockDims.z = 1;
	backwardKernelLoadTarget<<<gridDims, blockDims>>>(
		deviceFeatureArray,
		deviceErrors,
		deviceLabels
	);
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaMemcpy(errors, deviceErrors, sizeof(Feature) * batchSize, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(featureArray, deviceFeatureArray, sizeof(Feature) * batchSize, cudaMemcpyDeviceToHost));
    for (z = 0; z < batchSize; ++z)
    {
        LeNet5	deltas = { 0 };
		//load_target(&(featureArray[z]), &(errors[z]), labels[z]);
		DOT_PRODUCT_BACKWARD(featureArray[z].layer5, errors[z].layer5, errors[z].output, lenet->weight5_6, deltas.weight5_6, deltas.bias5_6);
		CONVOLUTION_BACKWARD(featureArray[z].layer4, errors[z].layer4, errors[z].layer5, lenet->weight4_5, deltas.weight4_5, deltas.bias4_5);
		SUBSAMP_MAX_BACKWARD(featureArray[z].layer3, errors[z].layer3, errors[z].layer4);
		CONVOLUTION_BACKWARD(featureArray[z].layer2, errors[z].layer2, errors[z].layer3, lenet->weight2_3, deltas.weight2_3, deltas.bias2_3);
		SUBSAMP_MAX_BACKWARD(featureArray[z].layer1, errors[z].layer1, errors[z].layer2);
		CONVOLUTION_BACKWARD(featureArray[z].input, errors[z].input, errors[z].layer1, lenet->weight0_1, deltas.weight0_1, deltas.bias0_1);
		FOREACH(j, GETCOUNT(LeNet5))
			buffer[j] += ((double *)&deltas)[j];
	}
	double k = ALPHA / batchSize;
	FOREACH(g, GETCOUNT(LeNet5))
		((double *)lenet)[g] += k * buffer[g];
}

/*
void Train(LeNet5 *lenet, image input, uint8 label)
{
	Feature features = { 0 };
	Feature errors = { 0 };
	LeNet5 deltas = { 0 };
	load_input(&features, input);
	forward(lenet, &features, relu);
	load_target(&features, &errors, label);
	backward(lenet, &deltas, &errors, &features, relugrad);
	FOREACH(i, GETCOUNT(LeNet5))
		((double *)lenet)[i] += ALPHA * ((double *)&deltas)[i];
}
*/
uint8* PredictBatch(
	LeNet5* lenet,
	Feature* featureArray,
	image* inputs,
	image* deviceInputs,
	int batchSize,
	LeNet5* deviceLenet,
	Feature* deviceFeatureArray,
	uint8 count
)
{
	//Set the allocated memory to 0 in the host & device.
	//Normal lenet is NOT set to 0 as they are our input.
	memset(featureArray, 0.0, sizeof(Feature) * batchSize);

	gpuErrchk(cudaMemset(deviceLenet, 0.0, sizeof(LeNet5)));
	gpuErrchk(cudaMemset(deviceFeatureArray, 0.0, sizeof(Feature) * batchSize));
	gpuErrchk(cudaMemset(deviceInputs, 0.0, sizeof(image) * batchSize));
    //For each training image load input into feature host array.
	//for (int i = 0; i < batchSize; ++i)
	//{
		//load_input(&(featureArray[i]), inputs[i]);
    //}
	gpuErrchk(cudaMemcpy(deviceInputs, inputs, sizeof(image) * batchSize, cudaMemcpyHostToDevice));
	//Copy the feature array to device.
	//gpuErrchk(cudaMemcpy(deviceFeatureArray, featureArray, sizeof(Feature) * batchSize, cudaMemcpyHostToDevice));
	//Copy the lenet input to device.
    gpuErrchk(cudaMemcpy(deviceLenet, lenet, sizeof(LeNet5), cudaMemcpyHostToDevice));
    
	dim3 gridDims(1, 1, 1);
    dim3 blockDims(batchSize, 1, 1);
	forwardKernelLoadInput<<<gridDims, blockDims>>>(
		deviceFeatureArray,
		deviceInputs
	);

	gridDims.x = 1;
	gridDims.y = 6;
	gridDims.z = batchSize;
	blockDims.x = 28;
	blockDims.y = 28;
	blockDims.z = 1;
    //First forward propagation kernel calls
	//Third configuration parameter is for the dynamic array allocation within the kernel
    forwardKernelFirst<<<gridDims, blockDims>>>(
		(void**)deviceLenet,
		(void**)deviceFeatureArray
	);
	
	blockDims.x = 14;
	blockDims.y = 14;
	forwardKernelFirstSubsamp<<<gridDims, blockDims>>>(
		(void**)deviceFeatureArray
	);

	gridDims.y = 16;
	blockDims.x = 10;
	blockDims.y = 10;
	
	forwardKernelSecond<<<gridDims, blockDims>>>(
		(void**)deviceLenet,
		(void**)deviceFeatureArray
	);

	blockDims.x = 5;
	blockDims.y = 5;
	forwardKernelSecondSubsamp<<<gridDims, blockDims>>>(
		(void**)deviceFeatureArray
	);
	
	gridDims.y = 120;
	blockDims.x = 1;
	blockDims.y = 1;
	//deviceOutput does not the same dimensions, have to make a separate kernel call here.
	forwardKernelLast<<<gridDims, blockDims>>>(
		(void**)deviceLenet,
		(void**)deviceFeatureArray
	);

	gridDims.y = 1;
	blockDims.x = 10;
	blockDims.y = 1;
	forwardKernelDot<<<gridDims, blockDims>>>(
		(void**)deviceLenet,
		(void**)deviceFeatureArray
	);
	
	gpuErrchk(cudaDeviceSynchronize());
	//Copy output back.
	gpuErrchk(cudaMemcpy(featureArray, deviceFeatureArray, sizeof(Feature) * batchSize, cudaMemcpyDeviceToHost));
	
	//Copy lenet back.
	//Not needed since lenet was not changed at all on the gpu.
	//cudaMemcpy(lenet, deviceLenet, sizeof(LeNet5), cudaMemcpyDeviceToHost);

	uint8* returnValues = (uint8*)malloc(sizeof(uint8) * batchSize);
	for (unsigned int i = 0; i < batchSize; i++)
	{
		returnValues[i] = get_result(&(featureArray[i]), count);
	}
	return returnValues;
}
/*
uint8 Predict(LeNet5 *lenet, image input, uint8 count)
{
	Feature features = { 0 };
	load_input(&features, input);
	forward(lenet, &features);
	return get_result(&features, count);
}
*/
void Initial(LeNet5 *lenet)
{
	for (double *pos = (double *)lenet->weight0_1; pos < (double *)lenet->bias0_1; *pos++ = f64rand());
	for (double *pos = (double *)lenet->weight0_1; pos < (double *)lenet->weight2_3; *pos++ *= sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (INPUT + LAYER1))));
	for (double *pos = (double *)lenet->weight2_3; pos < (double *)lenet->weight4_5; *pos++ *= sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (LAYER2 + LAYER3))));
	for (double *pos = (double *)lenet->weight4_5; pos < (double *)lenet->weight5_6; *pos++ *= sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (LAYER4 + LAYER5))));
	for (double *pos = (double *)lenet->weight5_6; pos < (double *)lenet->bias0_1; *pos++ *= sqrt(6.0 / (LAYER5 + OUTPUT)));
	for (int *pos = (int *)lenet->bias0_1; pos < (int *)(lenet + 1); *pos++ = 0);
}

void PrintResult(int confusion_matrix[OUTPUT][OUTPUT])
{
	// Print the confusion matrix
	printf("%15sPredicted label\n%10s", " ", " ");
	for (int col = 0; col < 10; col++)
		printf("%6d", col);
	printf("%10s\n", "Total");
	for (int n = 0; n < 70; n++)
		printf("%s", "-");
	printf("\nTrue label\n");
	int row_labels = 0;
	int total = 0;
	for (int row = 0; row < 10; row++) {
		row_labels = 0;
		printf("%10d", row);
		for (int col = 0; col < 10; col++) {
			printf("%6d", confusion_matrix[row][col]);
			row_labels += confusion_matrix[row][col];
		}
		printf("%10d\n", row_labels);
		total += row_labels;
	}
	for (int n = 0; n < 70; n++)
		printf("%s", "-");
	printf("\n%67s = %10d\n", "Total number of input images tested", total);
	for (int n = 0; n < 70; n++)
		printf("%s", "-");
	printf("\n");
}
