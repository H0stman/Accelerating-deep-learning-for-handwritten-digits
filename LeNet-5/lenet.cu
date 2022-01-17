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

#include "lenet.h"
#include <memory.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#define GETLENGTH(array) (sizeof(array)/sizeof(*(array)))								\

#define GETCOUNT(array)  (sizeof(array)/sizeof(double))									\

#define FOREACH(i,count) for (int i = 0; i < count; ++i)								\

#define CONVOLUTE_VALID(input, output, weight)											\
{																						\
	FOREACH(o0, GETLENGTH(output))														\
    {																					\
		FOREACH(o1, GETLENGTH(*output))													\
        {																				\
			FOREACH(w0, GETLENGTH(weight))												\
            {																			\
				FOREACH(w1, GETLENGTH(*weight))											\
                {																		\
					output[o0][o1] += input[o0 + w0][o1 + w1] * weight[w0][w1];			\
                }																		\
            }																			\
        }																				\
    }																					\
}																						

#define CONVOLUTE_FULL(input,output,weight)												\
{																						\
	FOREACH(i0,GETLENGTH(input))														\
		FOREACH(i1,GETLENGTH(*(input)))													\
			FOREACH(w0,GETLENGTH(weight))												\
				FOREACH(w1,GETLENGTH(*(weight)))										\
					(output)[i0 + w0][i1 + w1] += (input)[i0][i1] * (weight)[w0][w1];	\
}																						\

// Similar functionality as the code in Figure 16.4 of the textbook
#define CONVOLUTION_FORWARD(input, output, weight, bias)								\
{																						\
	for (int x = 0; x < GETLENGTH(weight); ++x)											\
    {																					\
		for (int y = 0; y < GETLENGTH(*weight); ++y)									\
        {																				\
			CONVOLUTE_VALID(input[x], output[y], weight[x][y]);							\
        }																				\
    }																					\
	FOREACH(j, GETLENGTH(output))														\
    {																					\
		FOREACH(i, GETCOUNT(output[j]))													\
        {																				\
		    ((double*)output[j])[i] = reluMacro(((double*)output[j])[i] + bias[j]);		\
        }																				\
    }																					\
}																						

#define CONVOLUTION_BACKWARD(input,inerror,outerror,weight,wd,bd,actiongrad)			\
{																						\
	for (int x = 0; x < GETLENGTH(weight); ++x)											\
		for (int y = 0; y < GETLENGTH(*weight); ++y)									\
			CONVOLUTE_FULL(outerror[y], inerror[x], weight[x][y]);						\
	FOREACH(i, GETCOUNT(inerror))														\
		((double *)inerror)[i] *= actiongrad(((double *)input)[i]);						\
	FOREACH(j, GETLENGTH(outerror))														\
		FOREACH(i, GETCOUNT(outerror[j]))												\
		bd[j] += ((double *)outerror[j])[i];											\
	for (int x = 0; x < GETLENGTH(weight); ++x)											\
		for (int y = 0; y < GETLENGTH(*weight); ++y)									\
			CONVOLUTE_VALID(input[x], wd[x][y], outerror[y]);							\
}																						\

// Similar functionality as the code in Figure 16.5 of the textbook
#define SUBSAMP_MAX_FORWARD(input, output)												\
{																						\
	const int len0 = GETLENGTH(*input) / GETLENGTH(*output);							\
	const int len1 = GETLENGTH(**input) / GETLENGTH(**output);							\
	FOREACH(i, GETLENGTH(output))														\
    {																					\
	    FOREACH(o0, GETLENGTH(*output))													\
        {																				\
	        FOREACH(o1, GETLENGTH(**output))											\
	        {																			\
		        int x0 = 0, x1 = 0, ismax;												\
		        FOREACH(l0, len0)														\
                {																		\
			        FOREACH(l1, len1)													\
		            {																	\
			            ismax = input[i][o0 * len0 + l0][o1 * len1 + l1] > input[i][o0 * len0 + x0][o1 * len1 + x1];\
			            x0 += ismax * (l0 - x0);										\
			            x1 += ismax * (l1 - x1);										\
		            }																	\
                }																		\
		        output[i][o0][o1] = input[i][o0 * len0 + x0][o1 * len1 + x1];			\
	        }																			\
        }																				\
    }																					\
}																						\

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
}																								\

#define DOT_PRODUCT_FORWARD(input, output, weight, bias)										\
{																								\
	for (int x = 0; x < GETLENGTH(weight); ++x)													\
    {																							\
		for (int y = 0; y < GETLENGTH(*weight); ++y)											\
        {																						\
			((double*)output)[y] += ((double*)input)[x] * weight[x][y];							\
        }																						\
    }																							\
	FOREACH(j, GETLENGTH(bias))																	\
    {																							\
		((double*)output)[j] = reluMacro(((double*)output)[j] + bias[j]);						\
    }																							\
}																								\

#define DOT_PRODUCT_BACKWARD(input,inerror,outerror,weight,wd,bd,actiongrad)	\
{																				\
	for (int x = 0; x < GETLENGTH(weight); ++x)									\
		for (int y = 0; y < GETLENGTH(*weight); ++y)							\
			((double *)inerror)[x] += ((double *)outerror)[y] * weight[x][y];	\
	FOREACH(i, GETCOUNT(inerror))												\
		((double *)inerror)[i] *= actiongrad(((double *)input)[i]);				\
	FOREACH(j, GETLENGTH(outerror))												\
		bd[j] += ((double *)outerror)[j];										\
	for (int x = 0; x < GETLENGTH(weight); ++x)									\
		for (int y = 0; y < GETLENGTH(*weight); ++y)							\
			wd[x][y] += ((double *)input)[x] * ((double *)outerror)[y];			\
}																				\

#define reluMacro(x)		\
{							\
    x * (x > 0);			\
}							\

// double relu(double x)
// {
// 	return x*(x > 0);
// }

double relugrad(double y)
{
	return y > 0;
}

static void forward(LeNet5 *lenet, Feature *features)
{
	CONVOLUTION_FORWARD(features->input, features->layer1, lenet->weight0_1, lenet->bias0_1);
	SUBSAMP_MAX_FORWARD(features->layer1, features->layer2);
	CONVOLUTION_FORWARD(features->layer2, features->layer3, lenet->weight2_3, lenet->bias2_3);
	SUBSAMP_MAX_FORWARD(features->layer3, features->layer4);
	CONVOLUTION_FORWARD(features->layer4, features->layer5, lenet->weight4_5, lenet->bias4_5);
	DOT_PRODUCT_FORWARD(features->layer5, features->output, lenet->weight5_6, lenet->bias5_6);
}

static void backward(LeNet5 *lenet, LeNet5 *deltas, Feature *errors, Feature *features, double(*actiongrad)(double))
{
	DOT_PRODUCT_BACKWARD(features->layer5, errors->layer5, errors->output, lenet->weight5_6, deltas->weight5_6, deltas->bias5_6, actiongrad);
	CONVOLUTION_BACKWARD(features->layer4, errors->layer4, errors->layer5, lenet->weight4_5, deltas->weight4_5, deltas->bias4_5, actiongrad);
	SUBSAMP_MAX_BACKWARD(features->layer3, errors->layer3, errors->layer4);
	CONVOLUTION_BACKWARD(features->layer2, errors->layer2, errors->layer3, lenet->weight2_3, deltas->weight2_3, deltas->bias2_3, actiongrad);
	SUBSAMP_MAX_BACKWARD(features->layer1, errors->layer1, errors->layer2);
	CONVOLUTION_BACKWARD(features->input, errors->input, errors->layer1, lenet->weight0_1, deltas->weight0_1, deltas->bias0_1, actiongrad);
}

static inline void load_input(Feature *features, image input)
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

static inline void softmax(double input[OUTPUT], double loss[OUTPUT], int label, int count)
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

static void load_target(Feature *features, Feature *errors, int label)
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

//Kernel function.
__global__ void forwardKernel(
	double**** lenetWeight,
	double* lenetBias,
	struct cudaPitchedPtr* featureConvInput,
	struct cudaPitchedPtr* featureConvOutput,
	struct cudaPitchedPtr* featureSubsampOutput
)
{
	
	/*
    //First we copy everything from the input into shared memory if the thread is part of the first 6 blockIdx.y's.
	extern __shared__ double* sharedFeatures[];
	
	sharedFeatures[blockIdx.y].input[asdas][sadsa] = featureArray[blockIdx.y].input[asdasd][asdfas];
    if (blockIdx.y <= 6)
    {
		

        CONVOLUTION_FORWARD(featureConvInput, featureConvOutput, lenetWeight, lenetBias);
        if (threadIdx.x > 14 && threadIdx.y > 14)
        {
            return;
        }

        SUBSAMP_MAX_FORWARD(featureConvOutput, featureSubsampOutput);
    }
    __syncglobaldevice();

	CONVOLUTION_FORWARD(features->layer2, features->layer3, lenet->weight2_3, lenet->bias2_3);
	SUBSAMP_MAX_FORWARD(features->layer3, features->layer4);
	CONVOLUTION_FORWARD(features->layer4, features->layer5, lenet->weight4_5, lenet->bias4_5);
	DOT_PRODUCT_FORWARD(features->layer5, features->output, lenet->weight5_6, lenet->bias5_6);
    return;
	*/
}

//Kernel function run for the output.
__global__ void forwardKernelLast(
	double**** lenetWeight,
	double* lenetBias,
	struct cudaPitchedPtr* featureConvInput,
	struct cudaPitchedPtr* featureConvOutput,
	double* featureSubsampOutput
)
{
	/*
    //First we copy everything from the input into shared memory if the thread is part of the first 6 blockIdx.y's.
	extern __shared__ double* sharedFeatures[];
	
	sharedFeatures[blockIdx.y].input[asdas][sadsa] = featureArray[blockIdx.y].input[asdasd][asdfas];
    if (blockIdx.y <= 6)
    {
		

        CONVOLUTION_FORWARD(featureConvInput, featureConvOutput, lenetWeight, lenetBias);
        if (threadIdx.x > 14 && threadIdx.y > 14)
        {
            return;
        }

        SUBSAMP_MAX_FORWARD(featureConvOutput, featureSubsampOutput);
    }
    __syncglobaldevice();

	CONVOLUTION_FORWARD(features->layer2, features->layer3, lenet->weight2_3, lenet->bias2_3);
	SUBSAMP_MAX_FORWARD(features->layer3, features->layer4);
	CONVOLUTION_FORWARD(features->layer4, features->layer5, lenet->weight4_5, lenet->bias4_5);
	DOT_PRODUCT_FORWARD(features->layer5, features->output, lenet->weight5_6, lenet->bias5_6);
    return;
	*/
}

void TrainBatch(
	LeNet5* lenet,
	Feature* featureArray,
	image* inputs,
	uint8* labels,
	int batchSize,
	LeNet5* deviceLenet,
	struct cudaPitchedPtr* deviceInputCPU,
	struct cudaPitchedPtr* deviceLayer1CPU,
	struct cudaPitchedPtr* deviceLayer2CPU,
	struct cudaPitchedPtr* deviceLayer3CPU,
	struct cudaPitchedPtr* deviceLayer4CPU,
	struct cudaPitchedPtr* deviceLayer5CPU,
	double* deviceOutput
)
{
	//Set the allocated memory to 0 in the host & device.
	//Normal lenet and inputs are NOT set to 0 as they are our input.
	memset(featureArray, 0, sizeof(Feature) * batchSize);

	cudaMemset(deviceLenet, 0, sizeof(LeNet5));

	//Copy the cpu 2d array of 3d arrays to the gpu.
	struct cudaPitchedPtr* deviceInputGPU = 0;
	cudaMalloc((void**)&deviceInputGPU, batchSize * sizeof(cudaPitchedPtr));
	cudaMemcpy(deviceInputGPU, deviceInputCPU, batchSize * sizeof(cudaPitchedPtr), cudaMemcpyHostToDevice);
	struct cudaPitchedPtr* deviceLayer1GPU = 0;
	cudaMalloc((void**)&deviceLayer1GPU, batchSize * sizeof(cudaPitchedPtr));
	cudaMemcpy(deviceLayer1GPU, deviceLayer1CPU, batchSize * sizeof(cudaPitchedPtr), cudaMemcpyHostToDevice);
	struct cudaPitchedPtr* deviceLayer2GPU = 0;
	cudaMalloc((void**)&deviceLayer2GPU, batchSize * sizeof(cudaPitchedPtr));
	cudaMemcpy(deviceLayer2GPU, deviceLayer2CPU, batchSize * sizeof(cudaPitchedPtr), cudaMemcpyHostToDevice);
	struct cudaPitchedPtr* deviceLayer3GPU = 0;
	cudaMalloc((void**)&deviceLayer3GPU, batchSize * sizeof(cudaPitchedPtr));
	cudaMemcpy(deviceLayer3GPU, deviceLayer3CPU, batchSize * sizeof(cudaPitchedPtr), cudaMemcpyHostToDevice);
	struct cudaPitchedPtr* deviceLayer4GPU = 0;
	cudaMalloc((void**)&deviceLayer4GPU, batchSize * sizeof(cudaPitchedPtr));
	cudaMemcpy(deviceLayer4GPU, deviceLayer4CPU, batchSize * sizeof(cudaPitchedPtr), cudaMemcpyHostToDevice);
	struct cudaPitchedPtr* deviceLayer5GPU = 0;
	cudaMalloc((void**)&deviceLayer5GPU, batchSize * sizeof(cudaPitchedPtr));
	cudaMemcpy(deviceLayer5GPU, deviceLayer5CPU, batchSize * sizeof(cudaPitchedPtr), cudaMemcpyHostToDevice);

	int i = 0;
    //For each training image load input into feature host array.
	for (i = 0; i < batchSize; ++i)
	{
		load_input(&(featureArray[i]), inputs[i]);

		cudaMemcpy3DParms p = { 0 };
		p.kind = cudaMemcpyHostToDevice;

		p.srcPtr = make_cudaPitchedPtr(featureArray[i].input[0][0], INPUT * sizeof(double), INPUT, LENGTH_FEATURE0);
		p.dstPtr = deviceInputCPU[i];
		p.extent = make_cudaExtent(INPUT * sizeof(double), LENGTH_FEATURE0, LENGTH_FEATURE0);
		cudaMemcpy3D(&p);

		//Set the layers to 0.
		cudaMemset3D(deviceLayer1CPU[i], 0, make_cudaExtent(LAYER1 * sizeof(double), LENGTH_FEATURE1, LENGTH_FEATURE1));
		cudaMemset3D(deviceLayer2CPU[i], 0, make_cudaExtent(LAYER2 * sizeof(double), LENGTH_FEATURE2, LENGTH_FEATURE2));
		cudaMemset3D(deviceLayer3CPU[i], 0, make_cudaExtent(LAYER3 * sizeof(double), LENGTH_FEATURE3, LENGTH_FEATURE3));
		cudaMemset3D(deviceLayer4CPU[i], 0, make_cudaExtent(LAYER4 * sizeof(double), LENGTH_FEATURE4, LENGTH_FEATURE4));
		cudaMemset3D(deviceLayer5CPU[i], 0, make_cudaExtent(LAYER5 * sizeof(double), LENGTH_FEATURE5, LENGTH_FEATURE5));
    }
	//Set the output to 0.
	cudaMemset2D(deviceOutput, OUTPUT * sizeof(double), 0, OUTPUT * sizeof(double), batchSize);

	//Copy the lenet input to device.
    cudaMemcpy(deviceLenet, lenet, sizeof(LeNet5), cudaMemcpyHostToDevice);
    
	dim3 gridDims(1, 6, batchSize);
    dim3 blockDims(32, 32, 1);

    //First forward propagation kernel call
	//Third configuration parameter is for the dynamic array allocation within the kernel
    forwardKernel<<<gridDims, blockDims, sizeof(Feature) * batchSize>>>(
		(double****)deviceLenet->weight0_1, //Might have to redo to send the address?
		deviceLenet->bias0_1,
		deviceInputGPU,
		deviceLayer1GPU,
		deviceLayer2GPU
	);

	gridDims.y = 16;
	blockDims.x = 14;
	blockDims.y = 14;
	forwardKernel<<<gridDims, blockDims, sizeof(Feature) * batchSize>>>(
		(double****)deviceLenet->weight2_3, //Might have to redo to send the address?
		deviceLenet->bias2_3,
		deviceLayer2GPU,
		deviceLayer3GPU,
		deviceLayer4GPU
	);

	//REWRITE THE GRIDDIMS & BLOCKDIMS HERE!
	//deviceOutput does not the same dimensions, have to make a separate kernel call here.
	forwardKernelLast<<<gridDims, blockDims, sizeof(Feature) * batchSize>>>(
		(double****)deviceLenet->weight4_5, //Might have to redo to send the address?
		deviceLenet->bias4_5,
		deviceLayer4GPU,
		deviceLayer5GPU,
		deviceOutput 
	);

	//Copy the results back.
	//Create a temporary output 2d array on the cpu to copy back to.
	double** tempOutput = (double**)malloc(sizeof(double*) * batchSize);
	for (int i = 0; i < batchSize; i++)
	{
		tempOutput[i] = (double*)malloc(sizeof(double) * OUTPUT);
	}
	cudaMemcpy2D(tempOutput, OUTPUT * sizeof(double), deviceOutput, OUTPUT * sizeof(double), OUTPUT * sizeof(double), batchSize, cudaMemcpyDeviceToHost);
    
	//Copy lenet back.
	//Not needed since lenet was not changed at all on the gpu.
	//cudaMemcpy(lenet, deviceLenet, sizeof(LeNet5), cudaMemcpyDeviceToHost);

	//Sequential backward propagation.
	double buffer[GETCOUNT(LeNet5)] = { 0 };
    for (i = 0; i < batchSize; ++i)
    {
		//Copy all the layers back from the GPU. (Except the output layer, has already been done above the for loop.)
		cudaMemcpy3DParms p = { 0 };
		p.kind = cudaMemcpyDeviceToHost;

		//Input
		p.srcPtr = deviceInputCPU[i];
		p.dstPtr = make_cudaPitchedPtr(featureArray[i].input[0][0], INPUT * sizeof(double), INPUT, LENGTH_FEATURE0);
		p.extent = make_cudaExtent(INPUT * sizeof(double), LENGTH_FEATURE0, LENGTH_FEATURE0);
		cudaMemcpy3D(&p);

		//Layer1
		p.srcPtr = deviceLayer1CPU[i];
		p.dstPtr = make_cudaPitchedPtr(featureArray[i].layer1[0][0], LAYER1 * sizeof(double), LAYER1, LENGTH_FEATURE1);
		p.extent = make_cudaExtent(LAYER1 * sizeof(double), LENGTH_FEATURE1, LENGTH_FEATURE1);
		cudaMemcpy3D(&p);

		//Layer2
		p.srcPtr = deviceLayer2CPU[i];
		p.dstPtr = make_cudaPitchedPtr(featureArray[i].layer2[0][0], LAYER2 * sizeof(double), LAYER2, LENGTH_FEATURE2);
		p.extent = make_cudaExtent(LAYER2 * sizeof(double), LENGTH_FEATURE2, LENGTH_FEATURE2);
		cudaMemcpy3D(&p);

		//Layer3
		p.srcPtr = deviceLayer3CPU[i];
		p.dstPtr = make_cudaPitchedPtr(featureArray[i].layer3[0][0], LAYER3 * sizeof(double), LAYER3, LENGTH_FEATURE3);
		p.extent = make_cudaExtent(LAYER3 * sizeof(double), LENGTH_FEATURE3, LENGTH_FEATURE3);
		cudaMemcpy3D(&p);

		//Layer4
		p.srcPtr = deviceLayer4CPU[i];
		p.dstPtr = make_cudaPitchedPtr(featureArray[i].layer4[0][0], LAYER4 * sizeof(double), LAYER4, LENGTH_FEATURE4);
		p.extent = make_cudaExtent(LAYER4 * sizeof(double), LENGTH_FEATURE4, LENGTH_FEATURE4);
		cudaMemcpy3D(&p);

		//Layer5
		p.srcPtr = deviceLayer5CPU[i];
		p.dstPtr = make_cudaPitchedPtr(featureArray[i].layer5[0][0], LAYER5 * sizeof(double), LAYER5, LENGTH_FEATURE5);
		p.extent = make_cudaExtent(LAYER5 * sizeof(double), LENGTH_FEATURE5, LENGTH_FEATURE5);
		cudaMemcpy3D(&p);

		//Move output into the featureArray.
		for (int j = 0; j < OUTPUT; j++)
		{
			featureArray[i].output[j] = tempOutput[i][j];
		}
		
        LeNet5	deltas = { 0 };
        Feature errors = { 0 };
		load_target(&(featureArray[i]), &errors, labels[i]);
		backward(lenet, &deltas, &errors, &(featureArray[i]), relugrad); // Backpropagation
		FOREACH(j, GETCOUNT(LeNet5))
        {
            buffer[j] += ((double*)&deltas)[j];
        }
	}
	double k = ALPHA / batchSize;
	FOREACH(i, GETCOUNT(LeNet5))
    {
        ((double*)lenet)[i] += k * buffer[i];
    }
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
uint8 Predict(LeNet5 *lenet, image input,uint8 count)
{
	Feature features = { 0 };
	load_input(&features, input);
	forward(lenet, &features);
	return get_result(&features, count);
}

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