
#include "cudaLib.cuh"

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ 
void saxpy_gpu (float* x, float* y, float scale, int size) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if(i<size) y[i] = (scale * x[i]) + y[i];
}

int runGpuSaxpy(int vectorSize) {

	std::cout << "Hello GPU Saxpy!\n";

	float * a, * b, * c;

	a = (float *) malloc(vectorSize * sizeof(float));
	b = (float *) malloc(vectorSize * sizeof(float));
	c = (float *) malloc(vectorSize * sizeof(float));

	if (a == NULL || b == NULL || c == NULL) {
		printf("Unable to malloc memory ... Exiting!");
		return -1;
	}

	vectorInit(a, vectorSize);
	vectorInit(b, vectorSize);

	std::memcpy(c, b, vectorSize * sizeof(float));
	float scale = 2.0f;

	#ifndef DEBUG_PRINT_DISABLE 
		printf("\n Adding vectors : \n");
		printf(" scale = %f\n", scale);
		printf(" a = { ");
		for (int i = 0; i < 15; ++i) {
			printf("%3.4f, ", a[i]);
		}
		printf(" ... }\n");
		printf(" b = { ");
		for (int i = 0; i < 15; ++i) {
			printf("%3.4f, ", b[i]);
		}
		printf(" ... }\n");
	#endif

	// Memory Transfer
	int size = vectorSize * sizeof(float);
	float* a_d, * c_d;
	cudaMalloc((void **) &a_d, size);
	cudaMemcpy(a_d, a, size, cudaMemcpyHostToDevice);
	cudaMalloc((void **) &c_d, size);
	cudaMemcpy(c_d, c, size, cudaMemcpyHostToDevice);


	dim3 DimGrid(vectorSize/256, 1, 1);
	if (vectorSize%256) DimGrid.x++;
	dim3 DimBlock(256, 1, 1);	

	// Run Saxpy
	saxpy_gpu<<<DimGrid,DimBlock>>>(a_d, c_d, scale, size);


	// Transfer C
	cudaMemcpy(c, c_d, size, cudaMemcpyDeviceToHost);
	#ifndef DEBUG_PRINT_DISABLE 
		printf(" c = { ");
		for (int i = 0; i < 15; ++i) {
			printf("%3.4f, ", c[i]);
		}
		printf(" ... }\n");
	#endif
	int errorCount = verifyVector(a, b, c, scale, vectorSize);
	std::cout << "Found " << errorCount << " / " << vectorSize << " errors \n";
	// Free Memory
	cudaFree(a_d); cudaFree(c_d); 


	return 0;
}

/* 
 Some helpful definitions

 generateThreadCount is the number of threads spawned initially. Each thread is responsible for sampleSize points. 
 *pSums is a pointer to an array that holds the number of 'hit' points for each thread. The length of this array is pSumSize.

 reduceThreadCount is the number of threads used to reduce the partial sums.
 *totals is a pointer to an array that holds reduced values.
 reduceSize is the number of partial sums that each reduceThreadCount reduces.

*/

__global__
void generatePoints (uint64_t * pSums, uint64_t pSumSize, uint64_t sampleSize) {
	//	Insert code here
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if(i >= pSumSize) return;
	// Setup RNG
	curandState_t rng;
	curand_init(clock64(), i, 0, &rng);
	float x, y;
	uint64_t hitCount = 0;

    // Get a new random value

	for (uint64_t idx = 0; idx < sampleSize; ++idx) {
		x = curand_uniform(&rng);
		y = curand_uniform(&rng);
		
		if (x * x + y * y <= 1.0f) {
			hitCount++;
		}
	}

	pSums[i] = hitCount;
}

__global__ 
void reduceCounts (uint64_t * pSums, uint64_t * totals, uint64_t pSumSize, uint64_t reduceSize) {
	//	Insert code here
}

int runGpuMCPi (uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {

	//  Check CUDA device presence
	int numDev;
	cudaGetDeviceCount(&numDev);
	if (numDev < 1) {
		std::cout << "CUDA device missing!\n";
		return -1;
	}

	auto tStart = std::chrono::high_resolution_clock::now();
		
	float approxPi = estimatePi(generateThreadCount, sampleSize, 
		reduceThreadCount, reduceSize);
	
	std::cout << "Estimated Pi = " << approxPi << "\n";

	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() << " seconds.";

	return 0;
}

double estimatePi(uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {
	uint64_t * pSums;
	pSums = (uint64_t *) malloc(generateThreadCount * sizeof(uint64_t));
	uint64_t * pSums_d;
	cudaMalloc((void **) &pSums_d, generateThreadCount * sizeof(uint64_t));

	dim3 DimGrid(generateThreadCount/256, 1, 1);
	if (generateThreadCount%256) DimGrid.x++;
	dim3 DimBlock(256, 1, 1);
	
	generatePoints<<<DimGrid,DimBlock>>>(pSums_d, generateThreadCount, sampleSize);

	cudaMemcpy(pSums, pSums_d, generateThreadCount * sizeof(uint64_t), cudaMemcpyDeviceToHost);
	cudaFree(pSums_d);

	uint64_t tSum = 0;

	for(int i = 0; i < generateThreadCount; i++){
		tSum += pSums[i];
	}
	
	double approxPi = 0.0;
	approxPi = (double) tSum / (double) (sampleSize * generateThreadCount) * 4.0;

	//      Insert code here
	// std::cout << "Sneaky, you are ...\n";
	// std::cout << "Compute pi, you must!\n";
	return approxPi;
}
