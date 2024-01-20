#include "CudaUtils.hpp"

cudaError_t initializeCudaResources(ParticleData& particles, float*& d_Xintensity, float*& d_Yintensity,
	float*& d_BlockStrengthX, float*& d_BlockStrengthY, float*& d_BlockStrength, float*& d_intensity, sf::Uint8*& d_pixels, ParticleData& d_particles, dim3 blockSizeField, int windowWidth, int windowHeight, cudaEvent_t& startEvent, cudaEvent_t& stopEvent) {
	cudaError_t cudaStatus;

	if ((cudaStatus = checkCudaError(cudaSetDevice(0), "setting CUDA device")) != cudaSuccess)
		return cudaStatus;

	// Alokowanie pamieci na GPU
	if ((cudaStatus = checkCudaError(cudaMalloc((void**)&d_BlockStrengthX, windowWidth * windowHeight / ((blockSizeField.x - 1) * (blockSizeField.y - 1)) * sizeof(float)), "allocating d_BlockStrengthX on device")) != cudaSuccess)
		return cudaStatus;
	if ((cudaStatus = checkCudaError(cudaMalloc((void**)&d_BlockStrengthY, windowWidth * windowHeight / ((blockSizeField.x - 1) * (blockSizeField.y - 1)) * sizeof(float)), "allocating d_BlockStrengthY on device")) != cudaSuccess)
		return cudaStatus;
	if ((cudaStatus = checkCudaError(cudaMalloc((void**)&d_BlockStrength, windowWidth * windowHeight / ((blockSizeField.x - 1) * (blockSizeField.y - 1)) * sizeof(float)), "allocating d_BlockStrength on device")) != cudaSuccess)
		return cudaStatus;
	if ((cudaStatus = checkCudaError(cudaMalloc((void**)&d_intensity, windowWidth * windowHeight * sizeof(float)), "allocating d_intensity on device")) != cudaSuccess)
		return cudaStatus;
	if ((cudaStatus = checkCudaError(cudaMalloc((void**)&d_Xintensity, windowWidth * windowHeight * sizeof(float)), "allocating d_Xintensity on device")) != cudaSuccess)
		return cudaStatus;
	if ((cudaStatus = checkCudaError(cudaMalloc((void**)&d_Yintensity, windowWidth * windowHeight * sizeof(float)), "allocating d_Yintensity on device")) != cudaSuccess)
		return cudaStatus;

	if ((cudaStatus = checkCudaError(cudaMalloc((void**)&d_particles.posX, NUMBER_OF_PARTICLES * sizeof(int)), "allocating d_particles.posX on device")) != cudaSuccess)
		return cudaStatus;
	if ((cudaStatus = checkCudaError(cudaMalloc((void**)&d_particles.posY, NUMBER_OF_PARTICLES * sizeof(int)), "allocating d_particles.posY on device")) != cudaSuccess)
		return cudaStatus;
	if ((cudaStatus = checkCudaError(cudaMalloc((void**)&d_particles.velX, NUMBER_OF_PARTICLES * sizeof(float)), "allocating d_particles.velX on device")) != cudaSuccess)
		return cudaStatus;
	if ((cudaStatus = checkCudaError(cudaMalloc((void**)&d_particles.velY, NUMBER_OF_PARTICLES * sizeof(float)), "allocating d_particles.velY on device")) != cudaSuccess)
		return cudaStatus;
	if ((cudaStatus = checkCudaError(cudaMalloc((void**)&d_particles.charge, NUMBER_OF_PARTICLES * sizeof(float)), "allocating d_particles.charge on device")) != cudaSuccess)
		return cudaStatus;

	if ((cudaStatus = checkCudaError(cudaMemcpy(d_particles.posX, particles.posX, NUMBER_OF_PARTICLES * sizeof(int), cudaMemcpyHostToDevice), "copying particles.posX to device")) != cudaSuccess)
		return cudaStatus;
	if ((cudaStatus = checkCudaError(cudaMemcpy(d_particles.posY, particles.posY, NUMBER_OF_PARTICLES * sizeof(int), cudaMemcpyHostToDevice), "copying particles.posY to device")) != cudaSuccess)
		return cudaStatus;
	if ((cudaStatus = checkCudaError(cudaMemcpy(d_particles.velX, particles.velX, NUMBER_OF_PARTICLES * sizeof(float), cudaMemcpyHostToDevice), "copying particles.velX to device")) != cudaSuccess)
		return cudaStatus;
	if ((cudaStatus = checkCudaError(cudaMemcpy(d_particles.velY, particles.velY, NUMBER_OF_PARTICLES * sizeof(float), cudaMemcpyHostToDevice), "copying particles.velY to device")) != cudaSuccess)
		return cudaStatus;
	if ((cudaStatus = checkCudaError(cudaMemcpy(d_particles.charge, particles.charge, NUMBER_OF_PARTICLES * sizeof(float), cudaMemcpyHostToDevice), "copying particles.charge to device")) != cudaSuccess)
		return cudaStatus;

	if ((cudaStatus = checkCudaError(cudaMalloc((void**)&d_pixels, windowWidth * windowHeight * sizeof(uchar4)), "allocating d_pixels on device")) != cudaSuccess)
		return cudaStatus;

	if ((cudaStatus = checkCudaError(cudaEventCreate(&startEvent), "creating startEvent")) != cudaSuccess)
		return cudaStatus;

	if ((cudaStatus = checkCudaError(cudaEventCreate(&stopEvent), "creating stopEvent")) != cudaSuccess)
		return cudaStatus;

	return cudaStatus;
}

cudaError_t handleResize(sf::RenderWindow& window, int& windowWidth, int& windowHeight, dim3& gridSizeField, dim3 blockSizeField, sf::Uint8*& d_pixels, float*& d_intensity, float*& d_Xintensity, float*& d_Yintensity,
	float*& d_BlockStrengthX, float*& d_BlockStrengthY, float*& d_BlockStrength, thrust::device_vector<int>& blocksToFind, sf::Uint8*& h_pixels) {
	cudaError_t cudaStatus;
	windowWidth = window.getSize().x;
	windowHeight = window.getSize().y;
	gridSizeField = dim3((windowWidth + blockSizeField.x - 1) / blockSizeField.x, (windowHeight + blockSizeField.y - 1) / blockSizeField.y);

	// Zwolnienie pamieci
	if ((cudaStatus = checkCudaError(cudaFree(d_pixels), "freeing d_pixels")) != cudaSuccess)
		return cudaStatus;
	if ((cudaStatus = checkCudaError(cudaFree(d_intensity), "freeing d_intensity")) != cudaSuccess)
		return cudaStatus;
	if ((cudaStatus = checkCudaError(cudaFree(d_Xintensity), "freeing d_Xintensity")) != cudaSuccess)
		return cudaStatus;
	if ((cudaStatus = checkCudaError(cudaFree(d_Yintensity), "freeing d_Yintensity")) != cudaSuccess)
		return cudaStatus;
	if ((cudaStatus = checkCudaError(cudaFree(d_BlockStrengthX), "freeing d_BlockStrengthX")) != cudaSuccess)
		return cudaStatus;
	if ((cudaStatus = checkCudaError(cudaFree(d_BlockStrengthY), "freeing d_BlockStrengthY")) != cudaSuccess)
		return cudaStatus;
	if ((cudaStatus = checkCudaError(cudaFree(d_BlockStrength), "freeing d_BlockStrength")) != cudaSuccess)
		return cudaStatus;

	// Zaalokowanie pamieci na gpu
	if ((cudaStatus = checkCudaError(cudaMalloc((void**)&d_pixels, windowWidth * windowHeight * sizeof(uchar4)), "allocating d_pixels on device")) != cudaSuccess)
		return cudaStatus;
	if ((cudaStatus = checkCudaError(cudaMalloc((void**)&d_BlockStrengthX, windowWidth * windowHeight / ((blockSizeField.x - 1) * (blockSizeField.y - 1)) * sizeof(float)), "allocating d_BlockStrengthX on device")) != cudaSuccess)
		return cudaStatus;
	if ((cudaStatus = checkCudaError(cudaMalloc((void**)&d_BlockStrengthY, windowWidth * windowHeight / ((blockSizeField.x - 1) * (blockSizeField.y - 1)) * sizeof(float)), "allocating d_BlockStrengthY on device")) != cudaSuccess)
		return cudaStatus;
	if ((cudaStatus = checkCudaError(cudaMalloc((void**)&d_BlockStrength, windowWidth * windowHeight / ((blockSizeField.x - 1) * (blockSizeField.y - 1)) * sizeof(float)), "allocating d_BlockStrength on device")) != cudaSuccess)
		return cudaStatus;
	if ((cudaStatus = checkCudaError(cudaMalloc((void**)&d_intensity, windowWidth * windowHeight * sizeof(float)), "allocating d_intensity on device")) != cudaSuccess)
		return cudaStatus;
	if ((cudaStatus = checkCudaError(cudaMalloc((void**)&d_Xintensity, windowWidth * windowHeight * sizeof(float)), "allocating d_Xintensity on device")) != cudaSuccess)
		return cudaStatus;
	if ((cudaStatus = checkCudaError(cudaMalloc((void**)&d_Yintensity, windowWidth * windowHeight * sizeof(float)), "allocating d_Yintensity on device")) != cudaSuccess)
		return cudaStatus;

	// Zwolnienie i zaalokowanie pamieci na cpu
	free(h_pixels);
	if ((h_pixels = (sf::Uint8*)malloc(windowWidth * windowHeight * sizeof(uchar4))) == nullptr)
		return cudaStatus;

	// Stworzenie nowego wektora blokow
	blocksToFind = thrust::device_vector<int>(windowHeight * windowWidth / ((blockSizeField.x - 1) * (blockSizeField.y - 1)));
	thrust::sequence(blocksToFind.begin(), blocksToFind.end());

	// Zaktualizowanie wielkosci okna
	sf::View view;
	view.setSize(static_cast<float>(windowWidth), static_cast<float>(windowHeight));
	view.setCenter(static_cast<float>(windowWidth) / 2, static_cast<float>(windowHeight) / 2);
	window.setView(view);

	return cudaSuccess;
}

cudaError_t checkCudaError(cudaError_t cudaError, const char* message) {
	if (cudaError != cudaSuccess) {
		fprintf(stderr, "CUDA error %s: %s\n", message, cudaGetErrorString(cudaError));
		return cudaError;
	}
	return cudaError;
}