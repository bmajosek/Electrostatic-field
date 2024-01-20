#pragma once
#include <iostream>
#include <vector>
#include "GlobalVariables.hpp"
#include "CudaUtils.hpp"
#include <cmath>

// Deklaracja funkcji uruchamiaj¹cej j¹dra CUDA
cudaError_t runCudaKernels(ParticleData& particles);

// Deklaracje j¹der CUDA
__global__ void updateParticles(ParticleData particles, int windowWidth, int windowHeight, float* Xintensity, float* Yintensity, sf::Uint8* d_pixels);
__global__ void visualizeField(int windowHeight, int windowWidth, ParticleData particles, sf::Uint8* d_pixels, float* blockStrengthX, float* blockStrengthY, float* blockStrength, float* Xintensity, float* Yintensity, float* d_intensity);
__global__ void calculatePixels(int windowHeight, int windowWidth, ParticleData particles, int* particleIndices, int* blockStarts, float* blockStrengthX, float* blockStrengthY, float* blockStrength, float* Xintensity, float* Yintensity, float* d_intensity);
__device__ int calculateNumberOfBlock(const int* posX, const int* posY, int index, int blockSizeField, int windowWidth);

// Struktura funkcji obliczaj¹cej numer bloku dla danego indeksu cz¹stki
struct CalculateBlockNumberFunctor {
	const int* posX;
	const int* posY;
	int blockSizeField;
	int windowWidth;

	CalculateBlockNumberFunctor(const int* _posX, const int* _posY, int _blockSizeField, int _windowWidth)
		: posX(_posX), posY(_posY), blockSizeField(_blockSizeField), windowWidth(_windowWidth) {}

	__device__
		int operator()(const int particleIndex) const {
		return calculateNumberOfBlock(posX, posY, particleIndex, blockSizeField, windowWidth);
	}
};