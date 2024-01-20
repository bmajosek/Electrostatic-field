#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
struct ParticleData {
	int* posX;
	int* posY;
	float* velX;
	float* velY;
	float* charge;
};
struct ParticleComparer {
	int* posX;
	int* posY;
	int windowWidth;
	int blockSizeField;

	ParticleComparer(int* _posX, int* _posY, int _windowWidth, int _blockSizeField)
		: posX(_posX), posY(_posY), windowWidth(_windowWidth), blockSizeField(_blockSizeField) {}

	__device__
		bool operator()(const int& a, const int& b) const {
		return (posY[a] / blockSizeField) == (posY[b] / blockSizeField)
			? posX[a] / blockSizeField < posX[b] / blockSizeField
			: (posY[a] / blockSizeField) < (posY[b] / blockSizeField);
	}
};