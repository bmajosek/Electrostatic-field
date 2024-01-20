#include <iostream>
#include <cmath>
#include <vector>
#include <chrono>
#include "GlobalVariables.hpp"
#include "cuda_runtime.h"
#include "CudaUtils.hpp"
#include "CpuSimulation.hpp"
#include "kernel.cuh"

// Jezeli jest więcej niż jeden argument przekazany do programu zostanie wówczas wywołana wersja na CPU
// Bezargumentowe wywołanie programu powoduje włączenie wersji na GPU
int main(int argc, char** argv) {
	bool isGPU = true;
	if (argc > 1) {
		isGPU = false;
	}
	// Inicjalizacja generatora liczb pseudolosowych
	srand(time(NULL));
	cudaError_t cudaStatus;

	// Pomiar czasu dla generowania danych
	auto start = std::chrono::high_resolution_clock::now();

	// Inicjalizacja wektorów danych cząstek
	std::vector<int> posX(NUMBER_OF_PARTICLES);
	std::vector<int> posY(NUMBER_OF_PARTICLES);
	std::vector<float> velX(NUMBER_OF_PARTICLES);
	std::vector<float> velY(NUMBER_OF_PARTICLES);
	std::vector<float> charge(NUMBER_OF_PARTICLES);

	// Inicjalizacja cząstek losowymi pozycjami, prędkościami i ładunkami
	for (int i = 0; i < NUMBER_OF_PARTICLES; ++i) {
		posX[i] = (rand() % INITIAL_WINDOW_WIDTH);
		posY[i] = (rand() % INITIAL_WINDOW_HEIGHT);
		velX[i] = (rand() % 2 == 0) ? -rand() % MAX_VELOCITY : rand() % MAX_VELOCITY;
		velY[i] = (rand() % 2 == 0) ? -rand() % MAX_VELOCITY : rand() % MAX_VELOCITY;
		charge[i] = (rand() % 2 == 0) ? SMALL_PROTON_CHARGE : SMALL_ELECTRON_CHARGE;
	}

	// Pomiar czasu dla generowania danych
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> duration = end - start;
	std::cout << "Wygenerowanie danych trwało: " << duration.count() << " sekund" << std::endl;

	// Inicjalizacja struktury ParticleData
	ParticleData particles;
	particles.posX = posX.data();
	particles.posY = posY.data();
	particles.velX = velX.data();
	particles.velY = velY.data();
	particles.charge = charge.data();

	if (isGPU) {
		// Wywołanie funkcji uruchamiającej jądra CUDA wersja na GPU
		cudaStatus = runCudaKernels(particles);

		// Obsługa błędów CUDA
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "CUDA run Cuda Kernels failed: %s\n", cudaGetErrorString(cudaStatus));
			return 1;
		}

		// Resetowanie urządzenia CUDA
		cudaStatus = cudaDeviceReset();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceReset failed!");
			return 1;
		}
	}

	else {
		// Wersja na CPU
		runCPUversion(particles);
	}

	return 0;
}