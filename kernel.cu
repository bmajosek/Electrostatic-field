#include "kernel.cuh"
using namespace std;

// Implementacja funkcji uruchamiającej jądra CUDA
cudaError_t runCudaKernels(ParticleData& particles) {
	cudaError_t cudaStatus;

	// Inicjalizacja zmiennych związanymi z oknem i renderowaniem
	int windowWidth = INITIAL_WINDOW_WIDTH;
	int windowHeight = INITIAL_WINDOW_HEIGHT;
	bool isPaused = false;
	sf::RenderWindow window(sf::VideoMode(windowWidth, windowHeight), "Electrostatic Field Simulation");
	window.setFramerateLimit(MAX_FPS);
	sf::View view;
	cudaEvent_t startEvent, stopEvent;
	int frameCount = 0;

	// Inicjalizacja widoku SFML
	view.setSize(static_cast<float>(windowWidth), static_cast<float>(windowHeight));
	view.setCenter(static_cast<float>(windowWidth) / 2, static_cast<float>(windowHeight) / 2);
	window.setView(view);

	// Inicjalizacja wymiarów bloków i siatki dla jąder CUDA
	dim3 blockSizeField(BLOCK_SIZE_FIELD, BLOCK_SIZE_FIELD);
	dim3 gridSizeField((windowWidth + blockSizeField.x - 1) / blockSizeField.x, (windowHeight + blockSizeField.y - 1) / blockSizeField.y);
	dim3 blockSize(BLOCK_SIZE);
	dim3 gridSize((NUMBER_OF_PARTICLES + blockSize.x - 1) / blockSize.x);

	// Inicjalizacja wskaźników na dane na GPU
	float* d_Xintensity, * d_Yintensity, * d_BlockStrengthX, * d_BlockStrengthY, * d_BlockStrength, * d_intensity;
	sf::Uint8* d_pixels;
	ParticleData d_particles;

	// Inicjalizacja wektora dla danych pikseli na CPU
	sf::Uint8* h_pixels = (sf::Uint8*)malloc(windowWidth * windowHeight * sizeof(uchar4));

	// Inicjalizacja wektora do przechowywania indeksów cząstek i bloków
	thrust::device_vector<int> blocksToFind(windowHeight * windowWidth / ((blockSizeField.x - 1) * (blockSizeField.y - 1)));
	thrust::device_vector<int> particleIndices(NUMBER_OF_PARTICLES);
	thrust::device_vector<int> blockIndices;

	// Inicjalizacja sekwencji indeksów cząstek
	thrust::sequence(particleIndices.begin(), particleIndices.end());

	// Inicjalizacja zasobów CUDA
	if ((cudaStatus = initializeCudaResources(particles, d_Xintensity, d_Yintensity, d_BlockStrengthX, d_BlockStrengthY, d_BlockStrength, d_intensity, d_pixels, d_particles, blockSizeField, windowWidth, windowHeight, startEvent, stopEvent)) != cudaSuccess) {
		fprintf(stderr, "CUDA resource initialization failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	// Rozpoczecie liczenia czasu
	cudaStatus = checkCudaError(cudaEventRecord(startEvent, 0), "starting CUDA event timer");
	if (cudaStatus != cudaSuccess) goto Error;
	// Główna pętla renderowania i aktualizacji
	while (window.isOpen()) {
		// Obsługa zdarzeń SFML
		sf::Event event;
		while (window.pollEvent(event)) {
			if (event.type == sf::Event::Closed) {
				window.close();
			}
			else if (event.type == sf::Event::Resized) {
				// Obsługa zmiany rozmiaru okna
				cudaStatus = checkCudaError(handleResize(window, windowWidth, windowHeight, gridSizeField, blockSizeField,
					d_pixels, d_intensity, d_Xintensity, d_Yintensity,
					d_BlockStrengthX, d_BlockStrengthY, d_BlockStrength, blocksToFind, h_pixels), "Resize handling failed: %s\n");

				if (cudaStatus != cudaSuccess) {
					goto Error;
				}
			}
			else if (event.type == sf::Event::KeyPressed) {
				// Zmiana stanu pauzy po naciśnięciu spacji
				if (event.key.code == sf::Keyboard::Space) {
					isPaused = !isPaused;
				}
			}
		}
		if (!isPaused)
		{
			// Sortowanie cząstek według bloków
			ParticleComparer particleComparer(d_particles.posX, d_particles.posY, windowWidth, blockSizeField.x);
			CalculateBlockNumberFunctor calculateBlockNumberFunctor(d_particles.posX, d_particles.posY, blockSizeField.x, windowWidth);
			thrust::sort(particleIndices.begin(), particleIndices.end(), particleComparer);

			blockIndices = thrust::device_vector<int>(NUMBER_OF_PARTICLES);
			thrust::sequence(blocksToFind.begin(), blocksToFind.end());
			thrust::transform(particleIndices.begin(), particleIndices.end(), blockIndices.begin(), calculateBlockNumberFunctor);
			thrust::device_vector<int> blockStartIndices(NUMBER_OF_PARTICLES);

			// Znajdowanie początków bloków
			thrust::lower_bound(blockIndices.begin(), blockIndices.end(), blocksToFind.begin(), blocksToFind.end(), blockStartIndices.begin(), thrust::less<int>());

			// Wywołanie jądra calculatePixels
			calculatePixels << <gridSizeField, blockSizeField >> > (windowHeight, windowWidth, d_particles, particleIndices.data().get(), blockStartIndices.data().get(), d_BlockStrengthX, d_BlockStrengthY, d_BlockStrength, d_Xintensity, d_Yintensity, d_intensity);

			// Sprawdzenie błędów CUDA
			cudaStatus = checkCudaError(cudaGetLastError(), "calculatePixels launch failed");
			if (cudaStatus != cudaSuccess) goto Error;

			cudaStatus = checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize returned error code after launching calculatePixels");
			if (cudaStatus != cudaSuccess) goto Error;

			// Wywołanie jądra visualizeField
			visualizeField << <gridSizeField, blockSizeField >> > (windowHeight, windowWidth, d_particles, d_pixels, d_BlockStrengthX, d_BlockStrengthY, d_BlockStrength, d_Xintensity, d_Yintensity, d_intensity);

			// Sprawdzenie błędów CUDA
			cudaStatus = checkCudaError(cudaGetLastError(), "visualizeField launch failed");
			if (cudaStatus != cudaSuccess) goto Error;

			cudaStatus = checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize returned error code after launching visualizeField");
			if (cudaStatus != cudaSuccess) goto Error;

			// Wywołanie jądra updateParticles
			updateParticles << <gridSize, blockSize >> > (d_particles, windowWidth, windowHeight, d_Xintensity, d_Yintensity, d_pixels);

			// Sprawdzenie błędów CUDA
			cudaStatus = checkCudaError(cudaGetLastError(), "updateParticles launch failed");
			if (cudaStatus != cudaSuccess) goto Error;

			cudaStatus = checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize returned error code after launching updateParticles");
			if (cudaStatus != cudaSuccess) goto Error;

			// Kopiowanie kolorów z GPU na CPU
			if ((cudaStatus = checkCudaError(cudaMemcpy(h_pixels, d_pixels, windowWidth * windowHeight * sizeof(uchar4), cudaMemcpyDeviceToHost), "copying colors to gpu")) != cudaSuccess)
				return cudaStatus;

			// Aktualizacja obrazu SFML
			sf::Image image;
			image.create(windowWidth, windowHeight, h_pixels);
			sf::Texture texture;
			texture.loadFromImage(image);
			sf::Sprite sprite(texture);

			// Rysowanie cząstek
			window.clear();
			window.draw(sprite);
			window.display();

			// Zliczanie klatek na sekundę
			++frameCount;

			// Jezeli doliczy MAX_FPS oblicza srednie FPS
			if (frameCount == MAX_FPS) {
				cudaStatus = checkCudaError(cudaEventRecord(stopEvent, 0), "stopping CUDA event timer");
				if (cudaStatus != cudaSuccess) goto Error;

				cudaStatus = checkCudaError(cudaEventSynchronize(stopEvent), "synchronizing CUDA event timer");
				if (cudaStatus != cudaSuccess) goto Error;
				float milliseconds;
				cudaStatus = checkCudaError(cudaEventElapsedTime(&milliseconds, startEvent, stopEvent), "calculating elapsed time");
				if (cudaStatus != cudaSuccess) goto Error;
				float avgFPS = frameCount / (milliseconds / 1000);
				cout << "FPS: " << avgFPS << " " << 1 << " FPS zajelo: " << milliseconds / (MAX_FPS * 1000) << endl;

				// Resetuj zegary i liczniki
				frameCount = 0;
				cudaStatus = checkCudaError(cudaEventRecord(startEvent, 0), "starting CUDA event timer");
				if (cudaStatus != cudaSuccess) goto Error;
			}
		}
	}

Error:
	// Zwolnienie pamięci na GPU i CPU
	cudaEventDestroy(startEvent);
	cudaEventDestroy(stopEvent);
	cudaFree(d_intensity);
	cudaFree(d_Xintensity);
	cudaFree(d_Yintensity);
	cudaFree(d_BlockStrength);
	cudaFree(d_BlockStrengthX);
	cudaFree(d_BlockStrengthY);
	cudaFree(d_particles.posX);
	cudaFree(d_particles.posY);
	cudaFree(d_particles.velX);
	cudaFree(d_particles.velY);
	cudaFree(d_particles.charge);
	cudaFree(d_pixels);
	free(h_pixels);

	return cudaStatus;
}

// Funkcja CUDA aktualizująca pozycje cząstek na podstawie natężenia pola elektrostatycznego
__global__ void updateParticles(ParticleData particles, int windowWidth, int windowHeight, float* Xintensity, float* Yintensity, sf::Uint8* d_pixels) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < NUMBER_OF_PARTICLES) {
		int tid = threadIdx.x;
		__shared__ int particlesPosX[BLOCK_SIZE];
		__shared__ int particlesPosY[BLOCK_SIZE];
		__shared__ float particlesVelX[BLOCK_SIZE];
		__shared__ float particlesVelY[BLOCK_SIZE];
		// Pobranie pozycji i prędkości
		particlesPosX[tid] = particles.posX[idx];
		particlesPosY[tid] = particles.posY[idx];
		// Obliczenie indeksu na siatce
		int pixelIndex = (particlesPosX[tid] + particlesPosY[tid] * windowWidth) % (windowHeight * windowWidth);
		particlesVelX[tid] = particles.velX[idx] + particles.charge[idx] * Xintensity[pixelIndex] * TIME_STEP;
		particlesVelY[tid] = particles.velY[idx] + particles.charge[idx] * Yintensity[pixelIndex] * TIME_STEP;

		// Obsługa kolizji z granicami okna
		if (particlesPosX[tid] <= PROTON || particlesPosX[tid] >= windowWidth - ELECTRON) {
			particlesVelX[tid] = -particlesVelX[tid] * BOUNDARY_COLLISION_FORCE;
		}
		if (particlesPosY[tid] <= 1 || particlesPosY[tid] >= windowHeight - 1) {
			particlesVelY[tid] = -particlesVelY[tid] * BOUNDARY_COLLISION_FORCE;
		}

		// Aktualizacja pozycji na podstawie prędkości
		particlesPosX[tid] += (particlesVelX[tid] < 0) ? floor(particlesVelX[tid] * TIME_STEP) : ceil(particlesVelX[tid] * TIME_STEP);
		particlesPosY[tid] += (particlesVelY[tid] < 0) ? floor(particlesVelY[tid] * TIME_STEP) : ceil(particlesVelY[tid] * TIME_STEP);

		// Kontrola pozycji w granicach okna
		particlesPosX[tid] = (particlesPosX[tid] < PROTON) ? PROTON : particlesPosX[tid];
		particlesPosX[tid] = (particlesPosX[tid] > windowWidth - ELECTRON) ? windowWidth - ELECTRON : particlesPosX[tid];
		particlesPosY[tid] = (particlesPosY[tid] < 1) ? 1 : particlesPosY[tid];
		particlesPosY[tid] = (particlesPosY[tid] > windowHeight - 1) ? windowHeight - 1 : particlesPosY[tid];
		// Ustawienie koloru piksela na podstawie pozycji cząstki
		int numberOfColor = COLORS * (particlesPosX[tid] + particlesPosY[tid] * windowWidth);
		d_pixels[numberOfColor] = MIN_COLOR;
		d_pixels[numberOfColor + 1] = MIN_COLOR;
		d_pixels[numberOfColor + 2] = MIN_COLOR;
		d_pixels[numberOfColor + 3] = MAX_COLOR;
		particles.posX[idx] = particlesPosX[tid];
		particles.posY[idx] = particlesPosY[tid];
		particles.velX[idx] = particlesVelX[tid];
		particles.velY[idx] = particlesVelY[tid];
	}
}

// Funkcja CUDA obliczająca intensywność pola elektrostatycznego dla pikseli wewnątrz bloków
__global__ void calculatePixels(int windowHeight, int windowWidth, ParticleData particles, int* particleIndices, int* blockStarts, float* blockStrengthX, float* blockStrengthY, float* blockStrength, float* Xintensity, float* Yintensity, float* d_intensity) {
	// Pobranie współrzędnych piksela
	int idx = blockIdx.x * blockDim.x + threadIdx.x, idy = blockIdx.y * blockDim.y + threadIdx.y;

	if (idx < windowWidth && idy < windowHeight) {
		int tid = threadIdx.x + threadIdx.y * blockDim.x;
		int bid = blockIdx.x + blockIdx.y * gridDim.x;
		int pixelIndex = idx + idy * windowWidth;
		// Inicjalizacja współdzielonych zmiennych dla bloku
		__shared__ float fieldStrengthInPixelX[BLOCK_SIZE_FIELD_SUM];
		__shared__ float fieldStrengthInPixelY[BLOCK_SIZE_FIELD_SUM];
		__shared__ float fieldStrengthInPixel[BLOCK_SIZE_FIELD_SUM];

		// Indeksy bloków w otoczeniu piksela
		int blockIndeciesX[] = { blockIdx.x - 1, blockIdx.x, blockIdx.x + 1 };
		int blockIndeciesY[] = { blockIdx.y - 1, blockIdx.y, blockIdx.y + 1 };

		// Inicjalizacja współdzielonych zmiennych dla wątku
		fieldStrengthInPixelX[tid] = 0.0f;
		fieldStrengthInPixelY[tid] = 0.0f;
		fieldStrengthInPixel[tid] = 0.0f;

		// Środek bloku w którym znajduje się watek
		int midX = blockIdx.x * blockDim.x + blockDim.x / 2;
		int midY = blockIdx.y * blockDim.y + blockDim.y / 2;

		// Iteracja po otoczeniu bloków
		for (int k = 0; k < 3; ++k) {
			for (int j = 0; j < 3; ++j) {
				int blockStartX = blockIndeciesX[k];
				int blockStartY = blockIndeciesY[j];

				// Sprawdzenie czy blok znajduje się w oknie
				if (blockStartX >= 0 && blockStartY >= 0 && blockStartX < gridDim.x && blockStartY < gridDim.y) {
					// Początkowy indeks cząstki w bloku
					int i = blockStarts[blockStartX + blockStartY * gridDim.x];

					// Iteracja po cząstkach w bloku
					while (i < NUMBER_OF_PARTICLES && particles.posX[particleIndices[i]] / blockDim.x == blockStartX && particles.posY[particleIndices[i]] / blockDim.y == blockStartY) {
						int dx = particles.posX[particleIndices[i]] - idx;
						int dy = particles.posY[particleIndices[i]] - idy;
						int distance = dx * dx + dy * dy;

						// Obliczenia intensywności pola dla piksela
						if (distance > 0) {
							int sqrtDistance = sqrtf(distance);
							fieldStrengthInPixelX[tid] += particles.charge[particleIndices[i]] * dx / (distance * sqrtDistance);
							fieldStrengthInPixelY[tid] += particles.charge[particleIndices[i]] * dy / (distance * sqrtDistance);
							fieldStrengthInPixel[tid] += particles.charge[particleIndices[i]] / distance;
						}

						// Obliczenia intensywności pola dla całego bloku
						if (tid == 0) {
							int dxMid = particles.posX[particleIndices[i]] - midX;
							int dyMid = particles.posY[particleIndices[i]] - midY;
							int distanceToMid = dxMid * dxMid + dyMid * dyMid;

							// Zaktualizowanie sumy intensywności w bloku
							if (i == bid && distanceToMid > 0) {
								int sqrtDistanceToMid = sqrtf(distanceToMid);
								blockStrengthX[bid] += particles.charge[particleIndices[i]] * dxMid / (distanceToMid * sqrtDistanceToMid);
								blockStrengthY[bid] += particles.charge[particleIndices[i]] * dyMid / (distanceToMid * sqrtDistanceToMid);
								blockStrength[bid] += particles.charge[particleIndices[i]] / (distanceToMid);
							}
						}
						++i;
					}
				}
			}
		}

		// Zapisanie obliczonych wartości do pamięci globalnej
		Xintensity[pixelIndex] = fieldStrengthInPixelX[tid];
		Yintensity[pixelIndex] = fieldStrengthInPixelY[tid];
		d_intensity[pixelIndex] = fieldStrengthInPixel[tid];
	}
}

// Funkcja CUDA wizualizująca pole elektrostatyczne na podstawie intensywności pola
__global__ void visualizeField(int windowHeight, int windowWidth, ParticleData particles, sf::Uint8* d_pixels, float* blockStrengthX, float* blockStrengthY, float* blockStrength, float* Xintensity, float* Yintensity, float* d_intensity) {
	// Pobranie współrzędnych piksela
	int idx = blockIdx.x * blockDim.x + threadIdx.x, idy = blockIdx.y * blockDim.y + threadIdx.y;

	if (idx < windowWidth && idy < windowHeight) {
		int tid = threadIdx.x + blockDim.x * threadIdx.y;
		// Indeks piksela w oknie
		int pixelIndex = idx + idy * windowWidth;

		// Inicjalizacja współdzielonych zmiennych dla wątku
		__shared__ float fieldStrengthInPixelX[BLOCK_SIZE_FIELD_SUM];
		__shared__ float fieldStrengthInPixelY[BLOCK_SIZE_FIELD_SUM];
		__shared__ float fieldStrengthInPixel[BLOCK_SIZE_FIELD_SUM];
		// Pobranie wartości intensywności pola z pamięci globalnej
		fieldStrengthInPixelX[tid] = Xintensity[pixelIndex];
		fieldStrengthInPixelY[tid] = Yintensity[pixelIndex];
		fieldStrengthInPixel[tid] = d_intensity[pixelIndex];

		// Ustalanie koloru piksela w zależności od pozycji, sprawdzenie czy jest to piksel przy krawedzi
		if (idx <= PROTON) {
			d_pixels[COLORS * pixelIndex] = MIN_COLOR;
			d_pixels[COLORS * pixelIndex + 1] = MIN_COLOR;
			d_pixels[COLORS * pixelIndex + 2] = MAX_COLOR;
			d_pixels[COLORS * pixelIndex + 3] = MAX_COLOR;
		}
		else if (idx >= windowWidth - ELECTRON) {
			d_pixels[COLORS * pixelIndex] = MAX_COLOR;
			d_pixels[COLORS * pixelIndex + 1] = MIN_COLOR;
			d_pixels[COLORS * pixelIndex + 2] = MIN_COLOR;
			d_pixels[COLORS * pixelIndex + 3] = MAX_COLOR;
		}
		else {
			// Iteracja po blokach w siatce
			for (int i = 0; i < gridDim.x; ++i) {
				for (int j = 0; j < gridDim.y; ++j) {
					// Środek bloku w oknie
					int midX = i * blockDim.x + blockDim.x / 2;
					int midY = j * blockDim.y + blockDim.y / 2;

					// Współrzędne odległości od środka bloku
					int dxMid = midX - idx;
					int dyMid = midY - idy;

					// Sprawdzenie, czy blok nie jest sąsiadem piksela
					if (!(i >= blockIdx.x - 1 && i <= blockIdx.x + 1 && j >= blockIdx.y - 1 && j <= blockIdx.y + 1)) {
						int distanceToMid = dxMid * dxMid + dyMid * dyMid;
						// Dodanie wpływu reszty bloków na intensywność pola
						if (!isnan(blockStrengthY[i + j * gridDim.x]) && distanceToMid > 0) {
							if (dyMid > 0)
								fieldStrengthInPixelY[tid] += blockStrengthY[i + j * gridDim.x] / (distanceToMid);
							else
								fieldStrengthInPixelY[tid] -= blockStrengthY[i + j * gridDim.x] / (distanceToMid);
							if (dxMid > 0)
								fieldStrengthInPixelX[tid] += blockStrengthX[i + j * gridDim.x] / (distanceToMid);
							else
								fieldStrengthInPixelX[tid] -= blockStrengthX[i + j * gridDim.x] / (distanceToMid);

							fieldStrengthInPixel[tid] += blockStrength[i + j * gridDim.x] / (distanceToMid);
						}
					}
				}
			}

			// Obliczenie dodatkowego natezenie z krawedzi ekranu
			fieldStrengthInPixel[tid] += PROTON_FORCE / pow(idx, 2) - ELECTRON_FORCE / pow(windowWidth - idx, 2);
			fieldStrengthInPixelX[tid] += PROTON_FORCE / pow(idx, 2) + ELECTRON_FORCE / pow(windowWidth - idx, 2);

			// Obliczenie kolorow pikseli
			if (fieldStrengthInPixel[tid] > 0) {
				d_pixels[COLORS * pixelIndex] = MIN_COLOR;
				d_pixels[COLORS * pixelIndex + 1] = MIN_COLOR;
				d_pixels[COLORS * pixelIndex + 2] = MAX_COLOR;
				d_pixels[COLORS * pixelIndex + 3] = INTENSITY_COLOR * fieldStrengthInPixel[tid] > MAX_COLOR ? MAX_COLOR : INTENSITY_COLOR * fieldStrengthInPixel[tid];
			}
			else {
				d_pixels[COLORS * pixelIndex] = MAX_COLOR;
				d_pixels[COLORS * pixelIndex + 1] = MIN_COLOR;
				d_pixels[COLORS * pixelIndex + 2] = MIN_COLOR;
				d_pixels[COLORS * pixelIndex + 3] = -INTENSITY_COLOR * fieldStrengthInPixel[tid] > MAX_COLOR ? MAX_COLOR : -INTENSITY_COLOR * fieldStrengthInPixel[tid];
			}

			Xintensity[pixelIndex] = fieldStrengthInPixelX[tid];
			Yintensity[pixelIndex] = fieldStrengthInPixelY[tid];
		}
	}
}

// Funkcja obliczająca numer bloku na podstawie współrzędnych cząstki
__device__ int calculateNumberOfBlock(const int* posX, const int* posY, int index, int blockSizeField, int windowWidth) {
	return posX[index] / blockSizeField + (posY[index] / blockSizeField) * ((windowWidth + blockSizeField - 1) / blockSizeField);
}