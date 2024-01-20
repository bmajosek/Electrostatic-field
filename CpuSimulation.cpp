#include "CpuSimulation.hpp"

void visualizeFieldCPU(int windowHeight, int windowWidth, ParticleData& particles, std::vector<PixelData>& h_pixels, float* x_intensity, float* y_intensity, float* h_intensity) {
	for (int i = 0; i < windowHeight; ++i) {
		for (int j = 0; j < windowWidth; ++j) {
			x_intensity[j + i * windowWidth] = 0;
			y_intensity[j + i * windowWidth] = 0;
			h_intensity[j + i * windowWidth] = 0;
		}
	}
	for (int i = 0; i < windowHeight; ++i) {
		for (int j = PROTON; j < windowWidth - ELECTRON; ++j) {
			for (int k = 0; k < NUMBER_OF_PARTICLES; ++k) {
				int dx = j - particles.posX[k];
				int dy = i - particles.posY[k];
				int distance = dx * dx + dy * dy;
				if (distance > 0) {
					x_intensity[j + i * windowWidth] += particles.charge[k] * dx / (distance * sqrtf(distance));
					y_intensity[j + i * windowWidth] += particles.charge[k] * dy / (distance * sqrtf(distance));
					h_intensity[j + i * windowWidth] += particles.charge[k] / distance;
				}
			}
			x_intensity[j + i * windowWidth] += ELECTRON_FORCE / pow(j, 2) + PROTON_FORCE / pow(windowWidth - j, 2);
			h_intensity[j + i * windowWidth] += -ELECTRON_FORCE / pow(j, 2) + PROTON_FORCE / pow(windowWidth - j, 2);
		}
	}
	for (int i = 0; i < windowHeight; ++i) {
		for (int j = PROTON; j < windowWidth - ELECTRON; ++j) {
			if (h_intensity[j + i * windowWidth] >= 0) {
				h_pixels[j + i * windowWidth].r = MIN_COLOR;
				h_pixels[j + i * windowWidth].g = MIN_COLOR;
				h_pixels[j + i * windowWidth].b = MAX_COLOR;
				h_pixels[j + i * windowWidth].a = INTENSITY_COLOR * h_intensity[j + i * windowWidth] > MAX_COLOR ? MAX_COLOR : INTENSITY_COLOR * h_intensity[j + i * windowWidth];
			}
			else {
				h_pixels[j + i * windowWidth].r = MAX_COLOR;
				h_pixels[j + i * windowWidth].g = MIN_COLOR;
				h_pixels[j + i * windowWidth].b = MIN_COLOR;
				h_pixels[j + i * windowWidth].a = -INTENSITY_COLOR * h_intensity[j + i * windowWidth] > MAX_COLOR ? MAX_COLOR : -INTENSITY_COLOR * h_intensity[j + i * windowWidth];
			}
		}
	}
	for (int i = 0; i < windowHeight; ++i) {
		for (int j = 0; j < PROTON; ++j) {
			h_pixels[j + i * windowWidth].r = MIN_COLOR;
			h_pixels[j + i * windowWidth].g = MIN_COLOR;
			h_pixels[j + i * windowWidth].b = MAX_COLOR;
			h_pixels[j + i * windowWidth].a = MAX_COLOR;
		}
	}
	for (int i = 0; i < windowHeight; ++i) {
		for (int j = windowWidth - ELECTRON; j < windowWidth; ++j) {
			h_pixels[j + i * windowWidth].r = MAX_COLOR;
			h_pixels[j + i * windowWidth].g = MIN_COLOR;
			h_pixels[j + i * windowWidth].b = MIN_COLOR;
			h_pixels[j + i * windowWidth].a = MAX_COLOR;
		}
	}
}

void updateParticlesCPU(ParticleData& particles, int windowWidth, int windowHeight, float* x_intensity, float* y_intensity, std::vector<PixelData>& h_pixels) {
	for (int idx = 0; idx < NUMBER_OF_PARTICLES; ++idx) {
		int posX = particles.posX[idx];
		int posY = particles.posY[idx];
		float charge = particles.charge[idx];

		// U¿ycie wczeœniej obliczonej intensywnoœci z visualizeField
		int pixelIndex = (posX + posY * windowWidth) % (windowWidth * windowHeight);
		float XfieldStrength = charge * x_intensity[pixelIndex];
		float YfieldStrength = charge * y_intensity[pixelIndex];

		// Aktualizacja pozycji cz¹stki
		particles.velX[idx] += XfieldStrength * TIME_STEP;
		particles.velY[idx] += YfieldStrength * TIME_STEP;

		// Obs³uga kolizji z granicami okna
		if (particles.posX[idx] <= PROTON || particles.posX[idx] >= windowWidth - ELECTRON) {
			particles.velX[idx] = -particles.velX[idx] * BOUNDARY_COLLISION_FORCE;
		}
		if (particles.posY[idx] <= 1 || particles.posY[idx] >= windowHeight - 1) {
			particles.velY[idx] = -particles.velY[idx] * BOUNDARY_COLLISION_FORCE;
		}

		// Aktualizacja pozycji na podstawie prêdkoœci
		particles.posX[idx] += (particles.velX[idx] < 0) ? floor(particles.velX[idx] * TIME_STEP) : ceil(particles.velX[idx] * TIME_STEP);
		particles.posY[idx] += (particles.velY[idx] < 0) ? floor(particles.velY[idx] * TIME_STEP) : ceil(particles.velY[idx] * TIME_STEP);

		// Kontrola pozycji w granicach okna
		particles.posX[idx] = (particles.posX[idx] < PROTON) ? PROTON : particles.posX[idx];
		particles.posX[idx] = (particles.posX[idx] > windowWidth - ELECTRON) ? windowWidth - ELECTRON : particles.posX[idx];
		particles.posY[idx] = (particles.posY[idx] < 1) ? 1 : particles.posY[idx];
		particles.posY[idx] = (particles.posY[idx] > windowHeight - 1) ? windowHeight - 1 : particles.posY[idx];

		// Ustawienie koloru piksela na podstawie pozycji cz¹stki
		h_pixels[particles.posX[idx] + particles.posY[idx] * windowWidth].r = sf::Color::Black.r;
		h_pixels[particles.posX[idx] + particles.posY[idx] * windowWidth].g = sf::Color::Black.g;
		h_pixels[particles.posX[idx] + particles.posY[idx] * windowWidth].b = sf::Color::Black.b;
		h_pixels[particles.posX[idx] + particles.posY[idx] * windowWidth].a = sf::Color::Black.a;
	}
}

void runCPUversion(ParticleData& particles)
{
	// Inicjalizacja zmiennych zwi¹zanymi z oknem i renderowaniem
	int windowWidth = INITIAL_WINDOW_WIDTH;
	int windowHeight = INITIAL_WINDOW_HEIGHT;
	sf::RenderWindow window(sf::VideoMode(windowWidth, windowHeight), "Electrostatic Field Simulation");
	window.setFramerateLimit(MAX_FPS);
	sf::View view;
	sf::Clock fpsClock;
	sf::Clock totalTimeClock;
	int frameCount = 0;
	float totalTime = 0.0f;
	float* x_intensity, * y_intensity, * intensity;
	x_intensity = (float*)malloc(sizeof(float) * windowHeight * windowWidth);
	y_intensity = (float*)malloc(sizeof(float) * windowHeight * windowWidth);
	intensity = (float*)malloc(sizeof(float) * windowHeight * windowWidth);
	std::vector<PixelData> h_pixels(windowWidth * windowHeight);
	// Inicjalizacja widoku SFML
	view.setSize(static_cast<float>(windowWidth), static_cast<float>(windowHeight));
	view.setCenter(static_cast<float>(windowWidth) / 2, static_cast<float>(windowHeight) / 2);
	window.setView(view);
	while (window.isOpen()) {
		// Obs³uga zdarzeñ SFML
		sf::Event event;
		while (window.pollEvent(event)) {
			if (event.type == sf::Event::Closed) {
				window.close();
			}
			else if (event.type == sf::Event::Resized) {
				// Obs³uga zmiany rozmiaru okna
				windowWidth = window.getSize().x;
				windowHeight = window.getSize().y;
				h_pixels = std::vector<PixelData>(windowWidth * windowHeight);
				free(x_intensity);
				free(y_intensity);
				free(intensity);
				x_intensity = (float*)malloc(sizeof(float) * windowHeight * windowWidth);
				y_intensity = (float*)malloc(sizeof(float) * windowHeight * windowWidth);
				intensity = (float*)malloc(sizeof(float) * windowHeight * windowWidth);
				view.setSize(static_cast<float>(windowWidth), static_cast<float>(windowHeight));
				view.setCenter(static_cast<float>(windowWidth) / 2, static_cast<float>(windowHeight) / 2);
				window.setView(view);
			}
		}

		// Obliczenie intensywnosci pola w kazdym pikselu
		visualizeFieldCPU(windowHeight, windowWidth, particles, h_pixels, x_intensity, y_intensity, intensity);
		// Obliczenie nowych pozycji i predkosci czasteczek
		updateParticlesCPU(particles, windowWidth, windowHeight, x_intensity, y_intensity, h_pixels);

		sf::Image image;
		image.create(windowWidth, windowHeight, reinterpret_cast<sf::Uint8*>(h_pixels.data()));
		sf::Texture texture;
		texture.loadFromImage(image);
		sf::Sprite sprite(texture);

		// Rysowanie cz¹stek
		window.clear();
		window.draw(sprite);
		window.display();

		// Zliczanie klatek na sekundê
		++frameCount;
		totalTime += fpsClock.restart().asSeconds();

		// Co 10 sekund wypisz œredni¹ iloœæ klatek na sekundê
		if (totalTimeClock.getElapsedTime().asSeconds() >= 10.0f) {
			float avgFPS = frameCount / totalTime;
			std::cout << "Average FPS: " << avgFPS << std::endl;

			// Resetuj zegary i liczniki
			frameCount = 0;
			totalTime = 0.0f;
			totalTimeClock.restart();
		}
	}
	free(x_intensity);
	free(y_intensity);
	free(intensity);
}