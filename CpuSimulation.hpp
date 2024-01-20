#pragma once
#include "GlobalVariables.hpp"
#include "PixelStructures.hpp"
#include <SFML/Graphics.hpp>
#include <vector>
#include <iostream>

struct PixelData
{
	unsigned char r;
	unsigned char g;
	unsigned char b;
	unsigned char a;
};

void updateParticlesCPU(ParticleData& particles, int windowWidth, int windowHeight, float* x_intensity, float* y_intensity, std::vector<PixelData>& h_pixels);
void visualizeFieldCPU(int windowHeight, int windowWidth, ParticleData& particles, std::vector<PixelData>& h_pixels, float* x_intensity, float* y_intensity, float* h_intensity);
void runCPUversion(ParticleData& particles);