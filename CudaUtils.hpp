#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <SFML/Graphics.hpp>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/copy.h>
#include <thrust/binary_search.h>
#include "GlobalVariables.hpp"
#include "PixelStructures.hpp"

cudaError_t initializeCudaResources(ParticleData& particles, float*& d_Xintensity, float*& d_Yintensity, float*& d_BlockStrengthX, float*& d_BlockStrengthY, float*& d_BlockStrength, float*& d_intensity, sf::Uint8*& d_pixels, ParticleData& d_particles, dim3 blockSizeField, int windowWidth, int windowHeight, cudaEvent_t& startEvent, cudaEvent_t& stopEvent);
cudaError_t handleResize(sf::RenderWindow& window, int& windowWidth, int& windowHeight, dim3& gridSizeField, dim3 blockSizeField, sf::Uint8*& d_pixels, float*& d_intensity, float*& d_Xintensity, float*& d_Yintensity, float*& d_BlockStrengthX, float*& d_BlockStrengthY, float*& d_BlockStrength, thrust::device_vector<int>& blocksToFind, sf::Uint8*& h_pixels);
cudaError_t checkCudaError(cudaError_t cudaError, const char* message);