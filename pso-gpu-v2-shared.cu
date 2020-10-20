
/* 
Use CUDA functions to calculate block size 
*/ 

#include <iostream>
#include <stdlib.h>
#include <cmath> 
#include <string>
#include <ctime> 
#include <cuda.h> 
#include <cuda_runtime.h>
#include <curand_kernel.h>

// Position struct contains x and y coordinates 
struct Position {
    float x, y; 

    std::string toString() {
        return "(" + std::to_string(x) + "," + std::to_string(y) + ")"; 
    }

    __device__ __host__ void operator+=(const Position& a) {
        x = x + a.x;
        y = y + a.y; 
    }

    __device__ __host__ void operator=(const Position& a) {
        x = a.x; 
        y = a.y; 
    }
}; 

// Particle struct has current location, best location and velocity 
struct Particle {
    Position best_position; 
    Position current_position; 
    Position velocity; 
    float best_value; 
};


const unsigned int N = 5000; 
const unsigned int ITERATIONS = 1000; 
const float SEARCH_MIN = -1000.0f; 
const float SEARCH_MAX = 1000.0f; 
const float w = 0.9f; 
const float c_ind = 1.0f; 
const float c_team = 2.0f; 

// return a random float between low and high 
float randomFloat(float low, float high) {
    float range = high-low; 
    float pct = static_cast <float>(rand()) / static_cast <float>(RAND_MAX); 
    return low + pct * range; 
}

// function to optimize 
__device__ __host__ float calcValue(Position p) {
    return pow(p.x, 2) + pow(p.y, 2); 
}

// Initialize state for random numbers 
__global__ void init_kernel(curandState *state, long seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x; 
    curand_init(seed, idx, 0, state);
}

// Returns the index of the particle with the best position
__global__ void updateTeamBestIndex(Particle *d_particles, float *d_team_best_value, int *d_team_best_index, int N) {
    __shared__ float best_value; 
    __shared__ int best_index; 
    best_value = d_particles[0].best_value;
    best_index = 0; 
    __syncthreads(); 
    
    int idx = threadIdx.x + blockIdx.x * blockDim.x; 
    if (idx < N) {
        if (d_particles[idx].best_value < best_value) {
            best_value = d_particles[idx].best_value; 
            best_index = idx; 
            __syncthreads(); 
        }
    }
    *d_team_best_value = best_value; 
    *d_team_best_index = best_index; 
}


// Update velocity for all particles 
__global__ void updateVelocity(Particle* d_particles, int *d_team_best_index, float w, float c_ind, float c_team, int N, curandState *state) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x; 

    __shared__ float best_x, best_y; 
    best_x = d_particles[*d_team_best_index].best_position.x; 
    best_y = d_particles[*d_team_best_index].best_position.y; 
    __syncthreads(); 

    if (idx < N) {
        float r_ind = curand_uniform(state);
        float r_team = curand_uniform(state);
        d_particles[idx].velocity.x = w * d_particles[idx].velocity.x + 
                       r_ind * c_ind * (d_particles[idx].best_position.x - d_particles[idx].current_position.x) + 
                       r_team * c_team * (best_x - d_particles[idx].current_position.x); 
        d_particles[idx].velocity.y = w * d_particles[idx].velocity.y + 
                       r_ind * c_ind * (d_particles[idx].best_position.y - d_particles[idx].current_position.y) + 
                       r_team * c_team * (best_y - d_particles[idx].current_position.y); 
    }
}

__global__ void updatePosition(Particle *d_particles, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x; 
    if (idx < N) {
        d_particles[idx].current_position += d_particles[idx].velocity; 
        float newValue = calcValue(d_particles[idx].current_position); 
        if (newValue < d_particles[idx].best_value) {
            d_particles[idx].best_value = newValue; 
            d_particles[idx].best_position = d_particles[idx].current_position; 
        }
    }
}


int main(void) {
    // Random seed for cpu 
    std::srand(std::time(NULL)); 
    // Random seed for gpu 
    curandState *state; 
    cudaMalloc(&state, sizeof(curandState)); 
    init_kernel<<<1,1>>>(state, clock()); 

    // Initialize particles 
    Particle* h_particles = new Particle[N]; 
    Particle* d_particles;  // for the gpu 

    for (int i = 0; i < N; i++) {
        // Random starting position
        h_particles[i].current_position.x = randomFloat(SEARCH_MIN, SEARCH_MAX); 
        h_particles[i].current_position.y = randomFloat(SEARCH_MIN, SEARCH_MAX); 
        h_particles[i].best_position.x = h_particles[i].current_position.x; 
        h_particles[i].best_position.y = h_particles[i].current_position.y; 
        h_particles[i].best_value = calcValue(h_particles[i].best_position); 
        // Random starting velocity 
        h_particles[i].velocity.x = randomFloat(SEARCH_MIN, SEARCH_MAX); 
        h_particles[i].velocity.y = randomFloat(SEARCH_MIN, SEARCH_MAX); 
    }

    // Allocate memory + copy data to gpu 
    size_t particleSize = sizeof(Particle) * N; 
    cudaMalloc((void **)&d_particles, particleSize); 
    cudaMemcpy(d_particles, h_particles, particleSize, cudaMemcpyHostToDevice); // dest, source, size, direction

    // initialize variables for gpu 
    int *d_team_best_index; 
    float *d_team_best_value; 

    // Allocate gpu memory 
    cudaMalloc((void **)&d_team_best_index, sizeof(int)); 
    cudaMalloc((void **)&d_team_best_value, sizeof(float)); 

    // Initialize team best index and value 
    updateTeamBestIndex<<<1,1>>>(d_particles, d_team_best_value, d_team_best_index, N); 

    // assign thread and blockcount 
    int blockSize = 32; 
    int gridSize = (N + blockSize - 1) / blockSize; 

    // for timing 
    long start = std::clock();
    // For i in interations 
    for (int i = 0; i < ITERATIONS; i++) {
        updateVelocity<<<gridSize, blockSize>>>(d_particles, d_team_best_index, w, c_ind, c_team, N, state); 
        updatePosition<<<gridSize, blockSize>>>(d_particles, N); 
        updateTeamBestIndex<<<gridSize, blockSize>>>(d_particles, d_team_best_value, d_team_best_index, N); 
    }

    long stop = std::clock(); 
    long elapsed = (stop - start) * 1000 / CLOCKS_PER_SEC;

    // copy best particle back to host 
    int team_best_index; 
    cudaMemcpy(&team_best_index, d_team_best_index, sizeof(int), cudaMemcpyDeviceToHost); 
    
    // copy particle data back to host 
    cudaMemcpy(h_particles, d_particles, particleSize, cudaMemcpyDeviceToHost);

    // print results 
    std::cout << "Ending Best: " << std::endl;
    std::cout << "Team best value: " << h_particles[team_best_index].best_value << std::endl;
    std::cout << "Team best position: " << h_particles[team_best_index].best_position.toString() << std::endl; 
    
    std::cout << "Run time: " << elapsed << "ms" << std::endl;

    cudaFree(d_particles); 
    cudaFree(d_team_best_index); 
    cudaFree(d_team_best_value); 
    cudaFree(state); 
    return 0; 
}