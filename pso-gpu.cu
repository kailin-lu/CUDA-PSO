#include <iostream>
#include <stdlib.h>
#include <cmath> 
#include <string>
#include <ctime> 
#include <cuda.h> 
#include <cuda_runtime.h>

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


const unsigned int N = 2000; 
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


// Returns the index of the particle with the best position
__global__ void updateTeamBestIndex(Particle *d_particles, float *d_team_best_value, int *d_team_best_index, int N) {
    *d_team_best_value = d_particles[0].best_value; 
    *d_team_best_index = 0; 
    for (int i = 1; i < N; i++) {
        if (d_particles[i].best_value < *d_team_best_value) {
            *d_team_best_value = d_particles[i].best_value; 
            *d_team_best_index = i; 
        }
    }
}

// Calculate velocity for a particle 
__device__ void updateParticleVelocity(Particle &p, Position team_best_position, float w, float c_ind, float c_team) {
    // float r_ind = (float)rand() / RAND_MAX; 
    // float r_team = (float)rand() / RAND_MAX;
    float r_ind = 0.5f; 
    float r_team = 0.5f; 
    p.velocity.x = w * p.velocity.x + 
                   r_ind * c_ind * (p.best_position.x - p.current_position.x) + 
                   r_team * c_team * (team_best_position.x - p.current_position.x); 
    p.velocity.y = w * p.velocity.y + 
                   r_ind * c_ind * (p.best_position.y - p.current_position.y) + 
                   r_team * c_team * (team_best_position.y - p.current_position.y); 
}

// Update velocity for all particles 
__global__ void updateVelocity(Particle* d_particles, int *d_team_best_index, float w, float c_ind, float c_team, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x; 
    if (idx < N) {
        updateParticleVelocity(d_particles[idx], d_particles[*d_team_best_index].best_position, w, c_ind, c_team);
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
    // Random seed 
    std::srand(std::time(NULL)); 

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

    // initialize variables to be copied back
    int *d_team_best_index; 
    float *d_team_best_value; 

    // Move team best value and team best index to the GPU 
    cudaMalloc((void **)&d_team_best_index, sizeof(int)); 
    cudaMalloc((void **)&d_team_best_value, sizeof(float)); 

    // Initialize team best index and value 
    updateTeamBestIndex<<<1,1>>>(d_particles, d_team_best_value, d_team_best_index, N); 

    // assign thread and blockcount 
    int threadCount = 256; 
    int blockCount = (N + threadCount - 1) / threadCount; 

    // for timing 
    long start = std::clock();
    // For i in interations 
    for (int i = 0; i < ITERATIONS; i++) {
        updateVelocity<<<blockCount, threadCount>>>(d_particles, d_team_best_index, w, c_ind, c_team, N); 
        updatePosition<<<blockCount, threadCount>>>(d_particles, N); 
        updateTeamBestIndex<<<1,1>>>(d_particles, d_team_best_value, d_team_best_index, N); 
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
    return 0; 
}