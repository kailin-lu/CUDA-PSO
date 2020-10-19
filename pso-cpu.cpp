#include <iostream>
#include <stdlib.h>
#include <cmath> 
#include <string>
#include <ctime> 

// Position struct contains x and y coordinates 
struct Position {
    float x, y; 

    std::string toString() {
        return "(" + std::to_string(x) + "," + std::to_string(y) + ")"; 
    }

    void operator+=(const Position& a) {
        x = x + a.x;
        y = y + a.y; 
    }

    void operator=(const Position& a) {
        x = a.x; 
        y = a.y; 
    }
}; 

// Particle struct has current location, best location and velocity 
struct Particle {
    Position best_position; 
    Position current_position; 
    Position velocity; 
    float current_value; 
};

float randomFloat(float low, float high); 
float calcValue(Position p); 
int getTeamBestIndex(Particle* particles, int N);
void updateVelocity(Particle &p, Position team_best_position, float w, float c_ind, float c_team);
void updatePosition(Particle &p);

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
float calcValue(Position p) {
    return pow(p.x, 2) + pow(p.y, 2); 
}

// Returns the index of the particle with the best position
int getTeamBestIndex(Particle* particles, int N) {
    int best_index = 0; 
    float current_team_best = particles[0].current_value; 
    for (int i = 1; i < N; i++) {
        if (particles[i].current_value < current_team_best) {
            best_index = i; 
            current_team_best = particles[i].current_value; 
        }
    }
    return best_index; 
}

// Calculate velocity for a particle 
void updateVelocity(Particle &p, Position team_best_position, float w, float c_ind, float c_team) {
    float r_ind = (float)rand() / RAND_MAX; 
    float r_team = (float)rand() / RAND_MAX;
    p.velocity.x = w * p.velocity.x + 
                   r_ind * c_ind * (p.best_position.x - p.current_position.x) + 
                   r_team * c_team * (team_best_position.x - p.current_position.x); 
    p.velocity.y = w * p.velocity.y + 
                   r_ind * c_ind * (p.best_position.y - p.current_position.y) + 
                   r_team * c_team * (team_best_position.y - p.current_position.y); 
}

// Updates current position, checks if best position and value need to be updated
void updatePosition(Particle &p) {
    p.current_position += p.velocity; 
    float newValue = calcValue(p.current_position); 
    if (newValue < p.current_value) {
        p.current_value = newValue; 
        p.best_position = p.current_position; 
    }
}


int main(void) {
    // Random seed 
    std::srand(std::time(NULL)); 

    // Initialize particles 
    Particle* h_particles = new Particle[N]; 

    for (int i = 0; i < N; i++) {
        // Random starting position
        h_particles[i].current_position.x = randomFloat(SEARCH_MIN, SEARCH_MAX); 
        h_particles[i].current_position.y = randomFloat(SEARCH_MIN, SEARCH_MAX); 
        h_particles[i].best_position.x = h_particles[i].current_position.x; 
        h_particles[i].best_position.y = h_particles[i].current_position.y; 
        h_particles[i].current_value = calcValue(h_particles[i].best_position); 
        // Random starting velocity 
        h_particles[i].velocity.x = randomFloat(SEARCH_MIN, SEARCH_MAX); 
        h_particles[i].velocity.y = randomFloat(SEARCH_MIN, SEARCH_MAX); 
    }

    // Calculate team best position and team best value 
    int team_best_index = getTeamBestIndex(h_particles, N); 
    Position team_best_position = h_particles[team_best_index].best_position; 
    float team_best_value = h_particles[team_best_index].current_value; 
    std::cout << "Starting Best: " << std::endl;
    std::cout << "Best Particle: " << team_best_index << std::endl; 
    std::cout << "Best value: " << team_best_value << std::endl; 
    std::cout << "Best position" << team_best_position.toString() << std::endl;

    // for timing 
    long start = std::clock(); 
    // For i in interations 
    for (int i = 0; i < ITERATIONS; i++) {
        // for each particle 
        for (int j = 0; j < N; j++) {
            // For each particle calculate velocity 
            updateVelocity(h_particles[i], team_best_position, w, c_ind, c_team);
            // Update position and particle best value + position
            updatePosition(h_particles[i]); 
        }
        // Calculate team best 
        team_best_index = getTeamBestIndex(h_particles, N); 
        team_best_position = h_particles[team_best_index].best_position; 
        team_best_value = h_particles[team_best_index].current_value; 
    }

    long stop = std::clock(); 
    long elapsed = stop - start; 

    // print results 
    std::cout << "Ending Best: " << std::endl;
    std::cout << "Team best value: " << team_best_value << std::endl;
    std::cout << "Team best position: " << team_best_position.toString() << std::endl; 
    
    std::cout << "Run time: " << elapsed << "ms" << std::endl;
    return 0; 
}