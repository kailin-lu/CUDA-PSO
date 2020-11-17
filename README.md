# Particle Swarm Optimization with CUDA 

Testing several implementations of particle swarm optimization with different CUDA features. 

**Conclusion:** Minimize the number of reads and writes to device and host memory, prioritizing use of thread block memory for updating local variables. CUDA's unified memory features with memory hints can also provide additional performance in speed. 

| Implementation                              | Avg Runtime w/ 5000 particles |
|---------------------------------------------|-------------------------------|
| C++ (sequential)                            | 180ms                         |
| CUDA (Naive)                                | 730ms                         |
| CUDA (Unified Memory + thread memory usage) | 98ms                          |

