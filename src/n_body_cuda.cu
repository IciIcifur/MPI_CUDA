#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>

#ifdef _WIN32
  #include <direct.h>
  #define mkdir _mkdir
#else
  #include <sys/stat.h>
  #include <sys/types.h>
#endif

#ifndef G
#define G 6.67430e-11
#endif

#define DEFAULT_DT 0.01
#define DEFAULT_TEND 1.0
#define DEFAULT_EPS 1e-5
#define BLOCK_SIZE 256

typedef struct {
    double mass;
    double x, y;
    double vx, vy;
    double fx, fy;
} Particle;

__global__ void reset_forces_kernel(Particle* particles, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        particles[idx].fx = 0.0;
        particles[idx].fy = 0.0;
    }
}

__global__ void compute_forces_kernel(Particle* particles, int n, double eps) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= n) return;

    double eps2 = eps * eps;
    
    Particle p_i = particles[i];

    for (int j = i + 1; j < n; j++) {
        Particle p_j = particles[j];

        double dx = p_j.x - p_i.x;
        double dy = p_j.y - p_i.y;
        double r2 = dx*dx + dy*dy + eps2;

        if (r2 < 1e-10) continue;

        double r = sqrt(r2);
        double inv_r3 = 1.0 / (r2 * r);
        double coef = G * p_i.mass * p_j.mass * inv_r3;

        double fx_ij = coef * dx;
        double fy_ij = coef * dy;

        atomicAdd(&particles[i].fx, fx_ij);
        atomicAdd(&particles[i].fy, fy_ij);

        atomicAdd(&particles[j].fx, -fx_ij);
        atomicAdd(&particles[j].fy, -fy_ij);
    }
}

__global__ void euler_step_kernel(Particle* particles, int n, double dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    double ax = particles[idx].fx / particles[idx].mass;
    double ay = particles[idx].fy / particles[idx].mass;

    particles[idx].vx += ax * dt;
    particles[idx].vy += ay * dt;

    particles[idx].x += particles[idx].vx * dt;
    particles[idx].y += particles[idx].vy * dt;
}

static void create_directory(const char* path) {
#ifdef _WIN32
    _mkdir(path);
#else
    mkdir(path, 0777);
#endif
}

int read_input_file(const char* filename, Particle** particles, int* n) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "ERROR: Cannot open file %s\n", filename);
        return 0;
    }

    if (fscanf(fp, "%d", n) != 1) {
        fprintf(stderr, "ERROR: Cannot read number of particles\n");
        fclose(fp);
        return 0;
    }

    if (*n <= 0) {
        fprintf(stderr, "ERROR: Invalid number of particles: %d\n", *n);
        fclose(fp);
        return 0;
    }

    *particles = (Particle*)malloc(sizeof(Particle) * (*n));
    if (!*particles) {
        fprintf(stderr, "ERROR: Memory allocation failed\n");
        fclose(fp);
        return 0;
    }

    for (int i = 0; i < *n; ++i) {
        double m,x,y,vx,vy;
        if (fscanf(fp, "%lf %lf %lf %lf %lf", &m, &x, &y, &vx, &vy) != 5) {
            fprintf(stderr, "ERROR: Cannot read data for particle %d\n", i+1);
            fclose(fp);
            free(*particles);
            return 0;
        }
        (*particles)[i].mass = m;
        (*particles)[i].x = x;
        (*particles)[i].y = y;
        (*particles)[i].vx = vx;
        (*particles)[i].vy = vy;
        (*particles)[i].fx = 0.0;
        (*particles)[i].fy = 0.0;
    }

    fclose(fp);
    return 1;
}

int validate_parameters(double t_end, double dt, int n) {
    if (t_end <= 0) { fprintf(stderr, "ERROR: t_end must be positive\n"); return 0; }
    if (dt <= 0) { fprintf(stderr, "ERROR: dt must be positive\n"); return 0; }
    if (dt > t_end) { fprintf(stderr, "ERROR: dt > t_end\n"); return 0; }
    if (n <= 0) { fprintf(stderr, "ERROR: n must be positive\n"); return 0; }
    return 1;
}

void check_cuda_error(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", context, cudaGetErrorString(err));
        exit(1);
    }
}

int main(int argc, char* argv[]) {
    printf("N-Body (2D) - CUDA - Fixed Version\n");
    printf("Usage: %s <t_end> <input_file> [dt] [eps]\n", argv[0]);
    printf(" input: first line n, next n lines: mass x y vx vy\n\n");

    if (argc < 3) {
        fprintf(stderr, "Not enough arguments.\n");
        return 1;
    }

    double t_end = atof(argv[1]);
    const char* input_file = argv[2];
    double dt = (argc >= 4) ? atof(argv[3]) : DEFAULT_DT;
    double eps = (argc >= 5) ? atof(argv[4]) : DEFAULT_EPS;

    Particle* cpu_particles = NULL;
    int n = 0;
    if (!read_input_file(input_file, &cpu_particles, &n)) {
        return 1;
    }

    if (!validate_parameters(t_end, dt, n)) {
        free(cpu_particles);
        return 1;
    }

    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        fprintf(stderr, "ERROR: No CUDA devices found\n");
        free(cpu_particles);
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("CUDA Device: %s (Compute %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("Particles: %d, t_end=%.6g, dt=%.6g, eps=%.6g\n", n, t_end, dt, eps);
    printf("Method: Symplectic Euler (Euler-Cromer) with symmetric forces (Fij=-Fji)\n\n");

    Particle* gpu_particles;
    size_t particles_size = sizeof(Particle) * n;

    check_cuda_error(cudaMalloc(&gpu_particles, particles_size), "cudaMalloc");
    check_cuda_error(cudaMemcpy(gpu_particles, cpu_particles, particles_size,
                               cudaMemcpyHostToDevice), "cudaMemcpy HtoD");

    create_directory("results");
    create_directory("results/nbody_cuda");
    const char* outpath = "results/nbody_cuda/trajectories.csv";
    FILE* out = fopen(outpath, "w");
    if (!out) {
        fprintf(stderr, "ERROR: Cannot open %s\n", outpath);
        cudaFree(gpu_particles);
        free(cpu_particles);
        return 1;
    }

    fprintf(out, "t");
    for (int i = 0; i < n; ++i) fprintf(out, ",x%d,y%d", i+1, i+1);
    fprintf(out, "\n");

    fprintf(out, "0.0");
    for (int i = 0; i < n; ++i) {
        fprintf(out, ",%.10g,%.10g", cpu_particles[i].x, cpu_particles[i].y);
    }
    fprintf(out, "\n");

    int blockSize = BLOCK_SIZE;
    if (blockSize > prop.maxThreadsPerBlock) {
        blockSize = prop.maxThreadsPerBlock;
    }
    int gridSize = (n + blockSize - 1) / blockSize;
    printf("CUDA config: grid=%d, block=%d\n\n", gridSize, blockSize);

    int total_steps = (int)ceil(t_end / dt);
    double t = 0.0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float total_ms = 0.0f;

    for (int step = 1; step <= total_steps; ++step) {
        cudaEventRecord(start);

        reset_forces_kernel<<<gridSize, blockSize>>>(gpu_particles, n);
        cudaDeviceSynchronize();
        check_cuda_error(cudaGetLastError(), "reset_forces");

        compute_forces_kernel<<<gridSize, blockSize>>>(gpu_particles, n, eps);
        cudaDeviceSynchronize();
        check_cuda_error(cudaGetLastError(), "compute_forces");

        euler_step_kernel<<<gridSize, blockSize>>>(gpu_particles, n, dt);
        cudaDeviceSynchronize();
        check_cuda_error(cudaGetLastError(), "euler_step");

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float step_ms = 0;
        cudaEventElapsedTime(&step_ms, start, stop);
        total_ms += step_ms;

        t += dt;

        cudaMemcpy(cpu_particles, gpu_particles, particles_size, cudaMemcpyDeviceToHost);

        fprintf(out, "%.10g", t);
        for (int i = 0; i < n; ++i) {
            fprintf(out, ",%.10g,%.10g", cpu_particles[i].x, cpu_particles[i].y);
        }
        fprintf(out, "\n");

        if (step % ((total_steps > 10) ? (total_steps / 10) : 1) == 0) {
            printf("Progress: %d%% (t=%.6g, step: %.3f ms)\n",
                   (int)((100.0 * step) / total_steps), t, step_ms);
        }
    }

    printf("\nPerformance:\n");
    printf("Total time: %.3f ms\n", total_ms);
    printf("Avg step time: %.3f ms\n", total_ms / total_steps);
    printf("Steps/sec: %.1f\n", 1000.0 / (total_ms / total_steps));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(gpu_particles);
    fclose(out);
    free(cpu_particles);

    printf("\nSimulation finished. Output: %s\n", outpath);
    return 0;
}
