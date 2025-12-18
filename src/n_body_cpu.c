#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#ifdef _WIN32
  #include <direct.h>
#else
  #include <sys/stat.h>
#endif

#ifdef _OPENMP
  #include <omp.h>
#else
  static inline int omp_get_max_threads(void) { return 1; }
#endif

#ifndef G
#define G 6.67430e-11
#endif

#define DEFAULT_DT 0.01
#define DEFAULT_TEND 1.0
#define DEFAULT_EPS 1e-5

typedef struct {
    double mass;
    double x, y;
    double vx, vy;
    double fx, fy;
} Particle;

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

void reset_forces(Particle* p, int n) {
    int i;
#ifdef _OPENMP
#pragma omp parallel for private(i)
#endif
    for (i = 0; i < n; ++i) {
        p[i].fx = 0.0;
        p[i].fy = 0.0;
    }
}

void compute_forces(Particle* p, int n, double eps) {
    double eps2 = eps * eps;
    int i, j;

#ifdef _OPENMP
#pragma omp parallel private(i, j)
#endif
    {

#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
        for (i = 0; i < n; ++i) {
            for (j = i + 1; j < n; ++j) {
                double dx = p[j].x - p[i].x;
                double dy = p[j].y - p[i].y;
                double r2 = dx*dx + dy*dy + eps2;

                if (r2 < 1e-10) continue;

                double r = sqrt(r2);
                double inv_r3 = 1.0 / (r2 * r);
                double coef = G * p[i].mass * p[j].mass * inv_r3;

                double fx_ij = coef * dx;
                double fy_ij = coef * dy;

#ifdef _OPENMP
#pragma omp atomic
#endif
                p[i].fx += fx_ij;

#ifdef _OPENMP
#pragma omp atomic
#endif
                p[i].fy += fy_ij;

#ifdef _OPENMP
#pragma omp atomic
#endif
                p[j].fx -= fx_ij;

#ifdef _OPENMP
#pragma omp atomic
#endif
                p[j].fy -= fy_ij;
            }
        }
    }
}

void euler_step(Particle* p, int n, double dt) {
    int i;
#ifdef _OPENMP
#pragma omp parallel for private(i)
#endif
    for (i = 0; i < n; ++i) {
        double ax = p[i].fx / p[i].mass;
        double ay = p[i].fy / p[i].mass;

        p[i].vx += ax * dt;
        p[i].vy += ay * dt;

        p[i].x += p[i].vx * dt;
        p[i].y += p[i].vy * dt;
    }
}

int validate_parameters(double t_end, double dt, int n) {
    if (t_end <= 0) { fprintf(stderr, "ERROR: t_end must be positive\n"); return 0; }
    if (dt <= 0) { fprintf(stderr, "ERROR: dt must be positive\n"); return 0; }
    if (dt > t_end) { fprintf(stderr, "ERROR: dt > t_end\n"); return 0; }
    if (n <= 0) { fprintf(stderr, "ERROR: n must be positive\n"); return 0; }
    return 1;
}

int main(int argc, char* argv[]) {
    printf("N-Body (2D) - CPU with OpenMP - Fixed Version\n");
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

    Particle* particles = NULL;
    int n = 0;
    if (!read_input_file(input_file, &particles, &n)) {
        return 1;
    }

    if (!validate_parameters(t_end, dt, n)) {
        free(particles);
        return 1;
    }

    int max_threads = omp_get_max_threads();
    printf("Particles: %d, t_end=%.6g, dt=%.6g, eps=%.6g\n", n, t_end, dt, eps);
    printf("OpenMP threads: %d\n", max_threads);
    printf("Method: Symplectic Euler (Euler-Cromer) with symmetric forces (Fij=-Fji)\n\n");

    create_directory("results");
    create_directory("results/nbody_cpu");
    const char* outpath = "results/nbody_cpu/trajectories.csv";
    FILE* out = fopen(outpath, "w");
    if (!out) {
        fprintf(stderr, "ERROR: Cannot open %s\n", outpath);
        free(particles);
        return 1;
    }

    fprintf(out, "t");
    for (int i = 0; i < n; ++i) fprintf(out, ",x%d,y%d", i+1, i+1);
    fprintf(out, "\n");

    fprintf(out, "0.0");
    for (int i = 0; i < n; ++i) {
        fprintf(out, ",%.10g,%.10g", particles[i].x, particles[i].y);
    }
    fprintf(out, "\n");

    int total_steps = (int)ceil(t_end / dt);
    double t = 0.0;

    for (int step = 1; step <= total_steps; ++step) {
        reset_forces(particles, n);

        compute_forces(particles, n, eps);

        euler_step(particles, n, dt);

        t += dt;

        fprintf(out, "%.10g", t);
        for (int i = 0; i < n; ++i) {
            fprintf(out, ",%.10g,%.10g", particles[i].x, particles[i].y);
        }
        fprintf(out, "\n");

        if (step % ((total_steps > 10) ? (total_steps / 10) : 1) == 0) {
            printf("Progress: %d%% (t=%.6g)\n",
                   (int)((100.0 * step) / total_steps), t);
        }
    }

    fclose(out);
    free(particles);

    printf("\nSimulation finished. Output: %s\n", outpath);
    return 0;
}
