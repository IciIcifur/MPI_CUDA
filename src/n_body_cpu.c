#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
  #include <direct.h>
  #include <io.h>
  #define access _access
  #define F_OK 0
#else
  #include <sys/stat.h>
  #include <unistd.h>
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

static double get_time_seconds(void) {
#ifdef USE_OMP_TIMER
    return omp_get_wtime();
#else
    return (double)clock() / (double)CLOCKS_PER_SEC;
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
        const double ax = p[i].fx / p[i].mass;
        const double ay = p[i].fy / p[i].mass;

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

static void get_output_paths(const char* argv_out,
                             const char** outpath,
                             const char** metricspath) {
    static char metrics_buf[4096];

    if (argv_out && argv_out[0] != '\0') {
        *outpath = argv_out;

        const char* last_sep = strrchr(argv_out, '\\');
        #ifdef _WIN32
        const char sep = '\\';
        #else
        const char sep = '/';
        #endif

        if (!last_sep) last_sep = strrchr(argv_out, '/');

        if (last_sep) {
            size_t dir_len = (size_t)(last_sep - argv_out);
            if (dir_len >= sizeof(metrics_buf)) dir_len = sizeof(metrics_buf) - 1;
            memcpy(metrics_buf, argv_out, dir_len);
            metrics_buf[dir_len] = '\0';
        } else {
            strcpy(metrics_buf, ".");
        }

        const size_t len = strlen(metrics_buf);
        if (len + 1 + strlen("metrics_cpu.txt") + 1 < sizeof(metrics_buf)) {
            metrics_buf[len] = sep;
            metrics_buf[len + 1] = '\0';
            strcat(metrics_buf, "metrics_cpu.txt");
        } else {
            strcpy(metrics_buf, "metrics_cpu.txt");
        }

        *metricspath = metrics_buf;
    } else {
        create_directory("results");
        create_directory("results/nbody_cpu");
        *outpath = "results/nbody_cpu/trajectories.csv";
        *metricspath = "results/nbody_cpu/metrics.txt";
    }
}

int main(const int argc, char* argv[]) {
    printf("N-Body - CPU OpenMP\n");
    printf("Usage: %s <t_end> <input_file> [dt] [out_csv] [eps]\n", argv[0]);

    if (argc < 3) {
        fprintf(stderr, "Not enough arguments.\n");
        return 1;
    }

    const double t_end = atof(argv[1]);
    const char* input_file = argv[2];
    const double dt = (argc >= 4) ? atof(argv[3]) : DEFAULT_DT;
    const char* out_csv_arg = (argc >= 5) ? argv[4] : NULL;
    const double eps = (argc >= 6) ? atof(argv[5]) : DEFAULT_EPS;

    Particle* particles = NULL;
    int n = 0;
    if (!read_input_file(input_file, &particles, &n)) {
        return 1;
    }

    if (!validate_parameters(t_end, dt, n)) {
        free(particles);
        return 1;
    }

    const int max_threads = omp_get_max_threads();
    printf("Particles: %d, t_end=%.6g, dt=%.6g, eps=%.6g\n", n, t_end, dt, eps);
    printf("OpenMP threads: %d\n", max_threads);

    const char* outpath = NULL;
    const char* metricspath = NULL;
    get_output_paths(out_csv_arg, &outpath, &metricspath);

    if (out_csv_arg) {
        char tmp[4096];
        strncpy(tmp, outpath, sizeof(tmp) - 1);
        tmp[sizeof(tmp) - 1] = '\0';
        char* last_sep = strrchr(tmp, '\\');
        if (!last_sep) last_sep = strrchr(tmp, '/');
        if (last_sep) {
            *last_sep = '\0';
            create_directory(tmp);
        }
    }

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

    const int total_steps = (int)ceil(t_end / dt);
    double t = 0.0;

    double total_time = 0.0;
    double min_step_time = 1e100;

    const double sim_start = get_time_seconds();

    for (int step = 1; step <= total_steps; ++step) {
        const double step_start = get_time_seconds();

        reset_forces(particles, n);
        compute_forces(particles, n, eps);
        euler_step(particles, n, dt);

        const double step_end = get_time_seconds();
        const double step_time = step_end - step_start;

        total_time += step_time;
        if (step_time < min_step_time) min_step_time = step_time;

        t += dt;

        fprintf(out, "%.10g", t);
        for (int i = 0; i < n; ++i) {
            fprintf(out, ",%.10g,%.10g", particles[i].x, particles[i].y);
        }
        fprintf(out, "\n");
    }

    const double sim_end = get_time_seconds();
    const double wall_time = sim_end - sim_start;

    fclose(out);

    FILE* mf = fopen(metricspath, "w");
    if (!mf) {
        fprintf(stderr, "ERROR: Cannot open %s for writing metrics\n", metricspath);
        free(particles);
        return 1;
    }

    fprintf(mf, "N-body CPU simulation metrics\n");
    fprintf(mf, "Input file: %s\n", input_file);
    fprintf(mf, "Particles: %d\n", n);
    fprintf(mf, "t_end: %.10g\n", t_end);
    fprintf(mf, "dt: %.10g\n", dt);
    fprintf(mf, "eps: %.10g\n", eps);
    fprintf(mf, "OpenMP threads: %d\n", max_threads);
    fprintf(mf, "Total steps: %d\n", total_steps);
    fprintf(mf, "Total compute time (sum of per-step): %.6f s\n", total_time);
    fprintf(mf, "Min step time: %.6f s\n", min_step_time);
    fprintf(mf, "Avg step time: %.6f s\n", total_time / total_steps);
    fprintf(mf, "Wall-clock time (whole simulation): %.6f s\n", wall_time);
    fprintf(mf, "Steps per second (avg): %.3f\n", total_steps / total_time);
    fprintf(mf, "Output trajectories: %s\n", outpath);
    fclose(mf);

    free(particles);

    printf("Simulation finished. Output: %s\n", outpath);
    printf("Metrics written to: %s\n", metricspath);
    return 0;
}