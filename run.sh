#!/usr/bin/env bash
set -euo pipefail

BUILD_TYPE=${BUILD_TYPE:-Debug}
BUILD_DIR=${BUILD_DIR:-cmake-build-${BUILD_TYPE,,}}

# Mandelbrot
MANDEL_NTHREADS_LIST=${MANDEL_NTHREADS_LIST:-"1 2 4 8"}
MANDEL_NPOINTS=${MANDEL_NPOINTS:-1000}

# N-body
NBODY_TEND=${NBODY_TEND:-10.0}
NBODY_INPUT=${NBODY_INPUT:-data/example_nbody.txt}

# Сборка
echo "==> Building project (type=${BUILD_TYPE})"

cmake -S . -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
cmake --build "${BUILD_DIR}" -- -j"$(nproc 2>/dev/null || echo 4)"

mkdir -p results/mandelbrot results/nbody

# Mandelbrot
MANDEL_EXE="${BUILD_DIR}/mandelbrot"

if [[ -x "${MANDEL_EXE}" ]]; then
  echo "==> Running Mandelbrot experiments"

  for nt in ${MANDEL_NTHREADS_LIST}; do
    OUT_FILE="results/mandelbrot/mandelbrot_${nt}threads_${MANDEL_NPOINTS}points.csv"
    echo "   - nthreads=${nt}, npoints=${MANDEL_NPOINTS} -> ${OUT_FILE}"
    "${MANDEL_EXE}" "${nt}" "${MANDEL_NPOINTS}" > "${OUT_FILE}"
  done
else
  echo "WARNING: ${MANDEL_EXE} not found or not executable, skipping Mandelbrot"
fi

# N-body CPU
NBODY_CPU_EXE="${BUILD_DIR}/nbody_cpu"

if [[ -x "${NBODY_CPU_EXE}" ]]; then
  echo "==> Running N-body CPU (OpenMP)"

  OUT_FILE="results/nbody/nbody_cpu_tend${NBODY_TEND}.csv"
  echo "   - tend=${NBODY_TEND}, input=${NBODY_INPUT} -> ${OUT_FILE}"
  "${NBODY_CPU_EXE}" "${NBODY_TEND}" "${NBODY_INPUT}" > "${OUT_FILE}"
else
  echo "WARNING: ${NBODY_CPU_EXE} not found or not executable, skipping N-body CPU"
fi

# N-body CUDA
NBODY_CUDA_EXE="${BUILD_DIR}/nbody_cuda"

if [[ -x "${NBODY_CUDA_EXE}" ]]; then
  echo "==> Running N-body CUDA"

  OUT_FILE="results/nbody/nbody_cuda_tend${NBODY_TEND}.csv"
  echo "   - tend=${NBODY_TEND}, input=${NBODY_INPUT} -> ${OUT_FILE}"
  "${NBODY_CUDA_EXE}" "${NBODY_TEND}" "${NBODY_INPUT}" > "${OUT_FILE}"
else
  echo "NOTE: ${NBODY_CUDA_EXE} not found, CUDA part not built yet"
fi

# Python
if [[ -f scripts/analyze.py ]]; then
  echo "==> Running Python analysis (scripts/analyze.py)"
  python3 scripts/analyze.py
else
  echo "NOTE: scripts/analyze.py not found, skipping analysis"
fi

echo "==> Done."