#ifndef ALLOCATE_H
#define ALLOCATE_H

void allocate();

#ifdef USE_KOKKOS
#include "allocate_kokkos.cc"
#endif

#if defined(USE_OPENMP) || defined(USE_OMPSS)
#include "allocate.cc"
#endif

#if defined(USE_OPENCL)
#include "allocate_opencl.cc"
#endif

#if defined(USE_CUDA)
#include "allocate_cuda.cc"
#endif

#endif
