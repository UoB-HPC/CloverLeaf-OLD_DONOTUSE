#ifndef ALLOCATE_H
#define ALLOCATE_H

void allocate();

#ifdef USE_KOKKOS
#include "allocate_kokkos.c"
#endif

#if defined(USE_OPENMP) || defined(USE_OMPSS)
#include "allocate.c"
#endif

#if defined(USE_OPENCL)
#include "allocate_opencl.c"
#endif

#endif
