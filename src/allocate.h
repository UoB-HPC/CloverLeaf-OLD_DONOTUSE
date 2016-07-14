#ifndef ALLOCATE_H
#define ALLOCATE_H

void allocate();

#ifdef USE_KOKKOS
#include "allocate_kokkos.c"
#else
#include "allocate.c"
#endif

#endif
