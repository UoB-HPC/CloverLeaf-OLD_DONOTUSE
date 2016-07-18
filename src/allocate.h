#ifndef ALLOCATE_H
#define ALLOCATE_H

void allocate();

#ifdef USE_KOKKOS
#include "allocate_kokkos.cpp"
#else
#include "allocate.cpp"
#endif

#endif
