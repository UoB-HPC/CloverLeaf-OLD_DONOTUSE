#include <stdio.h>
#include "definitions_c.h"
#include "initialise.h"
#include "hydro.h"
#include <mpi.h>
#include "clover.h"
#include <omp.h>

#ifdef USE_KOKKOS
#include <Kokkos_Core.hpp>
#endif


int main(int argc, char **argv)
{
#ifdef USE_KOKKOS
    Kokkos::initialize(argc, argv);
#endif

    clover_init_comms(argc, argv);

    BOSSPRINT(stdout, "Clover Version %8.3f\n", g_version);
    BOSSPRINT(stdout, "Task Count %d\n", parallel.max_task);
#ifndef USE_KOKKOS
    #pragma omp parallel
    #pragma omp master
    {BOSSPRINT(stdout, "OpenMP threads %d\n", omp_get_num_threads());}
#endif
    initialise();
    hydro();

#ifdef USE_KOKKOS
    Kokkos::finalize();
#endif
    MPI_Finalize();
}
