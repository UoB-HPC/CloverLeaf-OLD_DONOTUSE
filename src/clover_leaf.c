#include <stdio.h>
#include "definitions_c.h"
#include "initialise.h"
#include "hydro.h"
#include <mpi.h>
#include "clover.h"
#include <omp.h>
#include <iostream>

#ifdef USE_KOKKOS
#include <Kokkos_Core.hpp>
#endif

#ifdef USE_OPENCL
// #include "openclinit.cpp"
#endif

#ifdef USE_CUDA
#include "cudaInit.cpp"
#endif


int main(int argc, char** argv)
{

#ifdef USE_OMPSS
    printf("Using OMPSS\n");
#endif
#ifdef USE_OPENMP
    printf("Using OpenMP\n");
#endif
#ifdef USE_OPENCL
    printf("Using OpenCL\n");
#endif

#ifdef USE_KOKKOS
    printf("Using Kokkos\n");

    Kokkos::initialize(argc, argv);

    std::ostringstream msg;
    msg << "{" << std::endl ;

    if (Kokkos::hwloc::available()) {
        msg << "hwloc( NUMA[" << Kokkos::hwloc::get_available_numa_count()
            << "] x CORE["    << Kokkos::hwloc::get_available_cores_per_numa()
            << "] x HT["      << Kokkos::hwloc::get_available_threads_per_core()
            << "] )"
            << std::endl ;
    }
#if defined( KOKKOS_HAVE_CUDA )
    Kokkos::Cuda::print_configuration(msg);
#endif
    std::cerr << msg.str() << std::endl;
#endif


#ifdef USE_CUDA
    printf("Using CUDA\n");
    initCuda();
#endif

    clover_init_comms(argc, argv);

    BOSSPRINT(stdout, "Clover Version %8.3f\n", g_version);
    BOSSPRINT(stdout, "Task Count %d\n", parallel.max_task);
#if defined(USE_OMPSS) || defined(USE_OPENMP)
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
