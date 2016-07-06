#include <stdio.h>
#include "definitions_c.h"
#include "initialise.h"
#include "hydro.h"
#include <mpi.h>
#include "clover.h"
#include <Kokkos_Core.hpp>


int main(int argc, char **argv)
{
    Kokkos::initialize(argc, argv);
    clover_init_comms(argc, argv);
    BOSSPRINT(stdout, "\nClover Version %8.3f\n", g_version);
    BOSSPRINT(stdout, "Task Count %d\n", parallel.max_task);
    initialise();
    hydro();

    Kokkos::finalize();
    MPI_Finalize();
}
