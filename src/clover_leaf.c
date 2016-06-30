#include <stdio.h>
#include "definitions_c.h"
#include "initialise.h"
#include "hydro.h"
#include <mpi.h>
#include "clover.h"



int main(int argc, char **argv)
{
    clover_init_comms(argc, argv);
    BOSSPRINT(stdout, "\nClover Version %8.3f\n", g_version);
    initialise();
    hydro();

    MPI_Finalize();
}
