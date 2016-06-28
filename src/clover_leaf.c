#include <stdio.h>
#include "definitions_c.h"
#include "initialise.h"
#include "hydro.h"


int main()
{
    fprintf(stdout, "\nClover Version %8.3f\n", g_version);

    initialise();
    hydro();
}
