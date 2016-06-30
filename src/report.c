#include <stdio.h>
#include <stdlib.h>
#include "definitions_c.h"

void report_error(char *location, char *error)
{
    BOSSPRINT(stdout, "Error from %s:\n%s\n", location, error);
    BOSSPRINT(stdout, "CLOVER is terminating.");
    //TODO
    // clover_abort();
    exit(1);
}
