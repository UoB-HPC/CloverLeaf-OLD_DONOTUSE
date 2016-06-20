#include <stdio.h>

void report_error(char *location, char *error)
{
    fprintf(stdout, "Error from %s:\n%s\n", location, error);
    fprintf(stdout, "CLOVER is terminating.");
    //TODO
    // clover_abort();
}
