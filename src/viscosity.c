#include "viscosity.h"
// #include "kernels/viscosity_kernel_c.c"
#include "adaptors/viscosity.c"
#include "definitions_c.h"

void viscosity()
{
    viscosity(chunk);
}
