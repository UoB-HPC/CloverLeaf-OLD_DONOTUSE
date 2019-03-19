#include "viscosity.h"
// #include "kernels/viscosity_kernel_c.cc"
#include "adaptors/viscosity.cpp"
#include "definitions_c.h"

void viscosity()
{
    viscosity(chunk);
}
