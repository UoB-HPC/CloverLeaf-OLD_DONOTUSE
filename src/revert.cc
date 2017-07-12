#include "revert.h"
#include "definitions_c.h"
// #include "kernels/revert_kernel_c.c"
#include "adaptors/revert.cpp"


void revert()
{
    revert(chunk);
}
