#include "ideal_gas.h"
#include "definitions_c.h"
#include "adaptors/ideal_gas.cpp"


void ideal_gas(int tile, bool predict)
{
    ideal_gas_adaptor(tile, predict);
}


