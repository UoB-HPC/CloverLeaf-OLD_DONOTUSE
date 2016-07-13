#include "ideal_gas.h"
#include "definitions_c.h"
#include "kernels/ideal_gas_kernel_c.c"

void ideal_gas(int tile, bool predict)
{

    #pragma omp parallel
    {
        DOUBLEFOR(
            chunk.tiles[tile].t_ymin,
            chunk.tiles[tile].t_ymax,
            chunk.tiles[tile].t_xmin,
            chunk.tiles[tile].t_xmax,
        ({
            if (predict)
            {
                ideal_gas_kernel_c_(
                    j, k,
                    chunk.tiles[tile].t_xmin,
                    chunk.tiles[tile].t_xmax,
                    chunk.tiles[tile].t_ymin,
                    chunk.tiles[tile].t_ymax,
                    chunk.tiles[tile].field.density1,
                    chunk.tiles[tile].field.energy1,
                    chunk.tiles[tile].field.pressure,
                    chunk.tiles[tile].field.soundspeed
                );
            } else {
                ideal_gas_kernel_c_(
                    j, k,
                    chunk.tiles[tile].t_xmin,
                    chunk.tiles[tile].t_xmax,
                    chunk.tiles[tile].t_ymin,
                    chunk.tiles[tile].t_ymax,
                    chunk.tiles[tile].field.density0,
                    chunk.tiles[tile].field.energy0,
                    chunk.tiles[tile].field.pressure,
                    chunk.tiles[tile].field.soundspeed
                );

            }
        }));
    }
#ifdef USE_KOKKOS
    Kokkos::fence();
#endif
}
