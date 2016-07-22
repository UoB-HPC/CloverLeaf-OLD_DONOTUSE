
#include "../definitions_c.h"

#if defined(USE_KOKKOS)
#include "kokkos/accelerate.c"

void accelerate_kokkos()
{
    for (int tile = 0; tile < tiles_per_chunk; tile++) {
        accelerate_functor g(
            chunk.tiles[tile],
            chunk.tiles[tile].t_xmin,
            chunk.tiles[tile].t_xmax + 1,
            chunk.tiles[tile].t_ymin,
            chunk.tiles[tile].t_ymax + 1,
            dt);
        g.compute();
    }
}

#else

#include "../kernels/accelerate_kernel_c.c"

void accelerate_openmp()
{
    for (int tile = 0; tile < tiles_per_chunk; tile++) {
        #pragma omp parallel
        {
            DOUBLEFOR(
                chunk.tiles[tile].t_ymin,
                chunk.tiles[tile].t_ymax + 1,
                chunk.tiles[tile].t_xmin,
            chunk.tiles[tile].t_xmax + 1, {
                accelerate_kernel_c_(
                    j, k,
                    chunk.tiles[tile].t_xmin, chunk.tiles[tile].t_xmax,
                    chunk.tiles[tile].t_ymin, chunk.tiles[tile].t_ymax,
                    chunk.tiles[tile].field.xarea,
                    chunk.tiles[tile].field.yarea,
                    chunk.tiles[tile].field.volume,
                    chunk.tiles[tile].field.density0,
                    chunk.tiles[tile].field.pressure,
                    chunk.tiles[tile].field.viscosity,
                    chunk.tiles[tile].field.xvel0,
                    chunk.tiles[tile].field.yvel0,
                    chunk.tiles[tile].field.xvel1,
                    chunk.tiles[tile].field.yvel1,
                    dt);
            });
        }
    }
}

#endif

void accelerate_adaptor()
{
#if defined(USE_KOKKOS)
    accelerate_kokkos();
#else
    accelerate_openmp();
#endif
}
