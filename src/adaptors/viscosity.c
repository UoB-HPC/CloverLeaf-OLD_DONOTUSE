
#if defined(USE_KOKKOS)
#include "kokkos/viscosity.cpp"
void viscosity(struct chunk_type chunk)
{
    for (int tile = 0; tile < tiles_per_chunk; tile++) {
        viscosity_functor f(
            chunk.tiles[tile],
            chunk.tiles[tile].t_xmin,
            chunk.tiles[tile].t_xmax,
            chunk.tiles[tile].t_ymin,
            chunk.tiles[tile].t_ymax);
        f.compute();
    }
}
#endif

#if defined(USE_OPENMP) || defined(USE_OMPSS)
#include "../kernels/viscosity_kernel_c.c"
void viscosity(struct chunk_type chunk)
{
    for (int tile = 0; tile < tiles_per_chunk; tile++) {

        #pragma omp parallel
        {
            DOUBLEFOR(chunk.tiles[tile].t_ymin,
            chunk.tiles[tile].t_ymax,
            chunk.tiles[tile].t_xmin,
            chunk.tiles[tile].t_xmax, {
                viscosity_kernel_c_(
                    j, k,
                    chunk.tiles[tile].t_xmin,
                    chunk.tiles[tile].t_xmax,
                    chunk.tiles[tile].t_ymin,
                    chunk.tiles[tile].t_ymax,
                    chunk.tiles[tile].field.celldx,
                    chunk.tiles[tile].field.celldy,
                    chunk.tiles[tile].field.density0,
                    chunk.tiles[tile].field.pressure,
                    chunk.tiles[tile].field.viscosity,
                    chunk.tiles[tile].field.xvel0,
                    chunk.tiles[tile].field.yvel0);
            });
        }
    }
}
#endif

#if defined(USE_OPENCL)
#include "../kernels/viscosity_kernel_c.c"
void viscosity(struct chunk_type chunk)
{
    for (int tile = 0; tile < tiles_per_chunk; tile++) {
        DOUBLEFOR(chunk.tiles[tile].t_ymin,
                  chunk.tiles[tile].t_ymax,
                  chunk.tiles[tile].t_xmin,
        chunk.tiles[tile].t_xmax, {
            viscosity_kernel_c_(
                j, k,
                chunk.tiles[tile].t_xmin,
                chunk.tiles[tile].t_xmax,
                chunk.tiles[tile].t_ymin,
                chunk.tiles[tile].t_ymax,
                chunk.tiles[tile].field.celldx,
                chunk.tiles[tile].field.celldy,
                chunk.tiles[tile].field.density0,
                chunk.tiles[tile].field.pressure,
                chunk.tiles[tile].field.viscosity,
                chunk.tiles[tile].field.xvel0,
                chunk.tiles[tile].field.yvel0);
        });
    }
}
#endif