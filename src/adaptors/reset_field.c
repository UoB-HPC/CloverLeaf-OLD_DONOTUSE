
#if defined(USE_KOKKOS)
#include "kokkos/reset.cpp"

void reset_field(struct chunk_type chunk)
{
    for (int tile = 0; tile < tiles_per_chunk; tile++) {
        reset_field_functor f1(
            chunk.tiles[tile],
            chunk.tiles[tile].t_xmin,
            chunk.tiles[tile].t_xmax + 1,
            chunk.tiles[tile].t_ymin,
            chunk.tiles[tile].t_ymax + 1);
        f1.compute();
    }
}

#endif


#if defined(USE_OPENMP) || defined(USE_OMPSS)
#include "../kernels/reset_field_kernel_c.c"

void reset_field(struct chunk_type chunk)
{
    for (int tile = 0; tile < tiles_per_chunk; tile++) {
        #pragma omp parallel
        {
            DOUBLEFOR(chunk.tiles[tile].t_ymin,
            chunk.tiles[tile].t_ymax + 1,
            chunk.tiles[tile].t_xmin,
            chunk.tiles[tile].t_xmax + 1, {
                reset_field_kernel_c_(
                    j, k,
                    chunk.tiles[tile].t_xmin,
                    chunk.tiles[tile].t_xmax,
                    chunk.tiles[tile].t_ymin,
                    chunk.tiles[tile].t_ymax,
                    chunk.tiles[tile].field.density0,
                    chunk.tiles[tile].field.density1,
                    chunk.tiles[tile].field.energy0,
                    chunk.tiles[tile].field.energy1,
                    chunk.tiles[tile].field.xvel0,
                    chunk.tiles[tile].field.xvel1,
                    chunk.tiles[tile].field.yvel0,
                    chunk.tiles[tile].field.yvel1);
            });
        }
    }
}
#endif

#if defined(USE_OPENCL)
#include "../kernels/reset_field_kernel_c.c"

void reset_field(struct chunk_type chunk)
{
    for (int tile = 0; tile < tiles_per_chunk; tile++) {
        DOUBLEFOR(chunk.tiles[tile].t_ymin,
                  chunk.tiles[tile].t_ymax + 1,
                  chunk.tiles[tile].t_xmin,
        chunk.tiles[tile].t_xmax + 1, {
            reset_field_kernel_c_(
                j, k,
                chunk.tiles[tile].t_xmin,
                chunk.tiles[tile].t_xmax,
                chunk.tiles[tile].t_ymin,
                chunk.tiles[tile].t_ymax,
                chunk.tiles[tile].field.density0,
                chunk.tiles[tile].field.density1,
                chunk.tiles[tile].field.energy0,
                chunk.tiles[tile].field.energy1,
                chunk.tiles[tile].field.xvel0,
                chunk.tiles[tile].field.xvel1,
                chunk.tiles[tile].field.yvel0,
                chunk.tiles[tile].field.yvel1);
        });
    }
}
#endif