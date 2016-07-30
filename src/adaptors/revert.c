#if defined(USE_KOKKOS)

#include "kokkos/revert.cpp"

void revert(struct chunk_type chunk)
{
    for (int tile = 0; tile < tiles_per_chunk; tile++) {
        struct tile_type t = chunk.tiles[tile];
        revert_functor f(t,
                         t.t_xmin,
                         t.t_xmax,
                         t.t_ymin,
                         t.t_ymax);
        f.compute();
    }
}
#endif

#if defined(USE_OPENMP) || defined(USE_OMPSS)

#include "../kernels/revert_kernel_c.c"

void revert(struct chunk_type chunk)
{
    for (int tile = 0; tile < tiles_per_chunk; tile++) {

        #pragma omp parallel
        {
            DOUBLEFOR(
                chunk.tiles[tile].t_ymin,
                chunk.tiles[tile].t_ymax,
                chunk.tiles[tile].t_xmin,
            chunk.tiles[tile].t_xmax, {
                revert_kernel_c_(
                    j, k,
                    chunk.tiles[tile].t_xmin,
                    chunk.tiles[tile].t_xmax,
                    chunk.tiles[tile].t_ymin,
                    chunk.tiles[tile].t_ymax,
                    chunk.tiles[tile].field.density0,
                    chunk.tiles[tile].field.density1,
                    chunk.tiles[tile].field.energy0,
                    chunk.tiles[tile].field.energy1);
            });
        }
    }

}
#endif
#if defined(USE_OPENCL)

#include "../kernels/revert_kernel_c.c"

void revert(struct chunk_type chunk)
{
    for (int tile = 0; tile < tiles_per_chunk; tile++) {
        DOUBLEFOR(
            chunk.tiles[tile].t_ymin,
            chunk.tiles[tile].t_ymax,
            chunk.tiles[tile].t_xmin,
        chunk.tiles[tile].t_xmax, {
            revert_kernel_c_(
                j, k,
                chunk.tiles[tile].t_xmin,
                chunk.tiles[tile].t_xmax,
                chunk.tiles[tile].t_ymin,
                chunk.tiles[tile].t_ymax,
                chunk.tiles[tile].field.density0,
                chunk.tiles[tile].field.density1,
                chunk.tiles[tile].field.energy0,
                chunk.tiles[tile].field.energy1);
        });
    }

}
#endif