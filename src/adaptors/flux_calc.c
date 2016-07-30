
#if defined(USE_KOKKOS)
#include "kokkos/flux_calc.cpp"
void flux_calc(struct chunk_type chunk, double dt)
{
    for (int tile = 0; tile < tiles_per_chunk; tile++) {
        flux_calc_x_functor f1(
            chunk.tiles[tile],
            chunk.tiles[tile].t_xmin,
            chunk.tiles[tile].t_xmax + 1,
            chunk.tiles[tile].t_ymin,
            chunk.tiles[tile].t_ymax,
            dt);
        f1.compute();

        flux_calc_y_functor f2(
            chunk.tiles[tile],
            chunk.tiles[tile].t_xmin,
            chunk.tiles[tile].t_xmax,
            chunk.tiles[tile].t_ymin,
            chunk.tiles[tile].t_ymax + 1,
            dt);
        f2.compute();
        fence();
    }

}
#endif


#if defined(USE_OPENMP) || defined(USE_OMPSS)

#include "../kernels/flux_calc_kernel_c.c"

void flux_calc(struct chunk_type chunk, double dt)
{
    for (int tile = 0; tile < tiles_per_chunk; tile++) {
        #pragma omp parallel
        {

            DOUBLEFOR(
                chunk.tiles[tile].t_ymin,
                chunk.tiles[tile].t_ymax,
                chunk.tiles[tile].t_xmin,
            chunk.tiles[tile].t_xmax + 1, {
                flux_calc_x_kernel(
                    j, k,
                    chunk.tiles[tile].t_xmin,
                    chunk.tiles[tile].t_xmax,
                    chunk.tiles[tile].t_ymin,
                    chunk.tiles[tile].t_ymax,
                    dt,
                    chunk.tiles[tile].field.xarea,
                    chunk.tiles[tile].field.xvel0,
                    chunk.tiles[tile].field.xvel1,
                    chunk.tiles[tile].field.vol_flux_x);
            });

            DOUBLEFOR(
                chunk.tiles[tile].t_ymin,
                chunk.tiles[tile].t_ymax + 1,
                chunk.tiles[tile].t_xmin,
            chunk.tiles[tile].t_xmax, {
                flux_calc_y_kernel(
                    j, k,
                    chunk.tiles[tile].t_xmin,
                    chunk.tiles[tile].t_xmax,
                    chunk.tiles[tile].t_ymin,
                    chunk.tiles[tile].t_ymax,
                    dt,
                    chunk.tiles[tile].field.yarea,
                    chunk.tiles[tile].field.yvel0,
                    chunk.tiles[tile].field.yvel1,
                    chunk.tiles[tile].field.vol_flux_y);
            });
        }
    }

}
#endif

#if defined(USE_OPENCL)

#include "../kernels/flux_calc_kernel_c.c"

void flux_calc(struct chunk_type chunk, double dt)
{
    for (int tile = 0; tile < tiles_per_chunk; tile++) {
        DOUBLEFOR(
            chunk.tiles[tile].t_ymin,
            chunk.tiles[tile].t_ymax,
            chunk.tiles[tile].t_xmin,
        chunk.tiles[tile].t_xmax + 1, {
            flux_calc_x_kernel(
                j, k,
                chunk.tiles[tile].t_xmin,
                chunk.tiles[tile].t_xmax,
                chunk.tiles[tile].t_ymin,
                chunk.tiles[tile].t_ymax,
                dt,
                chunk.tiles[tile].field.xarea,
                chunk.tiles[tile].field.xvel0,
                chunk.tiles[tile].field.xvel1,
                chunk.tiles[tile].field.vol_flux_x);
        });

        DOUBLEFOR(
            chunk.tiles[tile].t_ymin,
            chunk.tiles[tile].t_ymax + 1,
            chunk.tiles[tile].t_xmin,
        chunk.tiles[tile].t_xmax, {
            flux_calc_y_kernel(
                j, k,
                chunk.tiles[tile].t_xmin,
                chunk.tiles[tile].t_xmax,
                chunk.tiles[tile].t_ymin,
                chunk.tiles[tile].t_ymax,
                dt,
                chunk.tiles[tile].field.yarea,
                chunk.tiles[tile].field.yvel0,
                chunk.tiles[tile].field.yvel1,
                chunk.tiles[tile].field.vol_flux_y);
        });
    }
}
#endif
