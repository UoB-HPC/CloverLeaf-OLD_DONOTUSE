
#if defined(USE_KOKKOS)

#include "kokkos/pdv.cpp"

void pdv(struct chunk_type chunk, bool predict, double dt)
{
    if (predict) {
        for (int tile = 0; tile < tiles_per_chunk; tile++) {
            pdv_predict_functor f(
                chunk.tiles[tile],
                chunk.tiles[tile].t_ymin,
                chunk.tiles[tile].t_ymax,
                chunk.tiles[tile].t_xmin,
                chunk.tiles[tile].t_xmax,
                dt);
            f.compute();
        }
    } else {
        for (int tile = 0; tile < tiles_per_chunk; tile++) {
            pdv_no_predict_functor f(
                chunk.tiles[tile],
                chunk.tiles[tile].t_ymin,
                chunk.tiles[tile].t_ymax,
                chunk.tiles[tile].t_xmin,
                chunk.tiles[tile].t_xmax,
                dt);
            f.compute();
        }
    }
}
#endif

#if defined(USE_OPENMP) || defined(USE_OMPSS)

#include "../kernels/PdV_kernel_c.c"

void pdv(struct chunk_type chunk, bool predict, double dt)
{
    #pragma omp parallel
    {
        if (predict)
        {
            for (int tile = 0; tile < tiles_per_chunk; tile++) {
                DOUBLEFOR(
                    chunk.tiles[tile].t_ymin,
                    chunk.tiles[tile].t_ymax,
                    chunk.tiles[tile].t_xmin,
                chunk.tiles[tile].t_xmax, {
                    pdv_kernel_predict_c_(
                        j, k,
                        chunk.tiles[tile].t_xmin,
                        chunk.tiles[tile].t_xmax,
                        chunk.tiles[tile].t_ymin,
                        chunk.tiles[tile].t_ymax,
                        dt,
                        chunk.tiles[tile].field.xarea,
                        chunk.tiles[tile].field.yarea,
                        chunk.tiles[tile].field.volume,
                        chunk.tiles[tile].field.density0,
                        chunk.tiles[tile].field.density1,
                        chunk.tiles[tile].field.energy0,
                        chunk.tiles[tile].field.energy1,
                        chunk.tiles[tile].field.pressure,
                        chunk.tiles[tile].field.viscosity,
                        chunk.tiles[tile].field.xvel0,
                        chunk.tiles[tile].field.xvel1,
                        chunk.tiles[tile].field.yvel0,
                        chunk.tiles[tile].field.yvel1,
                        chunk.tiles[tile].field.work_array1);
                });
            }
        } else {
            for (int tile = 0; tile < tiles_per_chunk; tile++)
            {
                DOUBLEFOR(
                    chunk.tiles[tile].t_ymin,
                    chunk.tiles[tile].t_ymax,
                    chunk.tiles[tile].t_xmin,
                chunk.tiles[tile].t_xmax, {
                    pdv_kernel_no_predict_c_(
                        j, k,
                        chunk.tiles[tile].t_xmin,
                        chunk.tiles[tile].t_xmax,
                        chunk.tiles[tile].t_ymin,
                        chunk.tiles[tile].t_ymax,
                        dt,
                        chunk.tiles[tile].field.xarea,
                        chunk.tiles[tile].field.yarea,
                        chunk.tiles[tile].field.volume,
                        chunk.tiles[tile].field.density0,
                        chunk.tiles[tile].field.density1,
                        chunk.tiles[tile].field.energy0,
                        chunk.tiles[tile].field.energy1,
                        chunk.tiles[tile].field.pressure,
                        chunk.tiles[tile].field.viscosity,
                        chunk.tiles[tile].field.xvel0,
                        chunk.tiles[tile].field.xvel1,
                        chunk.tiles[tile].field.yvel0,
                        chunk.tiles[tile].field.yvel1,
                        chunk.tiles[tile].field.work_array1);
                });
            }
        }
    }
}
#endif

#if defined(USE_OPENCL)

#include "../kernels/PdV_kernel_c.c"

void pdv(struct chunk_type chunk, bool predict, double dt)
{
    if (predict) {
        for (int tile = 0; tile < tiles_per_chunk; tile++) {
            DOUBLEFOR(
                chunk.tiles[tile].t_ymin,
                chunk.tiles[tile].t_ymax,
                chunk.tiles[tile].t_xmin,
            chunk.tiles[tile].t_xmax, {
                pdv_kernel_predict_c_(
                    j, k,
                    chunk.tiles[tile].t_xmin,
                    chunk.tiles[tile].t_xmax,
                    chunk.tiles[tile].t_ymin,
                    chunk.tiles[tile].t_ymax,
                    dt,
                    chunk.tiles[tile].field.xarea,
                    chunk.tiles[tile].field.yarea,
                    chunk.tiles[tile].field.volume,
                    chunk.tiles[tile].field.density0,
                    chunk.tiles[tile].field.density1,
                    chunk.tiles[tile].field.energy0,
                    chunk.tiles[tile].field.energy1,
                    chunk.tiles[tile].field.pressure,
                    chunk.tiles[tile].field.viscosity,
                    chunk.tiles[tile].field.xvel0,
                    chunk.tiles[tile].field.xvel1,
                    chunk.tiles[tile].field.yvel0,
                    chunk.tiles[tile].field.yvel1,
                    chunk.tiles[tile].field.work_array1);
            });
        }
    } else {
        for (int tile = 0; tile < tiles_per_chunk; tile++) {
            DOUBLEFOR(
                chunk.tiles[tile].t_ymin,
                chunk.tiles[tile].t_ymax,
                chunk.tiles[tile].t_xmin,
            chunk.tiles[tile].t_xmax, {
                pdv_kernel_no_predict_c_(
                    j, k,
                    chunk.tiles[tile].t_xmin,
                    chunk.tiles[tile].t_xmax,
                    chunk.tiles[tile].t_ymin,
                    chunk.tiles[tile].t_ymax,
                    dt,
                    chunk.tiles[tile].field.xarea,
                    chunk.tiles[tile].field.yarea,
                    chunk.tiles[tile].field.volume,
                    chunk.tiles[tile].field.density0,
                    chunk.tiles[tile].field.density1,
                    chunk.tiles[tile].field.energy0,
                    chunk.tiles[tile].field.energy1,
                    chunk.tiles[tile].field.pressure,
                    chunk.tiles[tile].field.viscosity,
                    chunk.tiles[tile].field.xvel0,
                    chunk.tiles[tile].field.xvel1,
                    chunk.tiles[tile].field.yvel0,
                    chunk.tiles[tile].field.yvel1,
                    chunk.tiles[tile].field.work_array1);
            });
        }
    }
}
#endif
