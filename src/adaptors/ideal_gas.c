
#if defined(USE_KOKKOS)
#include "kokkos/ideal_gas.cpp"

void ideal_gas_adaptor(int tile, bool predict)
{
    if (predict) {
        ideal_gas_functor g(
            chunk.tiles[tile],
            chunk.tiles[tile].t_xmin,
            chunk.tiles[tile].t_xmax,
            chunk.tiles[tile].t_ymin,
            chunk.tiles[tile].t_ymax,
            chunk.tiles[tile].field.density1,
            chunk.tiles[tile].field.energy1);
        g.compute();
    } else {
        ideal_gas_functor g(
            chunk.tiles[tile],
            chunk.tiles[tile].t_xmin,
            chunk.tiles[tile].t_xmax,
            chunk.tiles[tile].t_ymin,
            chunk.tiles[tile].t_ymax,
            chunk.tiles[tile].field.density0,
            chunk.tiles[tile].field.energy0);
        g.compute();
    }
}
#endif

#if defined(USE_OPENMP) || defined(USE_OMPSS)
#include "../kernels/ideal_gas_kernel_c.c"

void ideal_gas_adaptor(int tile, bool predict)
{
    #pragma omp parallel
    {
        DOUBLEFOR(
            chunk.tiles[tile].t_ymin,
            chunk.tiles[tile].t_ymax,
            chunk.tiles[tile].t_xmin,
        chunk.tiles[tile].t_xmax, {
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
        });
    }
}
#endif

#if defined(USE_OPENCL)

#include "../kernels/ideal_gas_kernel_c.c"
void ideal_gas_adaptor(int tile, bool predict)
{
    DOUBLEFOR(
        chunk.tiles[tile].t_ymin,
        chunk.tiles[tile].t_ymax,
        chunk.tiles[tile].t_xmin,
    chunk.tiles[tile].t_xmax, {
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
    });
}
#endif