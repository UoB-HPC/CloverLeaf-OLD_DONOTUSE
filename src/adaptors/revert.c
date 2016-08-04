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
        int x_min = chunk.tiles[tile].t_xmin,
            x_max = chunk.tiles[tile].t_xmax,
            y_min = chunk.tiles[tile].t_ymin,
            y_max = chunk.tiles[tile].t_ymax;

        cl::Kernel revert(openclProgram, "revert_kernel");
        revert.setArg(0,  x_min);
        revert.setArg(1,  x_max);
        revert.setArg(2,  y_min);
        revert.setArg(3,  y_max);
        revert.setArg(4, *chunk.tiles[tile].field.d_density0);
        revert.setArg(5, *chunk.tiles[tile].field.d_density1);
        revert.setArg(6, *chunk.tiles[tile].field.d_energy0);
        revert.setArg(7, *chunk.tiles[tile].field.d_energy1);
        openclQueue.enqueueNDRangeKernel(revert, cl::NullRange, cl::NDRange((x_max) - (x_min) + 1, (y_max) - (y_min) + 1), cl::NullRange);
    }
    openclQueue.finish();

}
#endif