
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
        int x_min = chunk.tiles[tile].t_xmin,
            x_max = chunk.tiles[tile].t_xmax,
            y_min = chunk.tiles[tile].t_ymin,
            y_max = chunk.tiles[tile].t_ymax;

        cl::Kernel reset_field(openclProgram, "reset_field_kernel");
        reset_field.setArg(0,  x_min);
        reset_field.setArg(1,  x_max);
        reset_field.setArg(2,  y_min);
        reset_field.setArg(3,  y_max);

        reset_field.setArg(4, *chunk.tiles[tile].field.d_density0);
        reset_field.setArg(5, *chunk.tiles[tile].field.d_density1);
        reset_field.setArg(6, *chunk.tiles[tile].field.d_energy0);
        reset_field.setArg(7, *chunk.tiles[tile].field.d_energy1);
        reset_field.setArg(8, *chunk.tiles[tile].field.d_xvel0);
        reset_field.setArg(9, *chunk.tiles[tile].field.d_xvel1);
        reset_field.setArg(10, *chunk.tiles[tile].field.d_yvel0);
        reset_field.setArg(11, *chunk.tiles[tile].field.d_yvel1);
        openclQueue.enqueueNDRangeKernel(reset_field, cl::NullRange, cl::NDRange((x_max + 1) - (x_min) + 1, (y_max + 1) - (y_min) + 1), cl::NullRange);
    }
    openclQueue.finish();
}
#endif
