
#if defined(USE_KOKKOS)

#include "kokkos/pdv.cpp"

void pdv(struct chunk_type chunk, bool predict, double dt)
{
    if (predict) {
        for (int tile = 0; tile < tiles_per_chunk; tile++) {
            pdv_predict_functor f(
                chunk.tiles[tile],
                chunk.tiles[tile].t_xmin,
                chunk.tiles[tile].t_xmax,
                chunk.tiles[tile].t_ymin,
                chunk.tiles[tile].t_ymax,
                dt);
            f.compute();
        }
    } else {
        for (int tile = 0; tile < tiles_per_chunk; tile++) {
            pdv_no_predict_functor f(
                chunk.tiles[tile],
                chunk.tiles[tile].t_xmin,
                chunk.tiles[tile].t_xmax,
                chunk.tiles[tile].t_ymin,
                chunk.tiles[tile].t_ymax,
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


void pdv(struct chunk_type chunk, bool predict, double dt)
{
    cl::Kernel pdv_kernel(openclProgram, "pdv_kernel");
    for (int tile = 0; tile < tiles_per_chunk; tile++) {
        int xmin = chunk.tiles[tile].t_xmin,
            xmax = chunk.tiles[tile].t_xmax,
            ymin = chunk.tiles[tile].t_ymin,
            ymax = chunk.tiles[tile].t_ymax;
        pdv_kernel.setArg(0,  xmin);
        pdv_kernel.setArg(1,  xmax);
        pdv_kernel.setArg(2,  ymin);
        pdv_kernel.setArg(3,  ymax);
        pdv_kernel.setArg(4, dt);

        pdv_kernel.setArg(5, *chunk.tiles[tile].field.d_xarea);
        pdv_kernel.setArg(6, *chunk.tiles[tile].field.d_yarea);
        pdv_kernel.setArg(7, *chunk.tiles[tile].field.d_volume);
        pdv_kernel.setArg(8, *chunk.tiles[tile].field.d_density0);
        pdv_kernel.setArg(9, *chunk.tiles[tile].field.d_density1);
        pdv_kernel.setArg(10, *chunk.tiles[tile].field.d_energy0);
        pdv_kernel.setArg(11, *chunk.tiles[tile].field.d_energy1);
        pdv_kernel.setArg(12, *chunk.tiles[tile].field.d_pressure);
        pdv_kernel.setArg(13, *chunk.tiles[tile].field.d_viscosity);
        pdv_kernel.setArg(14, *chunk.tiles[tile].field.d_xvel0);
        pdv_kernel.setArg(15, *chunk.tiles[tile].field.d_xvel1);
        pdv_kernel.setArg(16, *chunk.tiles[tile].field.d_yvel0);
        pdv_kernel.setArg(17, *chunk.tiles[tile].field.d_yvel1);
        pdv_kernel.setArg(18, *chunk.tiles[tile].field.d_work_array1);
        if (predict) {
            pdv_kernel.setArg(19, 0);
        } else {
            pdv_kernel.setArg(19, 1);
        }
        openclQueue.enqueueNDRangeKernel(pdv_kernel, cl::NullRange, cl::NDRange(xmax - xmin, ymax - ymin), cl::NullRange);
    }

    openclQueue.finish();

}
#endif
