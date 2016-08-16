#include "../definitions_c.h"

#if defined(USE_OPENMP) || defined(USE_OMPSS) || defined(USE_KOKKOS)
#include "../kernels/field_summary_kernel_c.c"
void field_summary(
    double* vol,
    double* ie,
    double* ke,
    double* mass,
    double* press)
{
    *vol = 0.0;
    *mass = 0.0;
    *ie = 0.0;
    *ke = 0.0;
    *press = 0.0;

    for (int tilen = 0; tilen < tiles_per_chunk; tilen++) {
        struct tile_type tile = chunk.tiles[tilen];
        int x_min = tile.t_xmin,
            x_max = tile.t_xmax,
            y_min = tile.t_ymin,
            y_max = tile.t_ymax;
        field_2d_t volume   = tile.field.volume;
        field_2d_t density0 = tile.field.density0;
        field_2d_t energy0  = tile.field.energy0;
        field_2d_t pressure = tile.field.pressure;
        field_2d_t xvel0    = tile.field.xvel0;
        field_2d_t yvel0    = tile.field.yvel0;

        double _vol   = 0.0,
               _mass  = 0.0,
               _ie    = 0.0,
               _ke    = 0.0,
               _press = 0.0;


        #pragma omp parallel for reduction(+:_vol,_mass,_ie,_ke,_press)
        for (int k = y_min; k <= y_max; k++) {
            for (int j = x_min; j <= x_max; j++) {
                field_summary_kernel(
                    j, k,
                    x_min, x_max,
                    y_min, y_max,
                    volume,
                    density0, energy0,
                    pressure,
                    xvel0, yvel0,
                    &_vol, &_mass, &_ie, &_ke, &_press);
            }
        }

        *vol   += _vol;
        *mass  += _mass;
        *ie    += _ie;
        *ke    += _ke;
        *press += _press;
    }
}
#endif

#if defined(USE_CUDA)

#include "../kernels/field_summary_kernel_c.c"

__global__ void field_summary_kernel(
    int x_min, int x_max,
    int y_min, int y_max,
    double* volume,
    double* density0,
    double* energy0,
    double* pressure,
    double* xvel0,
    double* yvel0,
    double* g_vol,
    double* g_mass,
    double* g_ie,
    double* g_ke,
    double* g_press)
{
    extern __shared__ double sdata[];

    int j = threadIdx.x + blockIdx.x * blockDim.x + x_min;
    int k = threadIdx.y + blockIdx.y * blockDim.y + y_min;

    int x = threadIdx.x,
        y = threadIdx.y;
    int lid = x + blockDim.x * y;
    int gid = blockIdx.x + gridDim.x * blockIdx.y;
    int lsize = blockDim.x * blockDim.y;

    sdata[lid + lsize * 0] =
        sdata[lid + lsize * 1] =
            sdata[lid + lsize * 2] =
                sdata[lid + lsize * 3] =
                    sdata[lid + lsize * 4] =
                        0.0;

    if (j <= x_max && k <= y_max) {

        field_summary_kernel(
            j, k,
            x_min, x_max,
            y_min, y_max,
            volume,
            density0, energy0,
            pressure,
            xvel0, yvel0,
            &sdata[lid + lsize * 0],
            &sdata[lid + lsize * 1],
            &sdata[lid + lsize * 2],
            &sdata[lid + lsize * 3],
            &sdata[lid + lsize * 4]);
    }

    __syncthreads();
    for (int s = lsize / 2; s > 0; s /= 2) {
        if (lid < s) {
            sdata[lid + lsize * 0] += sdata[lid + lsize * 0 + s];
            sdata[lid + lsize * 1] += sdata[lid + lsize * 1 + s];
            sdata[lid + lsize * 2] += sdata[lid + lsize * 2 + s];
            sdata[lid + lsize * 3] += sdata[lid + lsize * 3 + s];
            sdata[lid + lsize * 4] += sdata[lid + lsize * 4 + s];
        }
        __syncthreads();
    }
    if (lid == 0) {
        g_vol[gid]   = sdata[0 + lsize * 0];
        g_mass[gid]  = sdata[0 + lsize * 1];
        g_ie[gid]    = sdata[0 + lsize * 2];
        g_ke[gid]    = sdata[0 + lsize * 3];
        g_press[gid] = sdata[0 + lsize * 4];
    }
}

void field_summary(
    double* vol,
    double* ie,
    double* ke,
    double* mass,
    double* press)
{
    *vol = 0.0;
    *mass = 0.0;
    *ie = 0.0;
    *ke = 0.0;
    *press = 0.0;

    for (int tilen = 0; tilen < tiles_per_chunk; tilen++) {
        struct tile_type tile = chunk.tiles[tilen];


        int x_min = tile.t_xmin,
            x_max = tile.t_xmax,
            y_min = tile.t_ymin,
            y_max = tile.t_ymax;

        dim3 size = numBlocks(
                        dim3((x_max) - (x_min) + 1,
                             (y_max) - (y_min) + 1),
                        field_summary_blocksize);
        field_summary_kernel <<< size,
                             field_summary_blocksize,
                             field_summary_blocksize.x* field_summary_blocksize.y* sizeof(double) * 5>>>(
                                 x_min, x_max,
                                 y_min, y_max,
                                 tile.field.d_volume,
                                 tile.field.d_density0,
                                 tile.field.d_energy0,
                                 tile.field.d_pressure,
                                 tile.field.d_xvel0,
                                 tile.field.d_yvel0,
                                 tile.field.d_work_array1,
                                 tile.field.d_work_array2,
                                 tile.field.d_work_array3,
                                 tile.field.d_work_array4,
                                 tile.field.d_work_array5);

        gpuErrchk(cudaMemcpyAsync(
                      tile.field.work_array1,
                      tile.field.d_work_array1,
                      size.x * size.y * sizeof(double),
                      cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpyAsync(
                      tile.field.work_array2,
                      tile.field.d_work_array2,
                      size.x * size.y * sizeof(double),
                      cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpyAsync(
                      tile.field.work_array3,
                      tile.field.d_work_array3,
                      size.x * size.y * sizeof(double),
                      cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpyAsync(
                      tile.field.work_array4,
                      tile.field.d_work_array4,
                      size.x * size.y * sizeof(double),
                      cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpyAsync(
                      tile.field.work_array5,
                      tile.field.d_work_array5,
                      size.x * size.y * sizeof(double),
                      cudaMemcpyDeviceToHost));
        cudaThreadSynchronize();

        for (int i = 0; i < size.x * size.y; i++) {
            *vol   += tile.field.work_array1[i];
            *mass  += tile.field.work_array2[i];
            *ie    += tile.field.work_array3[i];
            *ke    += tile.field.work_array4[i];
            *press += tile.field.work_array5[i];
        }
    }

    if (profiler_on)
        cudaDeviceSynchronize();
}
#endif

#if defined(USE_OPENCL)
#include "../definitions_c.h"
#include "../kernels/field_summary_kernel_c.c"

void field_summary(
    double* vol,
    double* ie,
    double* ke,
    double* mass,
    double* press)
{
    *vol = 0.0;
    *mass = 0.0;
    *ie = 0.0;
    *ke = 0.0;
    *press = 0.0;

    for (int tilen = 0; tilen < tiles_per_chunk; tilen++) {
        struct tile_type tile = chunk.tiles[tilen];
        int x_min = tile.t_xmin,
            x_max = tile.t_xmax,
            y_min = tile.t_ymin,
            y_max = tile.t_ymax;
        field_2d_t volume   = tile.field.volume;
        field_2d_t density1 = tile.field.density1;
        field_2d_t energy1  = tile.field.energy1;
        field_2d_t pressure = tile.field.pressure;
        field_2d_t xvel1    = tile.field.xvel1;
        field_2d_t yvel1    = tile.field.yvel1;

        mapoclmem(tile.field.d_volume, volume, tile.field.volume_size, CL_MAP_READ);
        mapoclmem(tile.field.d_density1, density1, tile.field.density1_size, CL_MAP_READ);
        mapoclmem(tile.field.d_energy1, energy1, tile.field.energy1_size, CL_MAP_READ);
        mapoclmem(tile.field.d_pressure, pressure, tile.field.pressure_size, CL_MAP_READ);
        mapoclmem(tile.field.d_xvel1, xvel1, tile.field.xvel1_size, CL_MAP_READ);
        mapoclmem(tile.field.d_yvel1, yvel1, tile.field.yvel1_size, CL_MAP_READ);

        double _vol   = 0.0,
               _mass  = 0.0,
               _ie    = 0.0,
               _ke    = 0.0,
               _press = 0.0;

        for (int k = y_min; k <= y_max; k++) {
            for (int j = x_min; j <= x_max; j++) {
                field_summary_kernel(
                    j, k,
                    x_min, x_max,
                    y_min, y_max,
                    volume,
                    density1, energy1,
                    pressure,
                    xvel1, yvel1,
                    &_vol, &_mass, &_ie, &_ke, &_press);
            }
        }

        unmapoclmem(tile.field.d_volume,
                    tile.field.volume);
        unmapoclmem(tile.field.d_density1,
                    tile.field.density1);
        unmapoclmem(tile.field.d_energy1,
                    tile.field.energy1);
        unmapoclmem(tile.field.d_pressure,
                    tile.field.pressure);
        unmapoclmem(tile.field.d_xvel1,
                    tile.field.xvel1);
        unmapoclmem(tile.field.d_yvel1,
                    tile.field.yvel1);

        *vol   += _vol;
        *mass  += _mass;
        *ie    += _ie;
        *ke    += _ke;
        *press += _press;
    }
    if (profiler_on)
        openclQueue.finish();
}
#endif
