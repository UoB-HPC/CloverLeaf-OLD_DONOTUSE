#include "definitions_c.h"
#include <stdlib.h>
#include <cuda_runtime.h>

int size2d(int xmin, int xmax, int ymin, int ymax)
{
    return (xmax - xmin + 1) * (ymax - ymin + 1);
}

int size1d(int min, int max)
{
    return max - min + 1;
}

void* aligned_malloc(size_t size, int align)
{
    void* mem = malloc(size + (align - 1) + sizeof(void*));

    char* amem = ((char*)mem) + sizeof(void*);
    amem += align - ((uintptr_t)amem & (align - 1));

    ((void**)amem)[-1] = mem;
    return amem;
}

void allocate()
{

    for (int tile = 0; tile < tiles_per_chunk; tile++) {
        int xmin = chunk.tiles[tile].t_xmin,
            xmax = chunk.tiles[tile].t_xmax,
            ymin = chunk.tiles[tile].t_ymin,
            ymax = chunk.tiles[tile].t_ymax;

        chunk.tiles[tile].field.density0_size        = size2d(xmin - 2, xmax + 2, ymin - 2, ymax + 2);
        chunk.tiles[tile].field.density1_size        = size2d(xmin - 2, xmax + 2, ymin - 2, ymax + 2);
        chunk.tiles[tile].field.energy0_size         = size2d(xmin - 2, xmax + 2, ymin - 2, ymax + 2);
        chunk.tiles[tile].field.energy1_size         = size2d(xmin - 2, xmax + 2, ymin - 2, ymax + 2);
        chunk.tiles[tile].field.pressure_size        = size2d(xmin - 2, xmax + 2, ymin - 2, ymax + 2);
        chunk.tiles[tile].field.viscosity_size       = size2d(xmin - 2, xmax + 2, ymin - 2, ymax + 2);
        chunk.tiles[tile].field.soundspeed_size      = size2d(xmin - 2, xmax + 2, ymin - 2, ymax + 2);

        chunk.tiles[tile].field.xvel0_size           = size2d(xmin - 2, xmax + 3, ymin - 2, ymax + 3);
        chunk.tiles[tile].field.xvel1_size           = size2d(xmin - 2, xmax + 3, ymin - 2, ymax + 3);
        chunk.tiles[tile].field.yvel0_size           = size2d(xmin - 2, xmax + 3, ymin - 2, ymax + 3);
        chunk.tiles[tile].field.yvel1_size           = size2d(xmin - 2, xmax + 3, ymin - 2, ymax + 3);

        chunk.tiles[tile].field.vol_flux_x_size      = size2d(xmin - 2, xmax + 3, ymin - 2, ymax + 2);
        chunk.tiles[tile].field.mass_flux_x_size     = size2d(xmin - 2, xmax + 3, ymin - 2, ymax + 2);
        chunk.tiles[tile].field.vol_flux_y_size      = size2d(xmin - 2, xmax + 2, ymin - 2, ymax + 3);
        chunk.tiles[tile].field.mass_flux_y_size     = size2d(xmin - 2, xmax + 2, ymin - 2, ymax + 3);

        chunk.tiles[tile].field.work_array1_size     = size2d(xmin - 2, xmax + 3, ymin - 2, ymax + 3);
        chunk.tiles[tile].field.work_array2_size     = size2d(xmin - 2, xmax + 3, ymin - 2, ymax + 3);
        chunk.tiles[tile].field.work_array3_size     = size2d(xmin - 2, xmax + 3, ymin - 2, ymax + 3);
        chunk.tiles[tile].field.work_array4_size     = size2d(xmin - 2, xmax + 3, ymin - 2, ymax + 3);
        chunk.tiles[tile].field.work_array5_size     = size2d(xmin - 2, xmax + 3, ymin - 2, ymax + 3);
        chunk.tiles[tile].field.work_array6_size     = size2d(xmin - 2, xmax + 3, ymin - 2, ymax + 3);
        chunk.tiles[tile].field.work_array7_size     = size2d(xmin - 2, xmax + 3, ymin - 2, ymax + 3);

        chunk.tiles[tile].field.cellx_size           = size1d(xmin - 2, xmax + 2);
        chunk.tiles[tile].field.celly_size           = size1d(ymin - 2, ymax + 2);
        chunk.tiles[tile].field.vertexx_size         = size1d(xmin - 2, xmax + 3);
        chunk.tiles[tile].field.vertexy_size         = size1d(ymin - 2, ymax + 3);
        chunk.tiles[tile].field.celldx_size          = size1d(xmin - 2, xmax + 2);
        chunk.tiles[tile].field.celldy_size          = size1d(ymin - 2, ymax + 2);
        chunk.tiles[tile].field.vertexdx_size        = size1d(xmin - 2, xmax + 3);
        chunk.tiles[tile].field.vertexdy_size        = size1d(ymin - 2, ymax + 3);

        chunk.tiles[tile].field.volume_size          = size2d(xmin - 2, xmax + 2, ymin - 2, ymax + 2);
        chunk.tiles[tile].field.xarea_size           = size2d(xmin - 2, xmax + 3, ymin - 2, ymax + 2);
        chunk.tiles[tile].field.yarea_size           = size2d(xmin - 2, xmax + 2, ymin - 2, ymax + 3);


        chunk.tiles[tile].field.density0   = (double*)aligned_malloc(sizeof(double) * chunk.tiles[tile].field.density0_size, 4096);
        chunk.tiles[tile].field.density1   = (double*)aligned_malloc(sizeof(double) * chunk.tiles[tile].field.density1_size, 4096);
        chunk.tiles[tile].field.energy0    = (double*)aligned_malloc(sizeof(double) * chunk.tiles[tile].field.energy0_size, 4096);
        chunk.tiles[tile].field.energy1    = (double*)aligned_malloc(sizeof(double) * chunk.tiles[tile].field.energy1_size, 4096);
        chunk.tiles[tile].field.pressure   = (double*)aligned_malloc(sizeof(double) * chunk.tiles[tile].field.pressure_size, 4096);
        chunk.tiles[tile].field.viscosity  = (double*)aligned_malloc(sizeof(double) * chunk.tiles[tile].field.viscosity_size, 4096);
        chunk.tiles[tile].field.soundspeed = (double*)aligned_malloc(sizeof(double) * chunk.tiles[tile].field.soundspeed_size, 4096);

        chunk.tiles[tile].field.xvel0 = (double*)aligned_malloc(sizeof(double) * chunk.tiles[tile].field.xvel0_size, 4096);
        chunk.tiles[tile].field.xvel1 = (double*)aligned_malloc(sizeof(double) * chunk.tiles[tile].field.xvel1_size, 4096);
        chunk.tiles[tile].field.yvel0 = (double*)aligned_malloc(sizeof(double) * chunk.tiles[tile].field.yvel0_size, 4096);
        chunk.tiles[tile].field.yvel1 = (double*)aligned_malloc(sizeof(double) * chunk.tiles[tile].field.yvel1_size, 4096);

        chunk.tiles[tile].field.vol_flux_x  = (double*)aligned_malloc(sizeof(double) * chunk.tiles[tile].field.vol_flux_x_size, 4096);
        chunk.tiles[tile].field.mass_flux_x = (double*)aligned_malloc(sizeof(double) * chunk.tiles[tile].field.mass_flux_x_size, 4096);
        chunk.tiles[tile].field.vol_flux_y  = (double*)aligned_malloc(sizeof(double) * chunk.tiles[tile].field.vol_flux_y_size, 4096);
        chunk.tiles[tile].field.mass_flux_y = (double*)aligned_malloc(sizeof(double) * chunk.tiles[tile].field.mass_flux_y_size, 4096);

        chunk.tiles[tile].field.work_array1 = (double*)aligned_malloc(sizeof(double) * chunk.tiles[tile].field.work_array1_size, 4096);
        chunk.tiles[tile].field.work_array2 = (double*)aligned_malloc(sizeof(double) * chunk.tiles[tile].field.work_array2_size, 4096);
        chunk.tiles[tile].field.work_array3 = (double*)aligned_malloc(sizeof(double) * chunk.tiles[tile].field.work_array3_size, 4096);
        chunk.tiles[tile].field.work_array4 = (double*)aligned_malloc(sizeof(double) * chunk.tiles[tile].field.work_array4_size, 4096);
        chunk.tiles[tile].field.work_array5 = (double*)aligned_malloc(sizeof(double) * chunk.tiles[tile].field.work_array5_size, 4096);
        chunk.tiles[tile].field.work_array6 = (double*)aligned_malloc(sizeof(double) * chunk.tiles[tile].field.work_array6_size, 4096);
        chunk.tiles[tile].field.work_array7 = (double*)aligned_malloc(sizeof(double) * chunk.tiles[tile].field.work_array7_size, 4096);

        chunk.tiles[tile].field.cellx    = (double*)aligned_malloc(sizeof(double) * chunk.tiles[tile].field.cellx_size, 4096);
        chunk.tiles[tile].field.celly    = (double*)aligned_malloc(sizeof(double) * chunk.tiles[tile].field.celly_size, 4096);
        chunk.tiles[tile].field.vertexx  = (double*)aligned_malloc(sizeof(double) * chunk.tiles[tile].field.vertexx_size, 4096);
        chunk.tiles[tile].field.vertexy  = (double*)aligned_malloc(sizeof(double) * chunk.tiles[tile].field.vertexy_size, 4096);
        chunk.tiles[tile].field.celldx   = (double*)aligned_malloc(sizeof(double) * chunk.tiles[tile].field.celldx_size, 4096);
        chunk.tiles[tile].field.celldy   = (double*)aligned_malloc(sizeof(double) * chunk.tiles[tile].field.celldy_size, 4096);
        chunk.tiles[tile].field.vertexdx = (double*)aligned_malloc(sizeof(double) * chunk.tiles[tile].field.vertexdx_size, 4096);
        chunk.tiles[tile].field.vertexdy = (double*)aligned_malloc(sizeof(double) * chunk.tiles[tile].field.vertexdy_size, 4096);
        chunk.tiles[tile].field.volume   = (double*)aligned_malloc(sizeof(double) * chunk.tiles[tile].field.volume_size, 4096);
        chunk.tiles[tile].field.xarea    = (double*)aligned_malloc(sizeof(double) * chunk.tiles[tile].field.xarea_size, 4096);
        chunk.tiles[tile].field.yarea    = (double*)aligned_malloc(sizeof(double) * chunk.tiles[tile].field.yarea_size, 4096);



        gpuErrchk(cudaMalloc(&chunk.tiles[tile].field.d_density0,   chunk.tiles[tile].field.density0_size * sizeof(double)));
        gpuErrchk(cudaMalloc(&chunk.tiles[tile].field.d_density1,   chunk.tiles[tile].field.density1_size * sizeof(double)));
        gpuErrchk(cudaMalloc(&chunk.tiles[tile].field.d_energy0,    chunk.tiles[tile].field.energy0_size * sizeof(double)));
        gpuErrchk(cudaMalloc(&chunk.tiles[tile].field.d_energy1,    chunk.tiles[tile].field.energy1_size * sizeof(double)));
        gpuErrchk(cudaMalloc(&chunk.tiles[tile].field.d_pressure,   chunk.tiles[tile].field.pressure_size * sizeof(double)));
        gpuErrchk(cudaMalloc(&chunk.tiles[tile].field.d_viscosity,  chunk.tiles[tile].field.viscosity_size * sizeof(double)));
        gpuErrchk(cudaMalloc(&chunk.tiles[tile].field.d_soundspeed, chunk.tiles[tile].field.soundspeed_size * sizeof(double)));

        gpuErrchk(cudaMalloc(&chunk.tiles[tile].field.d_xvel0 , chunk.tiles[tile].field.xvel0_size * sizeof(double)));
        gpuErrchk(cudaMalloc(&chunk.tiles[tile].field.d_xvel1 , chunk.tiles[tile].field.xvel1_size * sizeof(double)));
        gpuErrchk(cudaMalloc(&chunk.tiles[tile].field.d_yvel0 , chunk.tiles[tile].field.yvel0_size * sizeof(double)));
        gpuErrchk(cudaMalloc(&chunk.tiles[tile].field.d_yvel1 , chunk.tiles[tile].field.yvel1_size * sizeof(double)));

        gpuErrchk(cudaMalloc(&chunk.tiles[tile].field.d_vol_flux_x  , chunk.tiles[tile].field.vol_flux_x_size * sizeof(double)));
        gpuErrchk(cudaMalloc(&chunk.tiles[tile].field.d_mass_flux_x , chunk.tiles[tile].field.mass_flux_x_size * sizeof(double)));
        gpuErrchk(cudaMalloc(&chunk.tiles[tile].field.d_vol_flux_y  , chunk.tiles[tile].field.vol_flux_y_size * sizeof(double)));
        gpuErrchk(cudaMalloc(&chunk.tiles[tile].field.d_mass_flux_y , chunk.tiles[tile].field.mass_flux_y_size * sizeof(double)));

        gpuErrchk(cudaMalloc(&chunk.tiles[tile].field.d_work_array1 , chunk.tiles[tile].field.work_array1_size * sizeof(double)));
        gpuErrchk(cudaMalloc(&chunk.tiles[tile].field.d_work_array2 , chunk.tiles[tile].field.work_array2_size * sizeof(double)));
        gpuErrchk(cudaMalloc(&chunk.tiles[tile].field.d_work_array3 , chunk.tiles[tile].field.work_array3_size * sizeof(double)));
        gpuErrchk(cudaMalloc(&chunk.tiles[tile].field.d_work_array4 , chunk.tiles[tile].field.work_array4_size * sizeof(double)));
        gpuErrchk(cudaMalloc(&chunk.tiles[tile].field.d_work_array5 , chunk.tiles[tile].field.work_array5_size * sizeof(double)));
        gpuErrchk(cudaMalloc(&chunk.tiles[tile].field.d_work_array6 , chunk.tiles[tile].field.work_array6_size * sizeof(double)));
        gpuErrchk(cudaMalloc(&chunk.tiles[tile].field.d_work_array7 , chunk.tiles[tile].field.work_array7_size * sizeof(double)));

        gpuErrchk(cudaMalloc(&chunk.tiles[tile].field.d_cellx    , chunk.tiles[tile].field.cellx_size * sizeof(double)));
        gpuErrchk(cudaMalloc(&chunk.tiles[tile].field.d_celly    , chunk.tiles[tile].field.celly_size * sizeof(double)));
        gpuErrchk(cudaMalloc(&chunk.tiles[tile].field.d_vertexx  , chunk.tiles[tile].field.vertexx_size * sizeof(double)));
        gpuErrchk(cudaMalloc(&chunk.tiles[tile].field.d_vertexy  , chunk.tiles[tile].field.vertexy_size * sizeof(double)));
        gpuErrchk(cudaMalloc(&chunk.tiles[tile].field.d_celldx   , chunk.tiles[tile].field.celldx_size * sizeof(double)));
        gpuErrchk(cudaMalloc(&chunk.tiles[tile].field.d_celldy   , chunk.tiles[tile].field.celldy_size * sizeof(double)));
        gpuErrchk(cudaMalloc(&chunk.tiles[tile].field.d_vertexdx , chunk.tiles[tile].field.vertexdx_size * sizeof(double)));
        gpuErrchk(cudaMalloc(&chunk.tiles[tile].field.d_vertexdy , chunk.tiles[tile].field.vertexdy_size * sizeof(double)));
        gpuErrchk(cudaMalloc(&chunk.tiles[tile].field.d_volume   , chunk.tiles[tile].field.volume_size * sizeof(double)));
        gpuErrchk(cudaMalloc(&chunk.tiles[tile].field.d_xarea    , chunk.tiles[tile].field.xarea_size * sizeof(double)));
        gpuErrchk(cudaMalloc(&chunk.tiles[tile].field.d_yarea    , chunk.tiles[tile].field.yarea_size * sizeof(double)));


        // TODO first touch
    }
}
