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


        gpuErrchk(cudaMallocHost(&chunk.tiles[tile].field.density0   , sizeof(double) * chunk.tiles[tile].field.density0_size));
        gpuErrchk(cudaMallocHost(&chunk.tiles[tile].field.density1   , sizeof(double) * chunk.tiles[tile].field.density1_size));
        gpuErrchk(cudaMallocHost(&chunk.tiles[tile].field.energy0    , sizeof(double) * chunk.tiles[tile].field.energy0_size));
        gpuErrchk(cudaMallocHost(&chunk.tiles[tile].field.energy1    , sizeof(double) * chunk.tiles[tile].field.energy1_size));
        gpuErrchk(cudaMallocHost(&chunk.tiles[tile].field.pressure   , sizeof(double) * chunk.tiles[tile].field.pressure_size));
        gpuErrchk(cudaMallocHost(&chunk.tiles[tile].field.viscosity  , sizeof(double) * chunk.tiles[tile].field.viscosity_size));
        gpuErrchk(cudaMallocHost(&chunk.tiles[tile].field.soundspeed , sizeof(double) * chunk.tiles[tile].field.soundspeed_size));

        gpuErrchk(cudaMallocHost(&chunk.tiles[tile].field.xvel0 , sizeof(double) * chunk.tiles[tile].field.xvel0_size));
        gpuErrchk(cudaMallocHost(&chunk.tiles[tile].field.xvel1 , sizeof(double) * chunk.tiles[tile].field.xvel1_size));
        gpuErrchk(cudaMallocHost(&chunk.tiles[tile].field.yvel0 , sizeof(double) * chunk.tiles[tile].field.yvel0_size));
        gpuErrchk(cudaMallocHost(&chunk.tiles[tile].field.yvel1 , sizeof(double) * chunk.tiles[tile].field.yvel1_size));

        gpuErrchk(cudaMallocHost(&chunk.tiles[tile].field.vol_flux_x  , sizeof(double) * chunk.tiles[tile].field.vol_flux_x_size));
        gpuErrchk(cudaMallocHost(&chunk.tiles[tile].field.mass_flux_x , sizeof(double) * chunk.tiles[tile].field.mass_flux_x_size));
        gpuErrchk(cudaMallocHost(&chunk.tiles[tile].field.vol_flux_y  , sizeof(double) * chunk.tiles[tile].field.vol_flux_y_size));
        gpuErrchk(cudaMallocHost(&chunk.tiles[tile].field.mass_flux_y , sizeof(double) * chunk.tiles[tile].field.mass_flux_y_size));

        gpuErrchk(cudaMallocHost(&chunk.tiles[tile].field.work_array1 , sizeof(double) * chunk.tiles[tile].field.work_array1_size));
        gpuErrchk(cudaMallocHost(&chunk.tiles[tile].field.work_array2 , sizeof(double) * chunk.tiles[tile].field.work_array2_size));
        gpuErrchk(cudaMallocHost(&chunk.tiles[tile].field.work_array3 , sizeof(double) * chunk.tiles[tile].field.work_array3_size));
        gpuErrchk(cudaMallocHost(&chunk.tiles[tile].field.work_array4 , sizeof(double) * chunk.tiles[tile].field.work_array4_size));
        gpuErrchk(cudaMallocHost(&chunk.tiles[tile].field.work_array5 , sizeof(double) * chunk.tiles[tile].field.work_array5_size));
        gpuErrchk(cudaMallocHost(&chunk.tiles[tile].field.work_array6 , sizeof(double) * chunk.tiles[tile].field.work_array6_size));
        gpuErrchk(cudaMallocHost(&chunk.tiles[tile].field.work_array7 , sizeof(double) * chunk.tiles[tile].field.work_array7_size));

        gpuErrchk(cudaMallocHost(&chunk.tiles[tile].field.cellx    , sizeof(double) * chunk.tiles[tile].field.cellx_size));
        gpuErrchk(cudaMallocHost(&chunk.tiles[tile].field.celly    , sizeof(double) * chunk.tiles[tile].field.celly_size));
        gpuErrchk(cudaMallocHost(&chunk.tiles[tile].field.vertexx  , sizeof(double) * chunk.tiles[tile].field.vertexx_size));
        gpuErrchk(cudaMallocHost(&chunk.tiles[tile].field.vertexy  , sizeof(double) * chunk.tiles[tile].field.vertexy_size));
        gpuErrchk(cudaMallocHost(&chunk.tiles[tile].field.celldx   , sizeof(double) * chunk.tiles[tile].field.celldx_size));
        gpuErrchk(cudaMallocHost(&chunk.tiles[tile].field.celldy   , sizeof(double) * chunk.tiles[tile].field.celldy_size));
        gpuErrchk(cudaMallocHost(&chunk.tiles[tile].field.vertexdx , sizeof(double) * chunk.tiles[tile].field.vertexdx_size));
        gpuErrchk(cudaMallocHost(&chunk.tiles[tile].field.vertexdy , sizeof(double) * chunk.tiles[tile].field.vertexdy_size));
        gpuErrchk(cudaMallocHost(&chunk.tiles[tile].field.volume   , sizeof(double) * chunk.tiles[tile].field.volume_size));
        gpuErrchk(cudaMallocHost(&chunk.tiles[tile].field.xarea    , sizeof(double) * chunk.tiles[tile].field.xarea_size));
        gpuErrchk(cudaMallocHost(&chunk.tiles[tile].field.yarea    , sizeof(double) * chunk.tiles[tile].field.yarea_size));



        gpuErrchk(cudaMalloc(&chunk.tiles[tile].field.d_density0,   chunk.tiles[tile].field.density0_size * sizeof(double)));
        gpuErrchk(cudaMemset(chunk.tiles[tile].field.d_density0, 0,   chunk.tiles[tile].field.density0_size * sizeof(double)));
        gpuErrchk(cudaMalloc(&chunk.tiles[tile].field.d_density1,   chunk.tiles[tile].field.density1_size * sizeof(double)));
        gpuErrchk(cudaMemset(chunk.tiles[tile].field.d_density1, 0,   chunk.tiles[tile].field.density1_size * sizeof(double)));
        gpuErrchk(cudaMalloc(&chunk.tiles[tile].field.d_energy0,    chunk.tiles[tile].field.energy0_size * sizeof(double)));
        gpuErrchk(cudaMemset(chunk.tiles[tile].field.d_energy0, 0,    chunk.tiles[tile].field.energy0_size * sizeof(double)));
        gpuErrchk(cudaMalloc(&chunk.tiles[tile].field.d_energy1,    chunk.tiles[tile].field.energy1_size * sizeof(double)));
        gpuErrchk(cudaMemset(chunk.tiles[tile].field.d_energy1, 0,    chunk.tiles[tile].field.energy1_size * sizeof(double)));
        gpuErrchk(cudaMalloc(&chunk.tiles[tile].field.d_pressure,   chunk.tiles[tile].field.pressure_size * sizeof(double)));
        gpuErrchk(cudaMemset(chunk.tiles[tile].field.d_pressure, 0,   chunk.tiles[tile].field.pressure_size * sizeof(double)));
        gpuErrchk(cudaMalloc(&chunk.tiles[tile].field.d_viscosity,  chunk.tiles[tile].field.viscosity_size * sizeof(double)));
        gpuErrchk(cudaMemset(chunk.tiles[tile].field.d_viscosity, 0,  chunk.tiles[tile].field.viscosity_size * sizeof(double)));
        gpuErrchk(cudaMalloc(&chunk.tiles[tile].field.d_soundspeed, chunk.tiles[tile].field.soundspeed_size * sizeof(double)));
        gpuErrchk(cudaMemset(chunk.tiles[tile].field.d_soundspeed, 0, chunk.tiles[tile].field.soundspeed_size * sizeof(double)));

        gpuErrchk(cudaMalloc(&chunk.tiles[tile].field.d_xvel0 , chunk.tiles[tile].field.xvel0_size * sizeof(double)));
        gpuErrchk(cudaMemset(chunk.tiles[tile].field.d_xvel0, 0 , chunk.tiles[tile].field.xvel0_size * sizeof(double)));
        gpuErrchk(cudaMalloc(&chunk.tiles[tile].field.d_xvel1 , chunk.tiles[tile].field.xvel1_size * sizeof(double)));
        gpuErrchk(cudaMemset(chunk.tiles[tile].field.d_xvel1, 0 , chunk.tiles[tile].field.xvel1_size * sizeof(double)));
        gpuErrchk(cudaMalloc(&chunk.tiles[tile].field.d_yvel0 , chunk.tiles[tile].field.yvel0_size * sizeof(double)));
        gpuErrchk(cudaMemset(chunk.tiles[tile].field.d_yvel0, 0 , chunk.tiles[tile].field.yvel0_size * sizeof(double)));
        gpuErrchk(cudaMalloc(&chunk.tiles[tile].field.d_yvel1 , chunk.tiles[tile].field.yvel1_size * sizeof(double)));
        gpuErrchk(cudaMemset(chunk.tiles[tile].field.d_yvel1, 0 , chunk.tiles[tile].field.yvel1_size * sizeof(double)));

        gpuErrchk(cudaMalloc(&chunk.tiles[tile].field.d_vol_flux_x  , chunk.tiles[tile].field.vol_flux_x_size * sizeof(double)));
        gpuErrchk(cudaMemset(chunk.tiles[tile].field.d_vol_flux_x, 0  , chunk.tiles[tile].field.vol_flux_x_size * sizeof(double)));
        gpuErrchk(cudaMalloc(&chunk.tiles[tile].field.d_mass_flux_x , chunk.tiles[tile].field.mass_flux_x_size * sizeof(double)));
        gpuErrchk(cudaMemset(chunk.tiles[tile].field.d_mass_flux_x, 0 , chunk.tiles[tile].field.mass_flux_x_size * sizeof(double)));
        gpuErrchk(cudaMalloc(&chunk.tiles[tile].field.d_vol_flux_y  , chunk.tiles[tile].field.vol_flux_y_size * sizeof(double)));
        gpuErrchk(cudaMemset(chunk.tiles[tile].field.d_vol_flux_y, 0  , chunk.tiles[tile].field.vol_flux_y_size * sizeof(double)));
        gpuErrchk(cudaMalloc(&chunk.tiles[tile].field.d_mass_flux_y , chunk.tiles[tile].field.mass_flux_y_size * sizeof(double)));
        gpuErrchk(cudaMemset(chunk.tiles[tile].field.d_mass_flux_y, 0 , chunk.tiles[tile].field.mass_flux_y_size * sizeof(double)));

        gpuErrchk(cudaMalloc(&chunk.tiles[tile].field.d_work_array1 , chunk.tiles[tile].field.work_array1_size * sizeof(double)));
        gpuErrchk(cudaMemset(chunk.tiles[tile].field.d_work_array1, 0 , chunk.tiles[tile].field.work_array1_size * sizeof(double)));
        gpuErrchk(cudaMalloc(&chunk.tiles[tile].field.d_work_array2 , chunk.tiles[tile].field.work_array2_size * sizeof(double)));
        gpuErrchk(cudaMemset(chunk.tiles[tile].field.d_work_array2, 0 , chunk.tiles[tile].field.work_array2_size * sizeof(double)));
        gpuErrchk(cudaMalloc(&chunk.tiles[tile].field.d_work_array3 , chunk.tiles[tile].field.work_array3_size * sizeof(double)));
        gpuErrchk(cudaMemset(chunk.tiles[tile].field.d_work_array3, 0 , chunk.tiles[tile].field.work_array3_size * sizeof(double)));
        gpuErrchk(cudaMalloc(&chunk.tiles[tile].field.d_work_array4 , chunk.tiles[tile].field.work_array4_size * sizeof(double)));
        gpuErrchk(cudaMemset(chunk.tiles[tile].field.d_work_array4, 0 , chunk.tiles[tile].field.work_array4_size * sizeof(double)));
        gpuErrchk(cudaMalloc(&chunk.tiles[tile].field.d_work_array5 , chunk.tiles[tile].field.work_array5_size * sizeof(double)));
        gpuErrchk(cudaMemset(chunk.tiles[tile].field.d_work_array5, 0 , chunk.tiles[tile].field.work_array5_size * sizeof(double)));
        gpuErrchk(cudaMalloc(&chunk.tiles[tile].field.d_work_array6 , chunk.tiles[tile].field.work_array6_size * sizeof(double)));
        gpuErrchk(cudaMemset(chunk.tiles[tile].field.d_work_array6, 0 , chunk.tiles[tile].field.work_array6_size * sizeof(double)));
        gpuErrchk(cudaMalloc(&chunk.tiles[tile].field.d_work_array7 , chunk.tiles[tile].field.work_array7_size * sizeof(double)));
        gpuErrchk(cudaMemset(chunk.tiles[tile].field.d_work_array7, 0 , chunk.tiles[tile].field.work_array7_size * sizeof(double)));

        gpuErrchk(cudaMalloc(&chunk.tiles[tile].field.d_cellx    , chunk.tiles[tile].field.cellx_size * sizeof(double)));
        gpuErrchk(cudaMemset(chunk.tiles[tile].field.d_cellx, 0    , chunk.tiles[tile].field.cellx_size * sizeof(double)));
        gpuErrchk(cudaMalloc(&chunk.tiles[tile].field.d_celly    , chunk.tiles[tile].field.celly_size * sizeof(double)));
        gpuErrchk(cudaMemset(chunk.tiles[tile].field.d_celly, 0    , chunk.tiles[tile].field.celly_size * sizeof(double)));
        gpuErrchk(cudaMalloc(&chunk.tiles[tile].field.d_vertexx  , chunk.tiles[tile].field.vertexx_size * sizeof(double)));
        gpuErrchk(cudaMemset(chunk.tiles[tile].field.d_vertexx, 0  , chunk.tiles[tile].field.vertexx_size * sizeof(double)));
        gpuErrchk(cudaMalloc(&chunk.tiles[tile].field.d_vertexy  , chunk.tiles[tile].field.vertexy_size * sizeof(double)));
        gpuErrchk(cudaMemset(chunk.tiles[tile].field.d_vertexy, 0  , chunk.tiles[tile].field.vertexy_size * sizeof(double)));
        gpuErrchk(cudaMalloc(&chunk.tiles[tile].field.d_celldx   , chunk.tiles[tile].field.celldx_size * sizeof(double)));
        gpuErrchk(cudaMemset(chunk.tiles[tile].field.d_celldx, 0   , chunk.tiles[tile].field.celldx_size * sizeof(double)));
        gpuErrchk(cudaMalloc(&chunk.tiles[tile].field.d_celldy   , chunk.tiles[tile].field.celldy_size * sizeof(double)));
        gpuErrchk(cudaMemset(chunk.tiles[tile].field.d_celldy, 0   , chunk.tiles[tile].field.celldy_size * sizeof(double)));
        gpuErrchk(cudaMalloc(&chunk.tiles[tile].field.d_vertexdx , chunk.tiles[tile].field.vertexdx_size * sizeof(double)));
        gpuErrchk(cudaMemset(chunk.tiles[tile].field.d_vertexdx, 0 , chunk.tiles[tile].field.vertexdx_size * sizeof(double)));
        gpuErrchk(cudaMalloc(&chunk.tiles[tile].field.d_vertexdy , chunk.tiles[tile].field.vertexdy_size * sizeof(double)));
        gpuErrchk(cudaMemset(chunk.tiles[tile].field.d_vertexdy, 0 , chunk.tiles[tile].field.vertexdy_size * sizeof(double)));
        gpuErrchk(cudaMalloc(&chunk.tiles[tile].field.d_volume   , chunk.tiles[tile].field.volume_size * sizeof(double)));
        gpuErrchk(cudaMemset(chunk.tiles[tile].field.d_volume, 0   , chunk.tiles[tile].field.volume_size * sizeof(double)));
        gpuErrchk(cudaMalloc(&chunk.tiles[tile].field.d_xarea    , chunk.tiles[tile].field.xarea_size * sizeof(double)));
        gpuErrchk(cudaMemset(chunk.tiles[tile].field.d_xarea, 0    , chunk.tiles[tile].field.xarea_size * sizeof(double)));
        gpuErrchk(cudaMalloc(&chunk.tiles[tile].field.d_yarea    , chunk.tiles[tile].field.yarea_size * sizeof(double)));
        gpuErrchk(cudaMemset(chunk.tiles[tile].field.d_yarea, 0    , chunk.tiles[tile].field.yarea_size * sizeof(double)));


        // TODO first touch
    }
}
