#include "definitions_c.h"
#include <stdlib.h>
#include <iostream>

int size2d(int xmin, int xmax, int ymin, int ymax)
{
    return (xmax - xmin + 1) * (ymax - ymin + 1);
}

int size1d(int min, int max)
{
    return max - min + 1;
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


        chunk.tiles[tile].field.density0   = NULL;
        chunk.tiles[tile].field.density1   = NULL;
        chunk.tiles[tile].field.energy0    = NULL;
        chunk.tiles[tile].field.energy1    = NULL;
        chunk.tiles[tile].field.pressure   = NULL;
        chunk.tiles[tile].field.viscosity  = NULL;
        chunk.tiles[tile].field.soundspeed = NULL;

        chunk.tiles[tile].field.xvel0 = NULL;
        chunk.tiles[tile].field.xvel1 = NULL;
        chunk.tiles[tile].field.yvel0 = NULL;
        chunk.tiles[tile].field.yvel1 = NULL;

        chunk.tiles[tile].field.vol_flux_x  = NULL;
        chunk.tiles[tile].field.mass_flux_x = NULL;
        chunk.tiles[tile].field.vol_flux_y  = NULL;
        chunk.tiles[tile].field.mass_flux_y = NULL;

        chunk.tiles[tile].field.work_array1 = NULL;
        chunk.tiles[tile].field.work_array2 = NULL;
        chunk.tiles[tile].field.work_array3 = NULL;
        chunk.tiles[tile].field.work_array4 = NULL;
        chunk.tiles[tile].field.work_array5 = NULL;
        chunk.tiles[tile].field.work_array6 = NULL;
        chunk.tiles[tile].field.work_array7 = NULL;

        chunk.tiles[tile].field.cellx    = NULL;
        chunk.tiles[tile].field.celly    = NULL;
        chunk.tiles[tile].field.vertexx  = NULL;
        chunk.tiles[tile].field.vertexy  = NULL;
        chunk.tiles[tile].field.celldx   = NULL;
        chunk.tiles[tile].field.celldy   = NULL;
        chunk.tiles[tile].field.vertexdx = NULL;
        chunk.tiles[tile].field.vertexdy = NULL;
        chunk.tiles[tile].field.volume   = NULL;
        chunk.tiles[tile].field.xarea    = NULL;
        chunk.tiles[tile].field.yarea    = NULL;


        chunk.tiles[tile].field.density0   = (double*)aligned_alloc(sizeof(double) * chunk.tiles[tile].field.density0_size, 4096);
        chunk.tiles[tile].field.density1   = (double*)aligned_alloc(sizeof(double) * chunk.tiles[tile].field.density1_size, 4096);
        chunk.tiles[tile].field.energy0    = (double*)aligned_alloc(sizeof(double) * chunk.tiles[tile].field.energy0_size, 4096);
        chunk.tiles[tile].field.energy1    = (double*)aligned_alloc(sizeof(double) * chunk.tiles[tile].field.energy1_size, 4096);
        chunk.tiles[tile].field.pressure   = (double*)aligned_alloc(sizeof(double) * chunk.tiles[tile].field.pressure_size, 4096);
        chunk.tiles[tile].field.viscosity  = (double*)aligned_alloc(sizeof(double) * chunk.tiles[tile].field.viscosity_size, 4096);
        chunk.tiles[tile].field.soundspeed = (double*)aligned_alloc(sizeof(double) * chunk.tiles[tile].field.soundspeed_size, 4096);

        chunk.tiles[tile].field.xvel0 = (double*)aligned_alloc(sizeof(double) * chunk.tiles[tile].field.xvel0_size, 4096);
        chunk.tiles[tile].field.xvel1 = (double*)aligned_alloc(sizeof(double) * chunk.tiles[tile].field.xvel1_size, 4096);
        chunk.tiles[tile].field.yvel0 = (double*)aligned_alloc(sizeof(double) * chunk.tiles[tile].field.yvel0_size, 4096);
        chunk.tiles[tile].field.yvel1 = (double*)aligned_alloc(sizeof(double) * chunk.tiles[tile].field.yvel1_size, 4096);

        chunk.tiles[tile].field.vol_flux_x  = (double*)aligned_alloc(sizeof(double) * chunk.tiles[tile].field.vol_flux_x_size, 4096);
        chunk.tiles[tile].field.mass_flux_x = (double*)aligned_alloc(sizeof(double) * chunk.tiles[tile].field.mass_flux_x_size, 4096);
        chunk.tiles[tile].field.vol_flux_y  = (double*)aligned_alloc(sizeof(double) * chunk.tiles[tile].field.vol_flux_y_size, 4096);
        chunk.tiles[tile].field.mass_flux_y = (double*)aligned_alloc(sizeof(double) * chunk.tiles[tile].field.mass_flux_y_size, 4096);

        chunk.tiles[tile].field.work_array1 = (double*)aligned_alloc(sizeof(double) * chunk.tiles[tile].field.work_array1_size, 4096);
        chunk.tiles[tile].field.work_array2 = (double*)aligned_alloc(sizeof(double) * chunk.tiles[tile].field.work_array2_size, 4096);
        chunk.tiles[tile].field.work_array3 = (double*)aligned_alloc(sizeof(double) * chunk.tiles[tile].field.work_array3_size, 4096);
        chunk.tiles[tile].field.work_array4 = (double*)aligned_alloc(sizeof(double) * chunk.tiles[tile].field.work_array4_size, 4096);
        chunk.tiles[tile].field.work_array5 = (double*)aligned_alloc(sizeof(double) * chunk.tiles[tile].field.work_array5_size, 4096);
        chunk.tiles[tile].field.work_array6 = (double*)aligned_alloc(sizeof(double) * chunk.tiles[tile].field.work_array6_size, 4096);
        chunk.tiles[tile].field.work_array7 = (double*)aligned_alloc(sizeof(double) * chunk.tiles[tile].field.work_array7_size, 4096);

        chunk.tiles[tile].field.cellx    = (double*)aligned_alloc(sizeof(double) * chunk.tiles[tile].field.cellx_size, 4096);
        chunk.tiles[tile].field.celly    = (double*)aligned_alloc(sizeof(double) * chunk.tiles[tile].field.celly_size, 4096);
        chunk.tiles[tile].field.vertexx  = (double*)aligned_alloc(sizeof(double) * chunk.tiles[tile].field.vertexx_size, 4096);
        chunk.tiles[tile].field.vertexy  = (double*)aligned_alloc(sizeof(double) * chunk.tiles[tile].field.vertexy_size, 4096);
        chunk.tiles[tile].field.celldx   = (double*)aligned_alloc(sizeof(double) * chunk.tiles[tile].field.celldx_size, 4096);
        chunk.tiles[tile].field.celldy   = (double*)aligned_alloc(sizeof(double) * chunk.tiles[tile].field.celldy_size, 4096);
        chunk.tiles[tile].field.vertexdx = (double*)aligned_alloc(sizeof(double) * chunk.tiles[tile].field.vertexdx_size, 4096);
        chunk.tiles[tile].field.vertexdy = (double*)aligned_alloc(sizeof(double) * chunk.tiles[tile].field.vertexdy_size, 4096);
        chunk.tiles[tile].field.volume   = (double*)aligned_alloc(sizeof(double) * chunk.tiles[tile].field.volume_size, 4096);
        chunk.tiles[tile].field.xarea    = (double*)aligned_alloc(sizeof(double) * chunk.tiles[tile].field.xarea_size, 4096);
        chunk.tiles[tile].field.yarea    = (double*)aligned_alloc(sizeof(double) * chunk.tiles[tile].field.yarea_size, 4096);

        cl_int err;

        chunk.tiles[tile].field.d_density0   = new cl::Buffer(openclContext, CL_MEM_READ_WRITE, sizeof(double) * chunk.tiles[tile].field.density0_size, NULL, &err); checkOclErr(err);
        chunk.tiles[tile].field.d_density1   = new cl::Buffer(openclContext, CL_MEM_READ_WRITE, sizeof(double) * chunk.tiles[tile].field.density1_size, NULL, &err); checkOclErr(err);
        chunk.tiles[tile].field.d_energy0    = new cl::Buffer(openclContext, CL_MEM_READ_WRITE, sizeof(double) * chunk.tiles[tile].field.energy0_size, NULL, &err); checkOclErr(err);
        chunk.tiles[tile].field.d_energy1    = new cl::Buffer(openclContext, CL_MEM_READ_WRITE, sizeof(double) * chunk.tiles[tile].field.energy1_size, NULL, &err); checkOclErr(err);
        chunk.tiles[tile].field.d_pressure   = new cl::Buffer(openclContext, CL_MEM_READ_WRITE, sizeof(double) * chunk.tiles[tile].field.pressure_size, NULL, &err); checkOclErr(err);
        chunk.tiles[tile].field.d_viscosity  = new cl::Buffer(openclContext, CL_MEM_READ_WRITE, sizeof(double) * chunk.tiles[tile].field.viscosity_size, NULL, &err); checkOclErr(err);
        chunk.tiles[tile].field.d_soundspeed = new cl::Buffer(openclContext, CL_MEM_READ_WRITE, sizeof(double) * chunk.tiles[tile].field.soundspeed_size, NULL, &err); checkOclErr(err);

        chunk.tiles[tile].field.d_xvel0 = new cl::Buffer(openclContext, CL_MEM_READ_WRITE, sizeof(double) * chunk.tiles[tile].field.xvel0_size, NULL, &err); checkOclErr(err);
        chunk.tiles[tile].field.d_xvel1 = new cl::Buffer(openclContext, CL_MEM_READ_WRITE, sizeof(double) * chunk.tiles[tile].field.xvel1_size, NULL, &err); checkOclErr(err);
        chunk.tiles[tile].field.d_yvel0 = new cl::Buffer(openclContext, CL_MEM_READ_WRITE, sizeof(double) * chunk.tiles[tile].field.yvel0_size, NULL, &err); checkOclErr(err);
        chunk.tiles[tile].field.d_yvel1 = new cl::Buffer(openclContext, CL_MEM_READ_WRITE, sizeof(double) * chunk.tiles[tile].field.yvel1_size, NULL, &err); checkOclErr(err);

        chunk.tiles[tile].field.d_vol_flux_x  = new cl::Buffer(openclContext, CL_MEM_READ_WRITE, sizeof(double) * chunk.tiles[tile].field.vol_flux_x_size, NULL, &err); checkOclErr(err);
        chunk.tiles[tile].field.d_mass_flux_x = new cl::Buffer(openclContext, CL_MEM_READ_WRITE, sizeof(double) * chunk.tiles[tile].field.mass_flux_x_size, NULL, &err); checkOclErr(err);
        chunk.tiles[tile].field.d_vol_flux_y  = new cl::Buffer(openclContext, CL_MEM_READ_WRITE, sizeof(double) * chunk.tiles[tile].field.vol_flux_y_size, NULL, &err); checkOclErr(err);
        chunk.tiles[tile].field.d_mass_flux_y = new cl::Buffer(openclContext, CL_MEM_READ_WRITE, sizeof(double) * chunk.tiles[tile].field.mass_flux_y_size, NULL, &err); checkOclErr(err);

        chunk.tiles[tile].field.d_work_array1 = new cl::Buffer(openclContext, CL_MEM_READ_WRITE, sizeof(double) * chunk.tiles[tile].field.work_array1_size, NULL, &err); checkOclErr(err);
        chunk.tiles[tile].field.d_work_array2 = new cl::Buffer(openclContext, CL_MEM_READ_WRITE, sizeof(double) * chunk.tiles[tile].field.work_array2_size, NULL, &err); checkOclErr(err);
        chunk.tiles[tile].field.d_work_array3 = new cl::Buffer(openclContext, CL_MEM_READ_WRITE, sizeof(double) * chunk.tiles[tile].field.work_array3_size, NULL, &err); checkOclErr(err);
        chunk.tiles[tile].field.d_work_array4 = new cl::Buffer(openclContext, CL_MEM_READ_WRITE, sizeof(double) * chunk.tiles[tile].field.work_array4_size, NULL, &err); checkOclErr(err);
        chunk.tiles[tile].field.d_work_array5 = new cl::Buffer(openclContext, CL_MEM_READ_WRITE, sizeof(double) * chunk.tiles[tile].field.work_array5_size, NULL, &err); checkOclErr(err);
        chunk.tiles[tile].field.d_work_array6 = new cl::Buffer(openclContext, CL_MEM_READ_WRITE, sizeof(double) * chunk.tiles[tile].field.work_array6_size, NULL, &err); checkOclErr(err);
        chunk.tiles[tile].field.d_work_array7 = new cl::Buffer(openclContext, CL_MEM_READ_WRITE, sizeof(double) * chunk.tiles[tile].field.work_array7_size, NULL, &err); checkOclErr(err);

        chunk.tiles[tile].field.d_cellx    = new cl::Buffer(openclContext, CL_MEM_READ_WRITE, sizeof(double)* chunk.tiles[tile].field.cellx_size, NULL, &err); checkOclErr(err);
        chunk.tiles[tile].field.d_celly    = new cl::Buffer(openclContext, CL_MEM_READ_WRITE, sizeof(double)* chunk.tiles[tile].field.celly_size, NULL, &err); checkOclErr(err);
        chunk.tiles[tile].field.d_vertexx  = new cl::Buffer(openclContext, CL_MEM_READ_WRITE, sizeof(double)* chunk.tiles[tile].field.vertexx_size, NULL, &err); checkOclErr(err);
        chunk.tiles[tile].field.d_vertexy  = new cl::Buffer(openclContext, CL_MEM_READ_WRITE, sizeof(double)* chunk.tiles[tile].field.vertexy_size, NULL, &err); checkOclErr(err);
        chunk.tiles[tile].field.d_celldx   = new cl::Buffer(openclContext, CL_MEM_READ_WRITE, sizeof(double)* chunk.tiles[tile].field.celldx_size, NULL, &err); checkOclErr(err);
        chunk.tiles[tile].field.d_celldy   = new cl::Buffer(openclContext, CL_MEM_READ_WRITE, sizeof(double)* chunk.tiles[tile].field.celldy_size, NULL, &err); checkOclErr(err);
        chunk.tiles[tile].field.d_vertexdx = new cl::Buffer(openclContext, CL_MEM_READ_WRITE, sizeof(double)* chunk.tiles[tile].field.vertexdx_size, NULL, &err); checkOclErr(err);
        chunk.tiles[tile].field.d_vertexdy = new cl::Buffer(openclContext, CL_MEM_READ_WRITE, sizeof(double)* chunk.tiles[tile].field.vertexdy_size, NULL, &err); checkOclErr(err);
        chunk.tiles[tile].field.d_volume   = new cl::Buffer(openclContext, CL_MEM_READ_WRITE, sizeof(double)* chunk.tiles[tile].field.volume_size, NULL, &err); checkOclErr(err);
        chunk.tiles[tile].field.d_xarea    = new cl::Buffer(openclContext, CL_MEM_READ_WRITE, sizeof(double)* chunk.tiles[tile].field.xarea_size, NULL, &err); checkOclErr(err);
        chunk.tiles[tile].field.d_yarea    = new cl::Buffer(openclContext, CL_MEM_READ_WRITE, sizeof(double)* chunk.tiles[tile].field.yarea_size, NULL, &err); checkOclErr(err);

        checkOclErr(openclQueue.enqueueFillBuffer(
                        *chunk.tiles[tile].field.d_density0,
                        0,
                        0, chunk.tiles[tile].field.density0_size * sizeof(double)));
        checkOclErr(openclQueue.enqueueFillBuffer(
                        *chunk.tiles[tile].field.d_density1,
                        0,
                        0, chunk.tiles[tile].field.density1_size * sizeof(double)));
        checkOclErr(openclQueue.enqueueFillBuffer(
                        *chunk.tiles[tile].field.d_energy0,
                        0,
                        0, chunk.tiles[tile].field.energy0_size * sizeof(double)));
        checkOclErr(openclQueue.enqueueFillBuffer(
                        *chunk.tiles[tile].field.d_energy1,
                        0,
                        0, chunk.tiles[tile].field.energy1_size * sizeof(double)));
        checkOclErr(openclQueue.enqueueFillBuffer(
                        *chunk.tiles[tile].field.d_pressure,
                        0,
                        0, chunk.tiles[tile].field.pressure_size * sizeof(double)));
        checkOclErr(openclQueue.enqueueFillBuffer(
                        *chunk.tiles[tile].field.d_viscosity,
                        0,
                        0, chunk.tiles[tile].field.viscosity_size * sizeof(double)));
        checkOclErr(openclQueue.enqueueFillBuffer(
                        *chunk.tiles[tile].field.d_soundspeed,
                        0,
                        0, chunk.tiles[tile].field.soundspeed_size * sizeof(double)));
        checkOclErr(openclQueue.enqueueFillBuffer(
                        *chunk.tiles[tile].field.d_xvel0,
                        0,
                        0, chunk.tiles[tile].field.xvel0_size * sizeof(double)));
        checkOclErr(openclQueue.enqueueFillBuffer(
                        *chunk.tiles[tile].field.d_xvel1,
                        0,
                        0, chunk.tiles[tile].field.xvel1_size * sizeof(double)));
        checkOclErr(openclQueue.enqueueFillBuffer(
                        *chunk.tiles[tile].field.d_yvel0,
                        0,
                        0, chunk.tiles[tile].field.yvel0_size * sizeof(double)));
        checkOclErr(openclQueue.enqueueFillBuffer(
                        *chunk.tiles[tile].field.d_yvel1,
                        0,
                        0, chunk.tiles[tile].field.yvel1_size * sizeof(double)));
        checkOclErr(openclQueue.enqueueFillBuffer(
                        *chunk.tiles[tile].field.d_vol_flux_x,
                        0,
                        0, chunk.tiles[tile].field.vol_flux_x_size * sizeof(double)));
        checkOclErr(openclQueue.enqueueFillBuffer(
                        *chunk.tiles[tile].field.d_mass_flux_x,
                        0,
                        0, chunk.tiles[tile].field.mass_flux_x_size * sizeof(double)));
        checkOclErr(openclQueue.enqueueFillBuffer(
                        *chunk.tiles[tile].field.d_vol_flux_y,
                        0,
                        0, chunk.tiles[tile].field.vol_flux_y_size * sizeof(double)));
        checkOclErr(openclQueue.enqueueFillBuffer(
                        *chunk.tiles[tile].field.d_mass_flux_y,
                        0,
                        0, chunk.tiles[tile].field.mass_flux_y_size * sizeof(double)));
        checkOclErr(openclQueue.enqueueFillBuffer(
                        *chunk.tiles[tile].field.d_work_array1,
                        0,
                        0, chunk.tiles[tile].field.work_array1_size * sizeof(double)));
        checkOclErr(openclQueue.enqueueFillBuffer(
                        *chunk.tiles[tile].field.d_work_array2,
                        0,
                        0, chunk.tiles[tile].field.work_array2_size * sizeof(double)));
        checkOclErr(openclQueue.enqueueFillBuffer(
                        *chunk.tiles[tile].field.d_work_array3,
                        0,
                        0, chunk.tiles[tile].field.work_array3_size * sizeof(double)));
        checkOclErr(openclQueue.enqueueFillBuffer(
                        *chunk.tiles[tile].field.d_work_array4,
                        0,
                        0, chunk.tiles[tile].field.work_array4_size * sizeof(double)));
        checkOclErr(openclQueue.enqueueFillBuffer(
                        *chunk.tiles[tile].field.d_work_array5,
                        0,
                        0, chunk.tiles[tile].field.work_array5_size * sizeof(double)));
        checkOclErr(openclQueue.enqueueFillBuffer(
                        *chunk.tiles[tile].field.d_work_array6,
                        0,
                        0, chunk.tiles[tile].field.work_array6_size * sizeof(double)));
        checkOclErr(openclQueue.enqueueFillBuffer(
                        *chunk.tiles[tile].field.d_work_array7,
                        0,
                        0, chunk.tiles[tile].field.work_array7_size * sizeof(double)));
        checkOclErr(openclQueue.enqueueFillBuffer(
                        *chunk.tiles[tile].field.d_cellx,
                        0,
                        0, chunk.tiles[tile].field.cellx_size * sizeof(double)));
        checkOclErr(openclQueue.enqueueFillBuffer(
                        *chunk.tiles[tile].field.d_celly,
                        0,
                        0, chunk.tiles[tile].field.celly_size * sizeof(double)));
        checkOclErr(openclQueue.enqueueFillBuffer(
                        *chunk.tiles[tile].field.d_vertexx,
                        0,
                        0, chunk.tiles[tile].field.vertexx_size * sizeof(double)));
        checkOclErr(openclQueue.enqueueFillBuffer(
                        *chunk.tiles[tile].field.d_vertexy,
                        0,
                        0, chunk.tiles[tile].field.vertexy_size * sizeof(double)));
        checkOclErr(openclQueue.enqueueFillBuffer(
                        *chunk.tiles[tile].field.d_celldx,
                        0,
                        0, chunk.tiles[tile].field.celldx_size * sizeof(double)));
        checkOclErr(openclQueue.enqueueFillBuffer(
                        *chunk.tiles[tile].field.d_celldy,
                        0,
                        0, chunk.tiles[tile].field.celldy_size * sizeof(double)));
        checkOclErr(openclQueue.enqueueFillBuffer(
                        *chunk.tiles[tile].field.d_vertexdx,
                        0,
                        0, chunk.tiles[tile].field.vertexdx_size * sizeof(double)));
        checkOclErr(openclQueue.enqueueFillBuffer(
                        *chunk.tiles[tile].field.d_vertexdy,
                        0,
                        0, chunk.tiles[tile].field.vertexdy_size * sizeof(double)));
        checkOclErr(openclQueue.enqueueFillBuffer(
                        *chunk.tiles[tile].field.d_volume,
                        0,
                        0, chunk.tiles[tile].field.volume_size * sizeof(double)));
        checkOclErr(openclQueue.enqueueFillBuffer(
                        *chunk.tiles[tile].field.d_xarea,
                        0,
                        0, chunk.tiles[tile].field.xarea_size * sizeof(double)));
        checkOclErr(openclQueue.enqueueFillBuffer(
                        *chunk.tiles[tile].field.d_yarea,
                        0,
                        0, chunk.tiles[tile].field.yarea_size * sizeof(double)));
    }
}
