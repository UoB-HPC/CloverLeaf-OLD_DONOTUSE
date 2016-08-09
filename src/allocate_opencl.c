#include "definitions_c.h"
#include <stdlib.h>

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



        chunk.tiles[tile].field.d_density0   = new cl::Buffer(openclContext, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(double) * chunk.tiles[tile].field.density0_size , chunk.tiles[tile].field.density0);
        chunk.tiles[tile].field.d_density1   = new cl::Buffer(openclContext, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(double) * chunk.tiles[tile].field.density1_size, chunk.tiles[tile].field.density1);
        chunk.tiles[tile].field.d_energy0    = new cl::Buffer(openclContext, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(double) * chunk.tiles[tile].field.energy0_size, chunk.tiles[tile].field.energy0);
        chunk.tiles[tile].field.d_energy1    = new cl::Buffer(openclContext, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(double) * chunk.tiles[tile].field.energy1_size, chunk.tiles[tile].field.energy1);
        chunk.tiles[tile].field.d_pressure   = new cl::Buffer(openclContext, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(double) * chunk.tiles[tile].field.pressure_size, chunk.tiles[tile].field.pressure);
        chunk.tiles[tile].field.d_viscosity  = new cl::Buffer(openclContext, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(double) * chunk.tiles[tile].field.viscosity_size, chunk.tiles[tile].field.viscosity);
        chunk.tiles[tile].field.d_soundspeed = new cl::Buffer(openclContext, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(double) * chunk.tiles[tile].field.soundspeed_size, chunk.tiles[tile].field.soundspeed);

        chunk.tiles[tile].field.d_xvel0 = new cl::Buffer(openclContext, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(double) * chunk.tiles[tile].field.xvel0_size, chunk.tiles[tile].field.xvel0);
        chunk.tiles[tile].field.d_xvel1 = new cl::Buffer(openclContext, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(double) * chunk.tiles[tile].field.xvel1_size, chunk.tiles[tile].field.xvel1);
        chunk.tiles[tile].field.d_yvel0 = new cl::Buffer(openclContext, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(double) * chunk.tiles[tile].field.yvel0_size, chunk.tiles[tile].field.yvel0);
        chunk.tiles[tile].field.d_yvel1 = new cl::Buffer(openclContext, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(double) * chunk.tiles[tile].field.yvel1_size, chunk.tiles[tile].field.yvel1);

        chunk.tiles[tile].field.d_vol_flux_x  = new cl::Buffer(openclContext, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(double) * chunk.tiles[tile].field.vol_flux_x_size, chunk.tiles[tile].field.vol_flux_x);
        chunk.tiles[tile].field.d_mass_flux_x = new cl::Buffer(openclContext, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(double) * chunk.tiles[tile].field.mass_flux_x_size, chunk.tiles[tile].field.mass_flux_x);
        chunk.tiles[tile].field.d_vol_flux_y  = new cl::Buffer(openclContext, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(double) * chunk.tiles[tile].field.vol_flux_y_size, chunk.tiles[tile].field.vol_flux_y);
        chunk.tiles[tile].field.d_mass_flux_y = new cl::Buffer(openclContext, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(double) * chunk.tiles[tile].field.mass_flux_y_size, chunk.tiles[tile].field.mass_flux_y);

        chunk.tiles[tile].field.d_work_array1 = new cl::Buffer(openclContext, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(double) * chunk.tiles[tile].field.work_array1_size, chunk.tiles[tile].field.work_array1);
        chunk.tiles[tile].field.d_work_array2 = new cl::Buffer(openclContext, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(double) * chunk.tiles[tile].field.work_array2_size, chunk.tiles[tile].field.work_array2);
        chunk.tiles[tile].field.d_work_array3 = new cl::Buffer(openclContext, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(double) * chunk.tiles[tile].field.work_array3_size, chunk.tiles[tile].field.work_array3);
        chunk.tiles[tile].field.d_work_array4 = new cl::Buffer(openclContext, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(double) * chunk.tiles[tile].field.work_array4_size, chunk.tiles[tile].field.work_array4);
        chunk.tiles[tile].field.d_work_array5 = new cl::Buffer(openclContext, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(double) * chunk.tiles[tile].field.work_array5_size, chunk.tiles[tile].field.work_array5);
        chunk.tiles[tile].field.d_work_array6 = new cl::Buffer(openclContext, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(double) * chunk.tiles[tile].field.work_array6_size, chunk.tiles[tile].field.work_array6);
        chunk.tiles[tile].field.d_work_array7 = new cl::Buffer(openclContext, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(double) * chunk.tiles[tile].field.work_array7_size, chunk.tiles[tile].field.work_array7);

        chunk.tiles[tile].field.d_cellx    = new cl::Buffer(openclContext, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(double)* chunk.tiles[tile].field.cellx_size, chunk.tiles[tile].field.cellx);
        chunk.tiles[tile].field.d_celly    = new cl::Buffer(openclContext, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(double)* chunk.tiles[tile].field.celly_size, chunk.tiles[tile].field.celly);
        chunk.tiles[tile].field.d_vertexx  = new cl::Buffer(openclContext, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(double)* chunk.tiles[tile].field.vertexx_size, chunk.tiles[tile].field.vertexx);
        chunk.tiles[tile].field.d_vertexy  = new cl::Buffer(openclContext, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(double)* chunk.tiles[tile].field.vertexy_size, chunk.tiles[tile].field.vertexy);
        chunk.tiles[tile].field.d_celldx   = new cl::Buffer(openclContext, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(double)* chunk.tiles[tile].field.celldx_size, chunk.tiles[tile].field.celldx);
        chunk.tiles[tile].field.d_celldy   = new cl::Buffer(openclContext, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(double)* chunk.tiles[tile].field.celldy_size, chunk.tiles[tile].field.celldy);
        chunk.tiles[tile].field.d_vertexdx = new cl::Buffer(openclContext, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(double)* chunk.tiles[tile].field.vertexdx_size, chunk.tiles[tile].field.vertexdx);
        chunk.tiles[tile].field.d_vertexdy = new cl::Buffer(openclContext, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(double)* chunk.tiles[tile].field.vertexdy_size, chunk.tiles[tile].field.vertexdy);
        chunk.tiles[tile].field.d_volume   = new cl::Buffer(openclContext, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(double)* chunk.tiles[tile].field.volume_size, chunk.tiles[tile].field.volume);
        chunk.tiles[tile].field.d_xarea    = new cl::Buffer(openclContext, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(double)* chunk.tiles[tile].field.xarea_size, chunk.tiles[tile].field.xarea);
        chunk.tiles[tile].field.d_yarea    = new cl::Buffer(openclContext, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(double)* chunk.tiles[tile].field.yarea_size, chunk.tiles[tile].field.yarea);


        // TODO first touch
    }
}
