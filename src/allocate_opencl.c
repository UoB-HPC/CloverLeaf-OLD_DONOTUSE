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

void allocate()
{

    for (int tile = 0; tile < tiles_per_chunk; tile++) {
        int xmin = chunk.tiles[tile].t_xmin,
            xmax = chunk.tiles[tile].t_xmax,
            ymin = chunk.tiles[tile].t_ymin,
            ymax = chunk.tiles[tile].t_ymax;

        int density0Size        = size2d(xmin - 2, xmax + 2, ymin - 2, ymax + 2);
        int density1Size        = size2d(xmin - 2, xmax + 2, ymin - 2, ymax + 2);
        int energy0Size         = size2d(xmin - 2, xmax + 2, ymin - 2, ymax + 2);
        int energy1Size         = size2d(xmin - 2, xmax + 2, ymin - 2, ymax + 2);
        int pressureSize        = size2d(xmin - 2, xmax + 2, ymin - 2, ymax + 2);
        int viscositySize       = size2d(xmin - 2, xmax + 2, ymin - 2, ymax + 2);
        int soundspeedSize      = size2d(xmin - 2, xmax + 2, ymin - 2, ymax + 2);

        int xvel0Size           = size2d(xmin - 2, xmax + 3, ymin - 2, ymax + 3);
        int xvel1Size           = size2d(xmin - 2, xmax + 3, ymin - 2, ymax + 3);
        int yvel0Size           = size2d(xmin - 2, xmax + 3, ymin - 2, ymax + 3);
        int yvel1Size           = size2d(xmin - 2, xmax + 3, ymin - 2, ymax + 3);

        int vol_flux_xSize      = size2d(xmin - 2, xmax + 3, ymin - 2, ymax + 2);
        int mass_flux_xSize     = size2d(xmin - 2, xmax + 3, ymin - 2, ymax + 2);
        int vol_flux_ySize      = size2d(xmin - 2, xmax + 2, ymin - 2, ymax + 3);
        int mass_flux_ySize     = size2d(xmin - 2, xmax + 2, ymin - 2, ymax + 3);

        int work_array1Size     = size2d(xmin - 2, xmax + 3, ymin - 2, ymax + 3);
        int work_array2Size     = size2d(xmin - 2, xmax + 3, ymin - 2, ymax + 3);
        int work_array3Size     = size2d(xmin - 2, xmax + 3, ymin - 2, ymax + 3);
        int work_array4Size     = size2d(xmin - 2, xmax + 3, ymin - 2, ymax + 3);
        int work_array5Size     = size2d(xmin - 2, xmax + 3, ymin - 2, ymax + 3);
        int work_array6Size     = size2d(xmin - 2, xmax + 3, ymin - 2, ymax + 3);
        int work_array7Size     = size2d(xmin - 2, xmax + 3, ymin - 2, ymax + 3);

        int cellxSize           = size1d(xmin - 2, xmax + 2);
        int cellySize           = size1d(ymin - 2, ymax + 2);
        int vertexxSize         = size1d(xmin - 2, xmax + 3);
        int vertexySize         = size1d(ymin - 2, ymax + 3);
        int celldxSize          = size1d(xmin - 2, xmax + 2);
        int celldySize          = size1d(ymin - 2, ymax + 2);
        int vertexdxSize        = size1d(xmin - 2, xmax + 3);
        int vertexdySize        = size1d(ymin - 2, ymax + 3);

        int volumeSize          = size2d(xmin - 2, xmax + 2, ymin - 2, ymax + 2);
        int xareaSize           = size2d(xmin - 2, xmax + 3, ymin - 2, ymax + 2);
        int yareaSize           = size2d(xmin - 2, xmax + 2, ymin - 2, ymax + 3);


        chunk.tiles[tile].field.density0   = (double*)calloc(sizeof(double), density0Size);
        chunk.tiles[tile].field.density1   = (double*)calloc(sizeof(double), density1Size);
        chunk.tiles[tile].field.energy0    = (double*)calloc(sizeof(double), energy0Size);
        chunk.tiles[tile].field.energy1    = (double*)calloc(sizeof(double), energy1Size);
        chunk.tiles[tile].field.pressure   = (double*)calloc(sizeof(double), pressureSize);
        chunk.tiles[tile].field.viscosity  = (double*)calloc(sizeof(double), viscositySize);
        chunk.tiles[tile].field.soundspeed = (double*)calloc(sizeof(double), soundspeedSize);

        chunk.tiles[tile].field.xvel0 = (double*)calloc(sizeof(double), xvel0Size);
        chunk.tiles[tile].field.xvel1 = (double*)calloc(sizeof(double), xvel1Size);
        chunk.tiles[tile].field.yvel0 = (double*)calloc(sizeof(double), yvel0Size);
        chunk.tiles[tile].field.yvel1 = (double*)calloc(sizeof(double), yvel1Size);

        chunk.tiles[tile].field.vol_flux_x  = (double*)calloc(sizeof(double), vol_flux_xSize);
        chunk.tiles[tile].field.mass_flux_x = (double*)calloc(sizeof(double), mass_flux_xSize);
        chunk.tiles[tile].field.vol_flux_y  = (double*)calloc(sizeof(double), vol_flux_ySize);
        chunk.tiles[tile].field.mass_flux_y = (double*)calloc(sizeof(double), mass_flux_ySize);

        chunk.tiles[tile].field.work_array1 = (double*)calloc(sizeof(double), work_array1Size);
        chunk.tiles[tile].field.work_array2 = (double*)calloc(sizeof(double), work_array2Size);
        chunk.tiles[tile].field.work_array3 = (double*)calloc(sizeof(double), work_array3Size);
        chunk.tiles[tile].field.work_array4 = (double*)calloc(sizeof(double), work_array4Size);
        chunk.tiles[tile].field.work_array5 = (double*)calloc(sizeof(double), work_array5Size);
        chunk.tiles[tile].field.work_array6 = (double*)calloc(sizeof(double), work_array6Size);
        chunk.tiles[tile].field.work_array7 = (double*)calloc(sizeof(double), work_array7Size);

        chunk.tiles[tile].field.cellx    = (double*)calloc(sizeof(double), cellxSize);
        chunk.tiles[tile].field.celly    = (double*)calloc(sizeof(double), cellySize);
        chunk.tiles[tile].field.vertexx  = (double*)calloc(sizeof(double), vertexxSize);
        chunk.tiles[tile].field.vertexy  = (double*)calloc(sizeof(double), vertexySize);
        chunk.tiles[tile].field.celldx   = (double*)calloc(sizeof(double), celldxSize);
        chunk.tiles[tile].field.celldy   = (double*)calloc(sizeof(double), celldySize);
        chunk.tiles[tile].field.vertexdx = (double*)calloc(sizeof(double), vertexdxSize);
        chunk.tiles[tile].field.vertexdy = (double*)calloc(sizeof(double), vertexdySize);
        chunk.tiles[tile].field.volume   = (double*)calloc(sizeof(double), volumeSize);
        chunk.tiles[tile].field.xarea    = (double*)calloc(sizeof(double), xareaSize);
        chunk.tiles[tile].field.yarea    = (double*)calloc(sizeof(double), yareaSize);



        chunk.tiles[tile].field.d_density0   = new cl::Buffer(openclContext, CL_MEM_USE_HOST_PTR, sizeof(double) * density0Size , chunk.tiles[tile].field.density0);
        chunk.tiles[tile].field.d_density1   = new cl::Buffer(openclContext, CL_MEM_USE_HOST_PTR, sizeof(double) * density1Size, chunk.tiles[tile].field.density1);
        chunk.tiles[tile].field.d_energy0    = new cl::Buffer(openclContext, CL_MEM_USE_HOST_PTR, sizeof(double) * energy0Size, chunk.tiles[tile].field.energy0);
        chunk.tiles[tile].field.d_energy1    = new cl::Buffer(openclContext, CL_MEM_USE_HOST_PTR, sizeof(double) * energy1Size, chunk.tiles[tile].field.energy1);
        chunk.tiles[tile].field.d_pressure   = new cl::Buffer(openclContext, CL_MEM_USE_HOST_PTR, sizeof(double) * pressureSize, chunk.tiles[tile].field.pressure);
        chunk.tiles[tile].field.d_viscosity  = new cl::Buffer(openclContext, CL_MEM_USE_HOST_PTR, sizeof(double) * viscositySize, chunk.tiles[tile].field.viscosity);
        chunk.tiles[tile].field.d_soundspeed = new cl::Buffer(openclContext, CL_MEM_USE_HOST_PTR, sizeof(double) * soundspeedSize, chunk.tiles[tile].field.soundspeed);

        chunk.tiles[tile].field.d_xvel0 = new cl::Buffer(openclContext, CL_MEM_USE_HOST_PTR, sizeof(double) * xvel0Size, chunk.tiles[tile].field.xvel0);
        chunk.tiles[tile].field.d_xvel1 = new cl::Buffer(openclContext, CL_MEM_USE_HOST_PTR, sizeof(double) * xvel1Size, chunk.tiles[tile].field.xvel1);
        chunk.tiles[tile].field.d_yvel0 = new cl::Buffer(openclContext, CL_MEM_USE_HOST_PTR, sizeof(double) * yvel0Size, chunk.tiles[tile].field.yvel0);
        chunk.tiles[tile].field.d_yvel1 = new cl::Buffer(openclContext, CL_MEM_USE_HOST_PTR, sizeof(double) * yvel1Size, chunk.tiles[tile].field.yvel1);

        chunk.tiles[tile].field.d_vol_flux_x  = new cl::Buffer(openclContext, CL_MEM_USE_HOST_PTR, sizeof(double) * vol_flux_xSize, chunk.tiles[tile].field.vol_flux_x);
        chunk.tiles[tile].field.d_mass_flux_x = new cl::Buffer(openclContext, CL_MEM_USE_HOST_PTR, sizeof(double) * mass_flux_xSize, chunk.tiles[tile].field.mass_flux_x);
        chunk.tiles[tile].field.d_vol_flux_y  = new cl::Buffer(openclContext, CL_MEM_USE_HOST_PTR, sizeof(double) * vol_flux_ySize, chunk.tiles[tile].field.vol_flux_y);
        chunk.tiles[tile].field.d_mass_flux_y = new cl::Buffer(openclContext, CL_MEM_USE_HOST_PTR, sizeof(double) * mass_flux_ySize, chunk.tiles[tile].field.mass_flux_y);

        chunk.tiles[tile].field.d_work_array1 = new cl::Buffer(openclContext, CL_MEM_USE_HOST_PTR, sizeof(double) * work_array1Size, chunk.tiles[tile].field.work_array1);
        chunk.tiles[tile].field.d_work_array2 = new cl::Buffer(openclContext, CL_MEM_USE_HOST_PTR, sizeof(double) * work_array2Size, chunk.tiles[tile].field.work_array2);
        chunk.tiles[tile].field.d_work_array3 = new cl::Buffer(openclContext, CL_MEM_USE_HOST_PTR, sizeof(double) * work_array3Size, chunk.tiles[tile].field.work_array3);
        chunk.tiles[tile].field.d_work_array4 = new cl::Buffer(openclContext, CL_MEM_USE_HOST_PTR, sizeof(double) * work_array4Size, chunk.tiles[tile].field.work_array4);
        chunk.tiles[tile].field.d_work_array5 = new cl::Buffer(openclContext, CL_MEM_USE_HOST_PTR, sizeof(double) * work_array5Size, chunk.tiles[tile].field.work_array5);
        chunk.tiles[tile].field.d_work_array6 = new cl::Buffer(openclContext, CL_MEM_USE_HOST_PTR, sizeof(double) * work_array6Size, chunk.tiles[tile].field.work_array6);
        chunk.tiles[tile].field.d_work_array7 = new cl::Buffer(openclContext, CL_MEM_USE_HOST_PTR, sizeof(double) * work_array7Size, chunk.tiles[tile].field.work_array7);

        chunk.tiles[tile].field.d_cellx    = new cl::Buffer(openclContext, CL_MEM_USE_HOST_PTR, sizeof(double)* cellxSize, chunk.tiles[tile].field.cellx);
        chunk.tiles[tile].field.d_celly    = new cl::Buffer(openclContext, CL_MEM_USE_HOST_PTR, sizeof(double)* cellySize, chunk.tiles[tile].field.celly);
        chunk.tiles[tile].field.d_vertexx  = new cl::Buffer(openclContext, CL_MEM_USE_HOST_PTR, sizeof(double)* vertexxSize, chunk.tiles[tile].field.vertexx);
        chunk.tiles[tile].field.d_vertexy  = new cl::Buffer(openclContext, CL_MEM_USE_HOST_PTR, sizeof(double)* vertexySize, chunk.tiles[tile].field.vertexy);
        chunk.tiles[tile].field.d_celldx   = new cl::Buffer(openclContext, CL_MEM_USE_HOST_PTR, sizeof(double)* celldxSize, chunk.tiles[tile].field.celldx);
        chunk.tiles[tile].field.d_celldy   = new cl::Buffer(openclContext, CL_MEM_USE_HOST_PTR, sizeof(double)* celldySize, chunk.tiles[tile].field.celldy);
        chunk.tiles[tile].field.d_vertexdx = new cl::Buffer(openclContext, CL_MEM_USE_HOST_PTR, sizeof(double)* vertexdxSize, chunk.tiles[tile].field.vertexdx);
        chunk.tiles[tile].field.d_vertexdy = new cl::Buffer(openclContext, CL_MEM_USE_HOST_PTR, sizeof(double)* vertexdySize, chunk.tiles[tile].field.vertexdy);
        chunk.tiles[tile].field.d_volume   = new cl::Buffer(openclContext, CL_MEM_USE_HOST_PTR, sizeof(double)* volumeSize, chunk.tiles[tile].field.volume);
        chunk.tiles[tile].field.d_xarea    = new cl::Buffer(openclContext, CL_MEM_USE_HOST_PTR, sizeof(double)* xareaSize, chunk.tiles[tile].field.xarea);
        chunk.tiles[tile].field.d_yarea    = new cl::Buffer(openclContext, CL_MEM_USE_HOST_PTR, sizeof(double)* yareaSize, chunk.tiles[tile].field.yarea);


        // TODO first touch
    }
}
