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


        chunk.tiles[tile].field.density0  = new Kokkos::View<double**>("density0", size1d(xmin - 2, xmax + 2), size1d(ymin - 2, ymax + 2));
        chunk.tiles[tile].field.density1  = new Kokkos::View<double**>("density1", size1d(xmin - 2, xmax + 2), size1d(ymin - 2, ymax + 2));
        chunk.tiles[tile].field.energy0   = new Kokkos::View<double**>("energy0", size1d(xmin - 2, xmax + 2), size1d(ymin - 2, ymax + 2));
        chunk.tiles[tile].field.energy1   = new Kokkos::View<double**>("energy1", size1d(xmin - 2, xmax + 2), size1d(ymin - 2, ymax + 2));
        chunk.tiles[tile].field.pressure  = new Kokkos::View<double**>("pressure", size1d(xmin - 2, xmax + 2), size1d(ymin - 2, ymax + 2));
        chunk.tiles[tile].field.viscosity = new Kokkos::View<double**>("viscosity", size1d(xmin - 2, xmax + 2), size1d(ymin - 2, ymax + 2));
        chunk.tiles[tile].field.soundspeed = new Kokkos::View<double**>("soundspeed", size1d(xmin - 2, xmax + 2), size1d(ymin - 2, ymax + 2));

        chunk.tiles[tile].field.xvel0 = new Kokkos::View<double**>("xvel0", size1d(xmin - 2, xmax + 3), size1d(ymin - 2, ymax + 3));
        chunk.tiles[tile].field.xvel1 = new Kokkos::View<double**>("xvel1", size1d(xmin - 2, xmax + 3), size1d(ymin - 2, ymax + 3));
        chunk.tiles[tile].field.yvel0 = new Kokkos::View<double**>("yvel0", size1d(xmin - 2, xmax + 3), size1d(ymin - 2, ymax + 3));
        chunk.tiles[tile].field.yvel1 = new Kokkos::View<double**>("yvel1", size1d(xmin - 2, xmax + 3), size1d(ymin - 2, ymax + 3));

        chunk.tiles[tile].field.vol_flux_x  = new Kokkos::View<double**>("vol_flux_x",  size1d(xmin - 2, xmax + 3), size1d(ymin - 2, ymax + 2));
        chunk.tiles[tile].field.mass_flux_x = new Kokkos::View<double**>("mass_flux_x", size1d(xmin - 2, xmax + 3), size1d(ymin - 2, ymax + 2));
        chunk.tiles[tile].field.vol_flux_y  = new Kokkos::View<double**>("vol_flux_y",  size1d(xmin - 2, xmax + 2), size1d(ymin - 2, ymax + 3));
        chunk.tiles[tile].field.mass_flux_y = new Kokkos::View<double**>("mass_flux_y", size1d(xmin - 2, xmax + 2), size1d(ymin - 2, ymax + 3));

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


        // TODO first touch
    }
}
