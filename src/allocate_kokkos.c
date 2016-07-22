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
using namespace Kokkos;
void allocate()
{
    for (int tile = 0; tile < tiles_per_chunk; tile++) {
        int xmin = chunk.tiles[tile].t_xmin,
            xmax = chunk.tiles[tile].t_xmax,
            ymin = chunk.tiles[tile].t_ymin,
            ymax = chunk.tiles[tile].t_ymax;

        int cellxSize    = size1d(xmin - 2, xmax + 2);
        int cellySize    = size1d(ymin - 2, ymax + 2);
        int vertexxSize  = size1d(xmin - 2, xmax + 3);
        int vertexySize  = size1d(ymin - 2, ymax + 3);
        int celldxSize   = size1d(xmin - 2, xmax + 2);
        int celldySize   = size1d(ymin - 2, ymax + 2);
        int vertexdxSize = size1d(xmin - 2, xmax + 3);
        int vertexdySize = size1d(ymin - 2, ymax + 3);

        chunk.tiles[tile].field.density0   = new Kokkos::View<double** >("density0",   size1d(ymin - 2, ymax + 2), size1d(xmin - 2, xmax + 2));
        chunk.tiles[tile].field.density1   = new Kokkos::View<double** >("density1",   size1d(ymin - 2, ymax + 2), size1d(xmin - 2, xmax + 2));
        chunk.tiles[tile].field.energy0    = new Kokkos::View<double** >("energy0",    size1d(ymin - 2, ymax + 2), size1d(xmin - 2, xmax + 2));
        chunk.tiles[tile].field.energy1    = new Kokkos::View<double** >("energy1",    size1d(ymin - 2, ymax + 2), size1d(xmin - 2, xmax + 2));
        chunk.tiles[tile].field.pressure   = new Kokkos::View<double** >("pressure",   size1d(ymin - 2, ymax + 2), size1d(xmin - 2, xmax + 2));
        chunk.tiles[tile].field.viscosity  = new Kokkos::View<double** >("viscosity",  size1d(ymin - 2, ymax + 2), size1d(xmin - 2, xmax + 2));
        chunk.tiles[tile].field.soundspeed = new Kokkos::View<double** >("soundspeed", size1d(ymin - 2, ymax + 2), size1d(xmin - 2, xmax + 2));

        chunk.tiles[tile].field.xvel0 = new Kokkos::View<double** >("xvel0", size1d(ymin - 2, ymax + 3), size1d(xmin - 2, xmax + 3));
        chunk.tiles[tile].field.xvel1 = new Kokkos::View<double** >("xvel1", size1d(ymin - 2, ymax + 3), size1d(xmin - 2, xmax + 3));
        chunk.tiles[tile].field.yvel0 = new Kokkos::View<double** >("yvel0", size1d(ymin - 2, ymax + 3), size1d(xmin - 2, xmax + 3));
        chunk.tiles[tile].field.yvel1 = new Kokkos::View<double** >("yvel1", size1d(ymin - 2, ymax + 3), size1d(xmin - 2, xmax + 3));

        chunk.tiles[tile].field.vol_flux_x  = new Kokkos::View<double** >("vol_flux_x",  size1d(ymin - 2, ymax + 2), size1d(xmin - 2, xmax + 3));
        chunk.tiles[tile].field.mass_flux_x = new Kokkos::View<double** >("mass_flux_x", size1d(ymin - 2, ymax + 2), size1d(xmin - 2, xmax + 3));
        chunk.tiles[tile].field.vol_flux_y  = new Kokkos::View<double** >("vol_flux_y",  size1d(ymin - 2, ymax + 3), size1d(xmin - 2, xmax + 2));
        chunk.tiles[tile].field.mass_flux_y = new Kokkos::View<double** >("mass_flux_y", size1d(ymin - 2, ymax + 3), size1d(xmin - 2, xmax + 2));

        chunk.tiles[tile].field.work_array1 = new Kokkos::View<double** >("work_array1", size1d(ymin - 2, ymax + 3), size1d(xmin - 2, xmax + 3));
        chunk.tiles[tile].field.work_array2 = new Kokkos::View<double** >("work_array2", size1d(ymin - 2, ymax + 3), size1d(xmin - 2, xmax + 3));
        chunk.tiles[tile].field.work_array3 = new Kokkos::View<double** >("work_array3", size1d(ymin - 2, ymax + 3), size1d(xmin - 2, xmax + 3));
        chunk.tiles[tile].field.work_array4 = new Kokkos::View<double** >("work_array4", size1d(ymin - 2, ymax + 3), size1d(xmin - 2, xmax + 3));
        chunk.tiles[tile].field.work_array5 = new Kokkos::View<double** >("work_array5", size1d(ymin - 2, ymax + 3), size1d(xmin - 2, xmax + 3));
        chunk.tiles[tile].field.work_array6 = new Kokkos::View<double** >("work_array6", size1d(ymin - 2, ymax + 3), size1d(xmin - 2, xmax + 3));
        chunk.tiles[tile].field.work_array7 = new Kokkos::View<double** >("work_array7", size1d(ymin - 2, ymax + 3), size1d(xmin - 2, xmax + 3));

        chunk.tiles[tile].field.cellx    = new Kokkos::View<double* >("cellx",    cellxSize);
        chunk.tiles[tile].field.celly    = new Kokkos::View<double* >("celly",    cellySize);
        chunk.tiles[tile].field.vertexx  = new Kokkos::View<double* >("vertexx",  vertexxSize);
        chunk.tiles[tile].field.vertexy  = new Kokkos::View<double* >("vertexy",  vertexySize);
        chunk.tiles[tile].field.celldx   = new Kokkos::View<double* >("celldx",   celldxSize);
        chunk.tiles[tile].field.celldy   = new Kokkos::View<double* >("celldy",   celldySize);
        chunk.tiles[tile].field.vertexdx = new Kokkos::View<double* >("vertexdx", vertexdxSize);
        chunk.tiles[tile].field.vertexdy = new Kokkos::View<double* >("vertexdy", vertexdySize);

        chunk.tiles[tile].field.volume   = new Kokkos::View<double** >("volume", size1d(ymin - 2, ymax + 3), size1d(xmin - 2, xmax + 2));
        chunk.tiles[tile].field.xarea    = new Kokkos::View<double** >("xarea",  size1d(ymin - 2, ymax + 2), size1d(xmin - 2, xmax + 3));
        chunk.tiles[tile].field.yarea    = new Kokkos::View<double** >("yarea",  size1d(ymin - 2, ymax + 3), size1d(xmin - 2, xmax + 2));


        // TODO first touch
    }
    fprintf(stderr, "lmao2\n");
}
