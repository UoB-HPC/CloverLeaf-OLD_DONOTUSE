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

        new(&chunk.tiles[tile].field.d_density0)     Kokkos::View<double**>("density0",   size1d(ymin - 2, ymax + 2), size1d(xmin - 2, xmax + 2));
        new(&chunk.tiles[tile].field.d_density1)     Kokkos::View<double**>("density1",   size1d(ymin - 2, ymax + 2), size1d(xmin - 2, xmax + 2));
        new(&chunk.tiles[tile].field.d_energy0)      Kokkos::View<double**>("energy0",    size1d(ymin - 2, ymax + 2), size1d(xmin - 2, xmax + 2));
        new(&chunk.tiles[tile].field.d_energy1)      Kokkos::View<double**>("energy1",    size1d(ymin - 2, ymax + 2), size1d(xmin - 2, xmax + 2));
        new(&chunk.tiles[tile].field.d_pressure)     Kokkos::View<double**>("pressure",   size1d(ymin - 2, ymax + 2), size1d(xmin - 2, xmax + 2));
        new(&chunk.tiles[tile].field.d_viscosity)    Kokkos::View<double**>("viscosity",  size1d(ymin - 2, ymax + 2), size1d(xmin - 2, xmax + 2));
        new(&chunk.tiles[tile].field.d_soundspeed)   Kokkos::View<double**>("soundspeed", size1d(ymin - 2, ymax + 2), size1d(xmin - 2, xmax + 2));

        new(&chunk.tiles[tile].field.d_xvel0)   Kokkos::View<double**>("xvel0", size1d(ymin - 2, ymax + 3), size1d(xmin - 2, xmax + 3));
        new(&chunk.tiles[tile].field.d_xvel1)   Kokkos::View<double**>("xvel1", size1d(ymin - 2, ymax + 3), size1d(xmin - 2, xmax + 3));
        new(&chunk.tiles[tile].field.d_yvel0)   Kokkos::View<double**>("yvel0", size1d(ymin - 2, ymax + 3), size1d(xmin - 2, xmax + 3));
        new(&chunk.tiles[tile].field.d_yvel1)   Kokkos::View<double**>("yvel1", size1d(ymin - 2, ymax + 3), size1d(xmin - 2, xmax + 3));

        new(&chunk.tiles[tile].field.d_vol_flux_x)    Kokkos::View<double**>("vol_flux_x",  size1d(ymin - 2, ymax + 2), size1d(xmin - 2, xmax + 3));
        new(&chunk.tiles[tile].field.d_mass_flux_x)   Kokkos::View<double**>("mass_flux_x", size1d(ymin - 2, ymax + 2), size1d(xmin - 2, xmax + 3));
        new(&chunk.tiles[tile].field.d_vol_flux_y)    Kokkos::View<double**>("vol_flux_y",  size1d(ymin - 2, ymax + 3), size1d(xmin - 2, xmax + 2));
        new(&chunk.tiles[tile].field.d_mass_flux_y)   Kokkos::View<double**>("mass_flux_y", size1d(ymin - 2, ymax + 3), size1d(xmin - 2, xmax + 2));

        new(&chunk.tiles[tile].field.d_work_array1)   Kokkos::View<double**>("work_array1", size1d(ymin - 2, ymax + 3), size1d(xmin - 2, xmax + 3));
        new(&chunk.tiles[tile].field.d_work_array2)   Kokkos::View<double**>("work_array2", size1d(ymin - 2, ymax + 3), size1d(xmin - 2, xmax + 3));
        new(&chunk.tiles[tile].field.d_work_array3)   Kokkos::View<double**>("work_array3", size1d(ymin - 2, ymax + 3), size1d(xmin - 2, xmax + 3));
        new(&chunk.tiles[tile].field.d_work_array4)   Kokkos::View<double**>("work_array4", size1d(ymin - 2, ymax + 3), size1d(xmin - 2, xmax + 3));
        new(&chunk.tiles[tile].field.d_work_array5)   Kokkos::View<double**>("work_array5", size1d(ymin - 2, ymax + 3), size1d(xmin - 2, xmax + 3));
        new(&chunk.tiles[tile].field.d_work_array6)   Kokkos::View<double**>("work_array6", size1d(ymin - 2, ymax + 3), size1d(xmin - 2, xmax + 3));
        new(&chunk.tiles[tile].field.d_work_array7)   Kokkos::View<double**>("work_array7", size1d(ymin - 2, ymax + 3), size1d(xmin - 2, xmax + 3));

        new(&chunk.tiles[tile].field.d_cellx)      Kokkos::View<double*>("cellx",    cellxSize);
        new(&chunk.tiles[tile].field.d_celly)      Kokkos::View<double*>("celly",    cellySize);
        new(&chunk.tiles[tile].field.d_vertexx)    Kokkos::View<double*>("vertexx",  vertexxSize);
        new(&chunk.tiles[tile].field.d_vertexy)    Kokkos::View<double*>("vertexy",  vertexySize);
        new(&chunk.tiles[tile].field.d_celldx)     Kokkos::View<double*>("celldx",   celldxSize);
        new(&chunk.tiles[tile].field.d_celldy)     Kokkos::View<double*>("celldy",   celldySize);
        new(&chunk.tiles[tile].field.d_vertexdx)   Kokkos::View<double*>("vertexdx", vertexdxSize);
        new(&chunk.tiles[tile].field.d_vertexdy)   Kokkos::View<double*>("vertexdy", vertexdySize);

        new(&chunk.tiles[tile].field.d_volume)     Kokkos::View<double**>("volume", size1d(ymin - 2, ymax + 3), size1d(xmin - 2, xmax + 2));
        new(&chunk.tiles[tile].field.d_xarea)      Kokkos::View<double**>("xarea",  size1d(ymin - 2, ymax + 2), size1d(xmin - 2, xmax + 3));
        new(&chunk.tiles[tile].field.d_yarea)      Kokkos::View<double**>("yarea",  size1d(ymin - 2, ymax + 3), size1d(xmin - 2, xmax + 2));


        new(&chunk.tiles[tile].field.density0)     host_view_2d_t(create_mirror_view(chunk.tiles[tile].field.d_density0));
        new(&chunk.tiles[tile].field.density1)     host_view_2d_t(create_mirror_view(chunk.tiles[tile].field.d_density1));
        new(&chunk.tiles[tile].field.energy0)      host_view_2d_t(create_mirror_view(chunk.tiles[tile].field.d_energy0));
        new(&chunk.tiles[tile].field.energy1)      host_view_2d_t(create_mirror_view(chunk.tiles[tile].field.d_energy1));
        new(&chunk.tiles[tile].field.pressure)     host_view_2d_t(create_mirror_view(chunk.tiles[tile].field.d_pressure));
        new(&chunk.tiles[tile].field.viscosity)    host_view_2d_t(create_mirror_view(chunk.tiles[tile].field.d_viscosity));
        new(&chunk.tiles[tile].field.soundspeed)   host_view_2d_t(create_mirror_view(chunk.tiles[tile].field.d_soundspeed));

        new(&chunk.tiles[tile].field.xvel0)        host_view_2d_t(create_mirror_view(chunk.tiles[tile].field.d_xvel0));
        new(&chunk.tiles[tile].field.xvel1)        host_view_2d_t(create_mirror_view(chunk.tiles[tile].field.d_xvel1));
        new(&chunk.tiles[tile].field.yvel0)        host_view_2d_t(create_mirror_view(chunk.tiles[tile].field.d_yvel0));
        new(&chunk.tiles[tile].field.yvel1)        host_view_2d_t(create_mirror_view(chunk.tiles[tile].field.d_yvel1));

        new(&chunk.tiles[tile].field.vol_flux_x)   host_view_2d_t(create_mirror_view(chunk.tiles[tile].field.d_vol_flux_x));
        new(&chunk.tiles[tile].field.mass_flux_x)  host_view_2d_t(create_mirror_view(chunk.tiles[tile].field.d_mass_flux_x));
        new(&chunk.tiles[tile].field.vol_flux_y)   host_view_2d_t(create_mirror_view(chunk.tiles[tile].field.d_vol_flux_y));
        new(&chunk.tiles[tile].field.mass_flux_y)  host_view_2d_t(create_mirror_view(chunk.tiles[tile].field.d_mass_flux_y));

        new(&chunk.tiles[tile].field.work_array1)  host_view_2d_t(create_mirror_view(chunk.tiles[tile].field.d_work_array1));
        new(&chunk.tiles[tile].field.work_array2)  host_view_2d_t(create_mirror_view(chunk.tiles[tile].field.d_work_array2));
        new(&chunk.tiles[tile].field.work_array3)  host_view_2d_t(create_mirror_view(chunk.tiles[tile].field.d_work_array3));
        new(&chunk.tiles[tile].field.work_array4)  host_view_2d_t(create_mirror_view(chunk.tiles[tile].field.d_work_array4));
        new(&chunk.tiles[tile].field.work_array5)  host_view_2d_t(create_mirror_view(chunk.tiles[tile].field.d_work_array5));
        new(&chunk.tiles[tile].field.work_array6)  host_view_2d_t(create_mirror_view(chunk.tiles[tile].field.d_work_array6));
        new(&chunk.tiles[tile].field.work_array7)  host_view_2d_t(create_mirror_view(chunk.tiles[tile].field.d_work_array7));

        new(&chunk.tiles[tile].field.cellx)        host_view_1d_t(create_mirror_view(chunk.tiles[tile].field.d_cellx));
        new(&chunk.tiles[tile].field.celly)        host_view_1d_t(create_mirror_view(chunk.tiles[tile].field.d_celly));
        new(&chunk.tiles[tile].field.vertexx)      host_view_1d_t(create_mirror_view(chunk.tiles[tile].field.d_vertexx));
        new(&chunk.tiles[tile].field.vertexy)      host_view_1d_t(create_mirror_view(chunk.tiles[tile].field.d_vertexy));
        new(&chunk.tiles[tile].field.celldx)       host_view_1d_t(create_mirror_view(chunk.tiles[tile].field.d_celldx));
        new(&chunk.tiles[tile].field.celldy)       host_view_1d_t(create_mirror_view(chunk.tiles[tile].field.d_celldy));
        new(&chunk.tiles[tile].field.vertexdx)     host_view_1d_t(create_mirror_view(chunk.tiles[tile].field.d_vertexdx));
        new(&chunk.tiles[tile].field.vertexdy)     host_view_1d_t(create_mirror_view(chunk.tiles[tile].field.d_vertexdy));

        new(&chunk.tiles[tile].field.volume)       host_view_2d_t(create_mirror_view(chunk.tiles[tile].field.d_volume));
        new(&chunk.tiles[tile].field.xarea)        host_view_2d_t(create_mirror_view(chunk.tiles[tile].field.d_xarea));
        new(&chunk.tiles[tile].field.yarea)        host_view_2d_t(create_mirror_view(chunk.tiles[tile].field.d_yarea));
    }
}
