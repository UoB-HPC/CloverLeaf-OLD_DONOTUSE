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

        chunk.tiles[tile].field.d_density0   =  Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space>("density0",   size1d(ymin - 2, ymax + 2), size1d(xmin - 2, xmax + 2));
        chunk.tiles[tile].field.d_density1   =  Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space>("density1",   size1d(ymin - 2, ymax + 2), size1d(xmin - 2, xmax + 2));
        chunk.tiles[tile].field.d_energy0    =  Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space>("energy0",    size1d(ymin - 2, ymax + 2), size1d(xmin - 2, xmax + 2));
        chunk.tiles[tile].field.d_energy1    =  Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space>("energy1",    size1d(ymin - 2, ymax + 2), size1d(xmin - 2, xmax + 2));
        chunk.tiles[tile].field.d_pressure   =  Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space>("pressure",   size1d(ymin - 2, ymax + 2), size1d(xmin - 2, xmax + 2));
        chunk.tiles[tile].field.d_viscosity  =  Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space>("viscosity",  size1d(ymin - 2, ymax + 2), size1d(xmin - 2, xmax + 2));
        chunk.tiles[tile].field.d_soundspeed =  Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space>("soundspeed", size1d(ymin - 2, ymax + 2), size1d(xmin - 2, xmax + 2));

        chunk.tiles[tile].field.d_xvel0 =  Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space>("xvel0", size1d(ymin - 2, ymax + 3), size1d(xmin - 2, xmax + 3));
        chunk.tiles[tile].field.d_xvel1 =  Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space>("xvel1", size1d(ymin - 2, ymax + 3), size1d(xmin - 2, xmax + 3));
        chunk.tiles[tile].field.d_yvel0 =  Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space>("yvel0", size1d(ymin - 2, ymax + 3), size1d(xmin - 2, xmax + 3));
        chunk.tiles[tile].field.d_yvel1 =  Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space>("yvel1", size1d(ymin - 2, ymax + 3), size1d(xmin - 2, xmax + 3));

        chunk.tiles[tile].field.d_vol_flux_x  =  Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space>("vol_flux_x",  size1d(ymin - 2, ymax + 2), size1d(xmin - 2, xmax + 3));
        chunk.tiles[tile].field.d_mass_flux_x =  Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space>("mass_flux_x", size1d(ymin - 2, ymax + 2), size1d(xmin - 2, xmax + 3));
        chunk.tiles[tile].field.d_vol_flux_y  =  Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space>("vol_flux_y",  size1d(ymin - 2, ymax + 3), size1d(xmin - 2, xmax + 2));
        chunk.tiles[tile].field.d_mass_flux_y =  Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space>("mass_flux_y", size1d(ymin - 2, ymax + 3), size1d(xmin - 2, xmax + 2));

        chunk.tiles[tile].field.d_work_array1 =  Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space>("work_array1", size1d(ymin - 2, ymax + 3), size1d(xmin - 2, xmax + 3));
        chunk.tiles[tile].field.d_work_array2 =  Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space>("work_array2", size1d(ymin - 2, ymax + 3), size1d(xmin - 2, xmax + 3));
        chunk.tiles[tile].field.d_work_array3 =  Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space>("work_array3", size1d(ymin - 2, ymax + 3), size1d(xmin - 2, xmax + 3));
        chunk.tiles[tile].field.d_work_array4 =  Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space>("work_array4", size1d(ymin - 2, ymax + 3), size1d(xmin - 2, xmax + 3));
        chunk.tiles[tile].field.d_work_array5 =  Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space>("work_array5", size1d(ymin - 2, ymax + 3), size1d(xmin - 2, xmax + 3));
        chunk.tiles[tile].field.d_work_array6 =  Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space>("work_array6", size1d(ymin - 2, ymax + 3), size1d(xmin - 2, xmax + 3));
        chunk.tiles[tile].field.d_work_array7 =  Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space>("work_array7", size1d(ymin - 2, ymax + 3), size1d(xmin - 2, xmax + 3));

        chunk.tiles[tile].field.d_cellx    =  Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space>("cellx",    cellxSize);
        chunk.tiles[tile].field.d_celly    =  Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space>("celly",    cellySize);
        chunk.tiles[tile].field.d_vertexx  =  Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space>("vertexx",  vertexxSize);
        chunk.tiles[tile].field.d_vertexy  =  Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space>("vertexy",  vertexySize);
        chunk.tiles[tile].field.d_celldx   =  Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space>("celldx",   celldxSize);
        chunk.tiles[tile].field.d_celldy   =  Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space>("celldy",   celldySize);
        chunk.tiles[tile].field.d_vertexdx =  Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space>("vertexdx", vertexdxSize);
        chunk.tiles[tile].field.d_vertexdy =  Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space>("vertexdy", vertexdySize);

        chunk.tiles[tile].field.d_volume   =  Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space>("volume", size1d(ymin - 2, ymax + 3), size1d(xmin - 2, xmax + 2));
        chunk.tiles[tile].field.d_xarea    =  Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space>("xarea",  size1d(ymin - 2, ymax + 2), size1d(xmin - 2, xmax + 3));
        chunk.tiles[tile].field.d_yarea    =  Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space>("yarea",  size1d(ymin - 2, ymax + 3), size1d(xmin - 2, xmax + 2));


        chunk.tiles[tile].field.density0   = create_mirror_view(chunk.tiles[tile].field.d_density0);
        chunk.tiles[tile].field.density1   = create_mirror_view(chunk.tiles[tile].field.d_density1);
        chunk.tiles[tile].field.energy0    = create_mirror_view(chunk.tiles[tile].field.d_energy0);
        chunk.tiles[tile].field.energy1    = create_mirror_view(chunk.tiles[tile].field.d_energy1);
        chunk.tiles[tile].field.pressure   = create_mirror_view(chunk.tiles[tile].field.d_pressure);
        chunk.tiles[tile].field.viscosity  = create_mirror_view(chunk.tiles[tile].field.d_viscosity);
        chunk.tiles[tile].field.soundspeed = create_mirror_view(chunk.tiles[tile].field.d_soundspeed);

        chunk.tiles[tile].field.xvel0 = create_mirror_view(chunk.tiles[tile].field.d_xvel0);
        chunk.tiles[tile].field.xvel1 = create_mirror_view(chunk.tiles[tile].field.d_xvel1);
        chunk.tiles[tile].field.yvel0 = create_mirror_view(chunk.tiles[tile].field.d_yvel0);
        chunk.tiles[tile].field.yvel1 = create_mirror_view(chunk.tiles[tile].field.d_yvel1);

        chunk.tiles[tile].field.vol_flux_x  = create_mirror_view(chunk.tiles[tile].field.d_vol_flux_x);
        chunk.tiles[tile].field.mass_flux_x = create_mirror_view(chunk.tiles[tile].field.d_mass_flux_x);
        chunk.tiles[tile].field.vol_flux_y  = create_mirror_view(chunk.tiles[tile].field.d_vol_flux_y);
        chunk.tiles[tile].field.mass_flux_y = create_mirror_view(chunk.tiles[tile].field.d_mass_flux_y);

        chunk.tiles[tile].field.work_array1 = create_mirror_view(chunk.tiles[tile].field.d_work_array1);
        chunk.tiles[tile].field.work_array2 = create_mirror_view(chunk.tiles[tile].field.d_work_array2);
        chunk.tiles[tile].field.work_array3 = create_mirror_view(chunk.tiles[tile].field.d_work_array3);
        chunk.tiles[tile].field.work_array4 = create_mirror_view(chunk.tiles[tile].field.d_work_array4);
        chunk.tiles[tile].field.work_array5 = create_mirror_view(chunk.tiles[tile].field.d_work_array5);
        chunk.tiles[tile].field.work_array6 = create_mirror_view(chunk.tiles[tile].field.d_work_array6);
        chunk.tiles[tile].field.work_array7 = create_mirror_view(chunk.tiles[tile].field.d_work_array7);

        chunk.tiles[tile].field.cellx    = create_mirror_view(chunk.tiles[tile].field.d_cellx);
        chunk.tiles[tile].field.celly    = create_mirror_view(chunk.tiles[tile].field.d_celly);
        chunk.tiles[tile].field.vertexx  = create_mirror_view(chunk.tiles[tile].field.d_vertexx);
        chunk.tiles[tile].field.vertexy  = create_mirror_view(chunk.tiles[tile].field.d_vertexy);
        chunk.tiles[tile].field.celldx   = create_mirror_view(chunk.tiles[tile].field.d_celldx);
        chunk.tiles[tile].field.celldy   = create_mirror_view(chunk.tiles[tile].field.d_celldy);
        chunk.tiles[tile].field.vertexdx = create_mirror_view(chunk.tiles[tile].field.d_vertexdx);
        chunk.tiles[tile].field.vertexdy = create_mirror_view(chunk.tiles[tile].field.d_vertexdy);

        chunk.tiles[tile].field.volume   = create_mirror_view(chunk.tiles[tile].field.d_volume);
        chunk.tiles[tile].field.xarea    = create_mirror_view(chunk.tiles[tile].field.d_xarea);
        chunk.tiles[tile].field.yarea    = create_mirror_view(chunk.tiles[tile].field.d_yarea);
    }
}
