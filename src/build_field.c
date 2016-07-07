
#include "build_field.h"
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

void build_field()
{

    for (int tile = 0; tile < tiles_per_chunk; tile++) {
        int xmin = chunk.tiles[tile].t_xmin,
            xmax = chunk.tiles[tile].t_xmax,
            ymin = chunk.tiles[tile].t_ymin,
            ymax = chunk.tiles[tile].t_ymax;

        int density0Size  = size2d(xmin - 2, xmax + 2, ymin - 2, ymax + 2);
        int density1Size  = size2d(xmin - 2, xmax + 2, ymin - 2, ymax + 2);
        int energy0Size   = size2d(xmin - 2, xmax + 2, ymin - 2, ymax + 2);
        int energy1Size   = size2d(xmin - 2, xmax + 2, ymin - 2, ymax + 2);
        int pressureSize  = size2d(xmin - 2, xmax + 2, ymin - 2, ymax + 2);
        int viscositySize = size2d(xmin - 2, xmax + 2, ymin - 2, ymax + 2);
        int soundspeedSize = size2d(xmin - 2, xmax + 2, ymin - 2, ymax + 2);

        int xvel0Size = size2d(xmin - 2, xmax + 3, ymin - 2, ymax + 3);
        int xvel1Size = size2d(xmin - 2, xmax + 3, ymin - 2, ymax + 3);
        int yvel0Size = size2d(xmin - 2, xmax + 3, ymin - 2, ymax + 3);
        int yvel1Size = size2d(xmin - 2, xmax + 3, ymin - 2, ymax + 3);

        int vol_flux_xSize  = size2d(xmin - 2, xmax + 3, ymin - 2, ymax + 2);
        int mass_flux_xSize = size2d(xmin - 2, xmax + 3, ymin - 2, ymax + 2);
        int vol_flux_ySize  = size2d(xmin - 2, xmax + 2, ymin - 2, ymax + 3);
        int mass_flux_ySize = size2d(xmin - 2, xmax + 2, ymin - 2, ymax + 3);

        int work_array1Size = size2d(xmin - 2, xmax + 3, ymin - 2, ymax + 3);
        int work_array2Size = size2d(xmin - 2, xmax + 3, ymin - 2, ymax + 3);
        int work_array3Size = size2d(xmin - 2, xmax + 3, ymin - 2, ymax + 3);
        int work_array4Size = size2d(xmin - 2, xmax + 3, ymin - 2, ymax + 3);
        int work_array5Size = size2d(xmin - 2, xmax + 3, ymin - 2, ymax + 3);
        int work_array6Size = size2d(xmin - 2, xmax + 3, ymin - 2, ymax + 3);
        int work_array7Size = size2d(xmin - 2, xmax + 3, ymin - 2, ymax + 3);

        int cellxSize    = size1d(xmin - 2, xmax + 2);
        int cellySize    = size1d(ymin - 2, ymax + 2);
        int vertexxSize  = size1d(xmin - 2, xmax + 3);
        int vertexySize  = size1d(ymin - 2, ymax + 3);
        int celldxSize   = size1d(xmin - 2, xmax + 2);
        int celldySize   = size1d(ymin - 2, ymax + 2);
        int vertexdxSize = size1d(xmin - 2, xmax + 3);
        int vertexdySize = size1d(ymin - 2, ymax + 3);

        int volumeSize   = size2d(xmin - 2, xmax + 2, ymin - 2, ymax + 2);
        int xareaSize    = size2d(xmin - 2, xmax + 3, ymin - 2, ymax + 2);
        int yareaSize    = size2d(xmin - 2, xmax + 2, ymin - 2, ymax + 3);

        // View<double**> density0("denisty0", size1d(xmin - 2, xmax + 2), size1d(ymin - 2, ymax + 2));

        chunk.tiles[tile].field.density0  = (double*)calloc(sizeof(double), density0Size);
        chunk.tiles[tile].field.density1  = (double*)calloc(sizeof(double), density1Size);
        chunk.tiles[tile].field.energy0   = (double*)calloc(sizeof(double), energy0Size);
        chunk.tiles[tile].field.energy1   = (double*)calloc(sizeof(double), energy1Size);
        chunk.tiles[tile].field.pressure  = (double*)calloc(sizeof(double), pressureSize);
        chunk.tiles[tile].field.viscosity = (double*)calloc(sizeof(double), viscositySize);
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


        // TODO first touch
    }
}
