
#include "build_field.h"
#include "definitions_c.h"
#include <stdlib.h>


void build_field()
{
    for (int tile = 0; tile < tiles_per_chunk; tile++) {
        chunk.tiles[tile].field.density0  = calloc(sizeof(double), (-(chunk.tiles[tile].t_xmin - 2) + chunk.tiles[tile].t_xmax + 2 + 1) * (-(chunk.tiles[tile].t_ymin - 2) + (chunk.tiles[tile].t_ymax + 2 + 1)));
        chunk.tiles[tile].field.density1  = calloc(sizeof(double), (-(chunk.tiles[tile].t_xmin - 2) + chunk.tiles[tile].t_xmax + 2 + 1) * (-(chunk.tiles[tile].t_ymin - 2) + (chunk.tiles[tile].t_ymax + 2 + 1)));
        chunk.tiles[tile].field.energy0   = calloc(sizeof(double), (-(chunk.tiles[tile].t_xmin - 2) + chunk.tiles[tile].t_xmax + 2 + 1) * (-(chunk.tiles[tile].t_ymin - 2) + (chunk.tiles[tile].t_ymax + 2 + 1)));
        chunk.tiles[tile].field.energy1   = calloc(sizeof(double), (-(chunk.tiles[tile].t_xmin - 2) + chunk.tiles[tile].t_xmax + 2 + 1) * (-(chunk.tiles[tile].t_ymin - 2) + (chunk.tiles[tile].t_ymax + 2 + 1)));
        chunk.tiles[tile].field.pressure  = calloc(sizeof(double), (-(chunk.tiles[tile].t_xmin - 2) + chunk.tiles[tile].t_xmax + 2 + 1) * (-(chunk.tiles[tile].t_ymin - 2) + (chunk.tiles[tile].t_ymax + 2 + 1)));
        chunk.tiles[tile].field.viscosity = calloc(sizeof(double), (-(chunk.tiles[tile].t_xmin - 2) + chunk.tiles[tile].t_xmax + 2 + 1) * (-(chunk.tiles[tile].t_ymin - 2) + (chunk.tiles[tile].t_ymax + 2 + 1)));
        chunk.tiles[tile].field.soundspeed = calloc(sizeof(double), (-(chunk.tiles[tile].t_xmin - 2) + chunk.tiles[tile].t_xmax + 2 + 1) * (-(chunk.tiles[tile].t_ymin - 2) + (chunk.tiles[tile].t_ymax + 2 + 1)));

        chunk.tiles[tile].field.xvel0 = calloc(sizeof(double), (-(chunk.tiles[tile].t_xmin - 2) + chunk.tiles[tile].t_xmax + 3 + 1) * (-(chunk.tiles[tile].t_ymin - 2) + (chunk.tiles[tile].t_ymax + 3 + 1)));
        chunk.tiles[tile].field.xvel1 = calloc(sizeof(double), (-(chunk.tiles[tile].t_xmin - 2) + chunk.tiles[tile].t_xmax + 3 + 1) * (-(chunk.tiles[tile].t_ymin - 2) + (chunk.tiles[tile].t_ymax + 3 + 1)));
        chunk.tiles[tile].field.yvel0 = calloc(sizeof(double), (-(chunk.tiles[tile].t_xmin - 2) + chunk.tiles[tile].t_xmax + 3 + 1) * (-(chunk.tiles[tile].t_ymin - 2) + (chunk.tiles[tile].t_ymax + 3 + 1)));
        chunk.tiles[tile].field.yvel1 = calloc(sizeof(double), (-(chunk.tiles[tile].t_xmin - 2) + chunk.tiles[tile].t_xmax + 3 + 1) * (-(chunk.tiles[tile].t_ymin - 2) + (chunk.tiles[tile].t_ymax + 3 + 1)));

        chunk.tiles[tile].field.vol_flux_x = calloc(sizeof(double), (-(chunk.tiles[tile].t_xmin - 2) + chunk.tiles[tile].t_xmax + 3 + 1) * (-(chunk.tiles[tile].t_ymin - 2) + (chunk.tiles[tile].t_ymax + 2 + 1)));
        chunk.tiles[tile].field.mass_flux_x = calloc(sizeof(double), (-(chunk.tiles[tile].t_xmin - 2) + chunk.tiles[tile].t_xmax + 3 + 1) * (-(chunk.tiles[tile].t_ymin - 2) + (chunk.tiles[tile].t_ymax + 2 + 1)));
        chunk.tiles[tile].field.vol_flux_y = calloc(sizeof(double), (-(chunk.tiles[tile].t_xmin - 2) + chunk.tiles[tile].t_xmax + 2 + 1) * (-(chunk.tiles[tile].t_ymin - 2) + (chunk.tiles[tile].t_ymax + 3 + 1)));
        chunk.tiles[tile].field.mass_flux_y = calloc(sizeof(double), (-(chunk.tiles[tile].t_xmin - 2) + chunk.tiles[tile].t_xmax + 2 + 1) * (-(chunk.tiles[tile].t_ymin - 2) + (chunk.tiles[tile].t_ymax + 3 + 1)));

        chunk.tiles[tile].field.work_array1 = calloc(sizeof(double), (-(chunk.tiles[tile].t_xmin - 2) + chunk.tiles[tile].t_xmax + 3 + 1) * (-(chunk.tiles[tile].t_ymin - 2) + (chunk.tiles[tile].t_ymax + 3 + 1)));
        chunk.tiles[tile].field.work_array2 = calloc(sizeof(double), (-(chunk.tiles[tile].t_xmin - 2) + chunk.tiles[tile].t_xmax + 3 + 1) * (-(chunk.tiles[tile].t_ymin - 2) + (chunk.tiles[tile].t_ymax + 3 + 1)));
        chunk.tiles[tile].field.work_array3 = calloc(sizeof(double), (-(chunk.tiles[tile].t_xmin - 2) + chunk.tiles[tile].t_xmax + 3 + 1) * (-(chunk.tiles[tile].t_ymin - 2) + (chunk.tiles[tile].t_ymax + 3 + 1)));
        chunk.tiles[tile].field.work_array4 = calloc(sizeof(double), (-(chunk.tiles[tile].t_xmin - 2) + chunk.tiles[tile].t_xmax + 3 + 1) * (-(chunk.tiles[tile].t_ymin - 2) + (chunk.tiles[tile].t_ymax + 3 + 1)));
        chunk.tiles[tile].field.work_array5 = calloc(sizeof(double), (-(chunk.tiles[tile].t_xmin - 2) + chunk.tiles[tile].t_xmax + 3 + 1) * (-(chunk.tiles[tile].t_ymin - 2) + (chunk.tiles[tile].t_ymax + 3 + 1)));
        chunk.tiles[tile].field.work_array6 = calloc(sizeof(double), (-(chunk.tiles[tile].t_xmin - 2) + chunk.tiles[tile].t_xmax + 3 + 1) * (-(chunk.tiles[tile].t_ymin - 2) + (chunk.tiles[tile].t_ymax + 3 + 1)));
        chunk.tiles[tile].field.work_array7 = calloc(sizeof(double), (-(chunk.tiles[tile].t_xmin - 2) + chunk.tiles[tile].t_xmax + 3 + 1) * (-(chunk.tiles[tile].t_ymin - 2) + (chunk.tiles[tile].t_ymax + 3 + 1)));

        chunk.tiles[tile].field.cellx   = calloc(sizeof(double), -(chunk.tiles[tile].t_xmin - 2) + (chunk.tiles[tile].t_xmax + 2 + 1));
        chunk.tiles[tile].field.celly   = calloc(sizeof(double), -(chunk.tiles[tile].t_ymin - 2) + (chunk.tiles[tile].t_ymax + 2 + 1));
        chunk.tiles[tile].field.vertexx = calloc(sizeof(double), -(chunk.tiles[tile].t_xmin - 2) + (chunk.tiles[tile].t_xmax + 3 + 1));
        chunk.tiles[tile].field.vertexy = calloc(sizeof(double), -(chunk.tiles[tile].t_ymin - 2) + (chunk.tiles[tile].t_ymax + 3 + 1));
        chunk.tiles[tile].field.celldx  = calloc(sizeof(double), -(chunk.tiles[tile].t_xmin - 2) + (chunk.tiles[tile].t_xmax + 2 + 1));
        chunk.tiles[tile].field.celldy  = calloc(sizeof(double), -(chunk.tiles[tile].t_ymin - 2) + (chunk.tiles[tile].t_ymax + 2 + 1));
        chunk.tiles[tile].field.vertexdx = calloc(sizeof(double), -(chunk.tiles[tile].t_xmin - 2) + (chunk.tiles[tile].t_xmax + 3 + 1));
        chunk.tiles[tile].field.vertexdy = calloc(sizeof(double), -(chunk.tiles[tile].t_ymin - 2) + (chunk.tiles[tile].t_ymax + 3 + 1));
        chunk.tiles[tile].field.volume  = calloc(sizeof(double), (-(chunk.tiles[tile].t_xmin - 2) + chunk.tiles[tile].t_xmax + 2 + 1) * (-(chunk.tiles[tile].t_ymin - 2) + (chunk.tiles[tile].t_ymax + 2) + 1));
        chunk.tiles[tile].field.xarea   = calloc(sizeof(double), (-(chunk.tiles[tile].t_xmin - 2) + chunk.tiles[tile].t_xmax + 3 + 1) * (-(chunk.tiles[tile].t_ymin - 2) + (chunk.tiles[tile].t_ymax + 2) + 1));
        chunk.tiles[tile].field.yarea   = calloc(sizeof(double), (-(chunk.tiles[tile].t_xmin - 2) + chunk.tiles[tile].t_xmax + 2 + 1) * (-(chunk.tiles[tile].t_ymin - 2) + (chunk.tiles[tile].t_ymax + 3) + 1));


        // TODO first touch?
    }
}
