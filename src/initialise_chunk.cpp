#include "initialise_chunk.h"
#include "kernels/initialise_chunk_kernel_c.c"
#include "definitions_c.h"

void initialise_chunk(int tile)
{
    double xmin, ymin, dx, dy;

    dx = (grid.xmax - grid.xmin) / (float)grid.x_cells;
    dy = (grid.ymax - grid.ymin) / (float)grid.y_cells;

    xmin = grid.xmin + dx * (float)(chunk.tiles[tile].t_left - 1);
    ymin = grid.ymin + dx * (float)(chunk.tiles[tile].t_bottom - 1);

    initialise_chunk_kernel_c_(
        chunk.tiles[tile].t_xmin,
        chunk.tiles[tile].t_xmax,
        chunk.tiles[tile].t_ymin,
        chunk.tiles[tile].t_ymax,
        xmin, ymin, dx, dy,
        chunk.tiles[tile].field.vertexx,
        chunk.tiles[tile].field.vertexdx,
        chunk.tiles[tile].field.vertexy,
        chunk.tiles[tile].field.vertexdy,
        chunk.tiles[tile].field.cellx,
        chunk.tiles[tile].field.celldx,
        chunk.tiles[tile].field.celly,
        chunk.tiles[tile].field.celldy,
        chunk.tiles[tile].field.volume,
        chunk.tiles[tile].field.xarea,
        chunk.tiles[tile].field.yarea);
}
