#include "revert.h"
#include "definitions_c.h"
#include "revert_kernel_c.c"


void revert()
{
    for (int tile = 0; tile < tiles_per_chunk; tile++) {
        revert_kernel_c_(
            &chunk.tiles[tile].t_xmin,
            &chunk.tiles[tile].t_xmax,
            &chunk.tiles[tile].t_ymin,
            &chunk.tiles[tile].t_ymax,
            chunk.tiles[tile].field.density0,
            chunk.tiles[tile].field.density1,
            chunk.tiles[tile].field.energy0,
            chunk.tiles[tile].field.energy1);
    }
}
