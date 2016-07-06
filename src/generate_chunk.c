#include "generate_chunk.h"
#include "generate_chunk_kernel_c.c"
#include "definitions_c.h"

void generate_chunk(int tile)
{
    double *state_density = (double*)malloc(sizeof(double) * number_of_states),
            *state_energy = (double*)malloc(sizeof(double) * number_of_states),
             *state_xvel = (double*)malloc(sizeof(double) * number_of_states),
              *state_yvel = (double*)malloc(sizeof(double) * number_of_states),
               *state_xmin = (double*)malloc(sizeof(double) * number_of_states),
                *state_xmax = (double*)malloc(sizeof(double) * number_of_states),
                 *state_ymin = (double*)malloc(sizeof(double) * number_of_states),
                  *state_ymax = (double*)malloc(sizeof(double) * number_of_states),
                   *state_radius = (double*)malloc(sizeof(double) * number_of_states);
    int *state_geometry = (int*)malloc(sizeof(int) * number_of_states);

    for (int state = 0; state < number_of_states; state++) {
        state_density[state] = states[state].density;
        state_energy[state] = states[state].energy;
        state_xvel[state] = states[state].xvel;
        state_yvel[state] = states[state].yvel;
        state_xmin[state] = states[state].xmin;
        state_xmax[state] = states[state].xmax;
        state_ymin[state] = states[state].ymin;
        state_ymax[state] = states[state].ymax;
        state_radius[state] = states[state].radius;
        state_geometry[state] = states[state].geometry;
    }

    generate_chunk_kernel_c_(
        &chunk.tiles[tile].t_xmin,
        &chunk.tiles[tile].t_xmax,
        &chunk.tiles[tile].t_ymin,
        &chunk.tiles[tile].t_ymax,
        chunk.tiles[tile].field.vertexx,
        chunk.tiles[tile].field.vertexy,
        chunk.tiles[tile].field.cellx,
        chunk.tiles[tile].field.celly,
        chunk.tiles[tile].field.density0,
        chunk.tiles[tile].field.energy0,
        chunk.tiles[tile].field.xvel0,
        chunk.tiles[tile].field.yvel0,
        &number_of_states,
        state_density,
        state_energy,
        state_xvel,
        state_yvel,
        state_xmin,
        state_xmax,
        state_ymin,
        state_ymax,
        state_radius,
        state_geometry,
        &g_rect,
        &g_circ,
        &g_point);
}
