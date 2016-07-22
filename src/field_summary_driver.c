#include "kernels/field_summary_kernel_c.c"
#include "definitions_c.h"

void field_summary_driver(
    double* vol,
    double* ie,
    double* ke,
    double* mass,
    double* press)
{
    double t_vol = 0.0;
    double t_mass = 0.0;
    double t_ie = 0.0;
    double t_ke = 0.0;
    double t_press = 0.0;

    for (int tile = 0; tile < tiles_per_chunk; tile++) {
        field_summary_kernel_c_(
            &chunk.tiles[tile].t_xmin,
            &chunk.tiles[tile].t_xmax,
            &chunk.tiles[tile].t_ymin,
            &chunk.tiles[tile].t_ymax,
            chunk.tiles[tile].field.volume,
            chunk.tiles[tile].field.density0,
            chunk.tiles[tile].field.energy0,
            chunk.tiles[tile].field.pressure,
            chunk.tiles[tile].field.xvel0,
            chunk.tiles[tile].field.yvel0,
            vol, mass, ie, ke, press);
        t_vol   += *vol;
        t_mass  += *mass;
        t_ie    += *ie;
        t_ke    += *ke;
        t_press += *press;
    }

    *vol = t_vol;
    *ie = t_ie;
    *ke = t_ke;
    *mass = t_mass;
    *press = t_press;
}