#include "../definitions_c.h"
#include "../kernels/field_summary_kernel_c.c"

void field_summary_openmp(
    struct tile_type tile,
    double* vol,
    double* mass,
    double* ie,
    double* ke,
    double* press)
{
    int x_min = tile.t_xmin,
        x_max = tile.t_xmax,
        y_min = tile.t_ymin,
        y_max = tile.t_ymax;
    field_2d_t volume   = tile.field.volume;
    field_2d_t density0 = tile.field.density0;
    field_2d_t energy0  = tile.field.energy0;
    field_2d_t pressure = tile.field.pressure;
    field_2d_t xvel0    = tile.field.xvel0;
    field_2d_t yvel0    = tile.field.yvel0;

    double _vol   = 0.0,
           _mass  = 0.0,
           _ie    = 0.0,
           _ke    = 0.0,
           _press = 0.0;


    // #pragma omp parallel for reduction(+:_vol,_mass,_ie,_ke,_press)
    for (int k = y_min; k <= y_max; k++) {
        for (int j = x_min; j <= x_max; j++) {
            field_summary_kernel(
                j, k,
                x_min, x_max,
                y_min, y_max,
                volume,
                density0, energy0,
                pressure,
                xvel0, yvel0,
                &_vol, &_mass, &_ie, &_ke, &_press);
        }
    }


    *vol   += _vol;
    *mass  += _mass;
    *ie    += _ie;
    *ke    += _ke;
    *press += _press;
}

void field_summary(
    double* vol,
    double* ie,
    double* ke,
    double* mass,
    double* press)
{
    *vol = 0.0;
    *mass = 0.0;
    *ie = 0.0;
    *ke = 0.0;
    *press = 0.0;

#if defined(USE_KOKKOS)
    for (int tile = 0; tile < tiles_per_chunk; tile++) {
        field_summary_openmp(
            chunk.tiles[tile],
            vol,
            mass,
            ie,
            ke,
            press);
    }
#endif

#if defined(USE_OPENMP) || defined(USE_OMPSS)
    for (int tile = 0; tile < tiles_per_chunk; tile++) {
        field_summary_openmp(
            chunk.tiles[tile],
            vol,
            mass,
            ie,
            ke,
            press);
    }
#endif
}