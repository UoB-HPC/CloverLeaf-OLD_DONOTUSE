#include "../definitions_c.h"
#include "../kernels/field_summary_kernel_c.c"

#if defined(USE_OPENMP) || defined(USE_OMPSS) || defined(USE_KOKKOS)
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

    for (int tilen = 0; tilen < tiles_per_chunk; tilen++) {
        struct tile_type tile = chunk.tiles[tilen];
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


        #pragma omp parallel for reduction(+:_vol,_mass,_ie,_ke,_press)
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
}
#endif

#if defined(USE_OPENCL)
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

    for (int tilen = 0; tilen < tiles_per_chunk; tilen++) {
        struct tile_type tile = chunk.tiles[tilen];
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

        openclQueue.enqueueMapBuffer(
            *tile.field.d_volume,
            CL_TRUE, CL_MAP_READ, 0,
            sizeof(double) * tile.field.volume_size);
        openclQueue.enqueueMapBuffer(
            *tile.field.d_density0,
            CL_TRUE, CL_MAP_READ, 0,
            sizeof(double) * tile.field.density0_size);
        openclQueue.enqueueMapBuffer(
            *tile.field.d_energy0,
            CL_TRUE, CL_MAP_READ, 0,
            sizeof(double) * tile.field.energy0_size);
        openclQueue.enqueueMapBuffer(
            *tile.field.d_pressure,
            CL_TRUE, CL_MAP_READ, 0,
            sizeof(double) * tile.field.pressure_size);
        openclQueue.enqueueMapBuffer(
            *tile.field.d_xvel0,
            CL_TRUE, CL_MAP_READ, 0,
            sizeof(double) * tile.field.xvel0_size);
        openclQueue.enqueueMapBuffer(
            *tile.field.d_yvel0,
            CL_TRUE, CL_MAP_READ, 0,
            sizeof(double) * tile.field.yvel0_size);

        double _vol   = 0.0,
               _mass  = 0.0,
               _ie    = 0.0,
               _ke    = 0.0,
               _press = 0.0;

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

        openclQueue.enqueueUnmapMemObject(
            *tile.field.d_volume,
            tile.field.volume);
        openclQueue.enqueueUnmapMemObject(
            *tile.field.d_density0,
            tile.field.density0);
        openclQueue.enqueueUnmapMemObject(
            *tile.field.d_energy0,
            tile.field.energy0);
        openclQueue.enqueueUnmapMemObject(
            *tile.field.d_pressure,
            tile.field.pressure);
        openclQueue.enqueueUnmapMemObject(
            *tile.field.d_xvel0,
            tile.field.xvel0);
        openclQueue.enqueueUnmapMemObject(
            *tile.field.d_yvel0,
            tile.field.yvel0);

        *vol   += _vol;
        *mass  += _mass;
        *ie    += _ie;
        *ke    += _ke;
        *press += _press;
    }
}
#endif
