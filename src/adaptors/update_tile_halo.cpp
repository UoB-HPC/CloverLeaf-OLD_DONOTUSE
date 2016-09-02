
#if defined(USE_OPENMP) || defined(USE_OMPSS)
#include "../definitions_c.h"
#include "../kernels/update_tile_halo_kernel.c"


void update_tile_halo(int* fields, int depth)
{
    int t_left, t_right, t_up, t_down;

    for (int tile = 0; tile < tiles_per_chunk; tile++) {
        t_up = chunk.tiles[tile].tile_neighbours[TILE_TOP];
        t_down = chunk.tiles[tile].tile_neighbours[TILE_BOTTOM];

        t_left = chunk.tiles[tile].tile_neighbours[TILE_LEFT];
        t_right = chunk.tiles[tile].tile_neighbours[TILE_RIGHT];

        if (t_up != EXTERNAL_TILE) {
            update_tile_halo_t_kernel_c_(
                &chunk.tiles[tile].t_xmin,
                &chunk.tiles[tile].t_xmax,
                &chunk.tiles[tile].t_ymin,
                &chunk.tiles[tile].t_ymax,
                chunk.tiles[tile].field.density0,
                chunk.tiles[tile].field.energy0,
                chunk.tiles[tile].field.pressure,
                chunk.tiles[tile].field.viscosity,
                chunk.tiles[tile].field.soundspeed,
                chunk.tiles[tile].field.density1,
                chunk.tiles[tile].field.energy1,
                chunk.tiles[tile].field.xvel0,
                chunk.tiles[tile].field.yvel0,
                chunk.tiles[tile].field.xvel1,
                chunk.tiles[tile].field.yvel1,
                chunk.tiles[tile].field.vol_flux_x,
                chunk.tiles[tile].field.vol_flux_y,
                chunk.tiles[tile].field.mass_flux_x,
                chunk.tiles[tile].field.mass_flux_y,
                &chunk.tiles[t_up - 1].t_xmin,
                &chunk.tiles[t_up - 1].t_xmax,
                &chunk.tiles[t_up - 1].t_ymin,
                &chunk.tiles[t_up - 1].t_ymax,
                chunk.tiles[t_up - 1].field.density0,
                chunk.tiles[t_up - 1].field.energy0,
                chunk.tiles[t_up - 1].field.pressure,
                chunk.tiles[t_up - 1].field.viscosity,
                chunk.tiles[t_up - 1].field.soundspeed,
                chunk.tiles[t_up - 1].field.density1,
                chunk.tiles[t_up - 1].field.energy1,
                chunk.tiles[t_up - 1].field.xvel0,
                chunk.tiles[t_up - 1].field.yvel0,
                chunk.tiles[t_up - 1].field.xvel1,
                chunk.tiles[t_up - 1].field.yvel1,
                chunk.tiles[t_up - 1].field.vol_flux_x,
                chunk.tiles[t_up - 1].field.vol_flux_y,
                chunk.tiles[t_up - 1].field.mass_flux_x,
                chunk.tiles[t_up - 1].field.mass_flux_y,
                fields,
                &depth);
        }
        if (t_down != EXTERNAL_TILE) {
            update_tile_halo_b_kernel_c_(
                &chunk.tiles[tile].t_xmin,
                &chunk.tiles[tile].t_xmax,
                &chunk.tiles[tile].t_ymin,
                &chunk.tiles[tile].t_ymax,
                chunk.tiles[tile].field.density0,
                chunk.tiles[tile].field.energy0,
                chunk.tiles[tile].field.pressure,
                chunk.tiles[tile].field.viscosity,
                chunk.tiles[tile].field.soundspeed,
                chunk.tiles[tile].field.density1,
                chunk.tiles[tile].field.energy1,
                chunk.tiles[tile].field.xvel0,
                chunk.tiles[tile].field.yvel0,
                chunk.tiles[tile].field.xvel1,
                chunk.tiles[tile].field.yvel1,
                chunk.tiles[tile].field.vol_flux_x,
                chunk.tiles[tile].field.vol_flux_y,
                chunk.tiles[tile].field.mass_flux_x,
                chunk.tiles[tile].field.mass_flux_y,
                &chunk.tiles[t_down - 1].t_xmin,
                &chunk.tiles[t_down - 1].t_xmax,
                &chunk.tiles[t_down - 1].t_ymin,
                &chunk.tiles[t_down - 1].t_ymax,
                chunk.tiles[t_down - 1].field.density0,
                chunk.tiles[t_down - 1].field.energy0,
                chunk.tiles[t_down - 1].field.pressure,
                chunk.tiles[t_down - 1].field.viscosity,
                chunk.tiles[t_down - 1].field.soundspeed,
                chunk.tiles[t_down - 1].field.density1,
                chunk.tiles[t_down - 1].field.energy1,
                chunk.tiles[t_down - 1].field.xvel0,
                chunk.tiles[t_down - 1].field.yvel0,
                chunk.tiles[t_down - 1].field.xvel1,
                chunk.tiles[t_down - 1].field.yvel1,
                chunk.tiles[t_down - 1].field.vol_flux_x,
                chunk.tiles[t_down - 1].field.vol_flux_y,
                chunk.tiles[t_down - 1].field.mass_flux_x,
                chunk.tiles[t_down - 1].field.mass_flux_y,
                fields,
                &depth);
        }

        if (t_left != EXTERNAL_TILE) {
            update_tile_halo_l_kernel_c_(
                chunk.tiles[tile].t_xmin,
                chunk.tiles[tile].t_xmax,
                chunk.tiles[tile].t_ymin,
                chunk.tiles[tile].t_ymax,
                chunk.tiles[tile].field.density0,
                chunk.tiles[tile].field.energy0,
                chunk.tiles[tile].field.pressure,
                chunk.tiles[tile].field.viscosity,
                chunk.tiles[tile].field.soundspeed,
                chunk.tiles[tile].field.density1,
                chunk.tiles[tile].field.energy1,
                chunk.tiles[tile].field.xvel0,
                chunk.tiles[tile].field.yvel0,
                chunk.tiles[tile].field.xvel1,
                chunk.tiles[tile].field.yvel1,
                chunk.tiles[tile].field.vol_flux_x,
                chunk.tiles[tile].field.vol_flux_y,
                chunk.tiles[tile].field.mass_flux_x,
                chunk.tiles[tile].field.mass_flux_y,
                chunk.tiles[t_left - 1].t_xmin,
                chunk.tiles[t_left - 1].t_xmax,
                chunk.tiles[t_left - 1].t_ymin,
                chunk.tiles[t_left - 1].t_ymax,
                chunk.tiles[t_left - 1].field.density0,
                chunk.tiles[t_left - 1].field.energy0,
                chunk.tiles[t_left - 1].field.pressure,
                chunk.tiles[t_left - 1].field.viscosity,
                chunk.tiles[t_left - 1].field.soundspeed,
                chunk.tiles[t_left - 1].field.density1,
                chunk.tiles[t_left - 1].field.energy1,
                chunk.tiles[t_left - 1].field.xvel0,
                chunk.tiles[t_left - 1].field.yvel0,
                chunk.tiles[t_left - 1].field.xvel1,
                chunk.tiles[t_left - 1].field.yvel1,
                chunk.tiles[t_left - 1].field.vol_flux_x,
                chunk.tiles[t_left - 1].field.vol_flux_y,
                chunk.tiles[t_left - 1].field.mass_flux_x,
                chunk.tiles[t_left - 1].field.mass_flux_y,
                fields,
                &depth);
        }
        if (t_right != EXTERNAL_TILE) {
            update_tile_halo_r_kernel_c_(
                &chunk.tiles[tile].t_xmin,
                &chunk.tiles[tile].t_xmax,
                &chunk.tiles[tile].t_ymin,
                &chunk.tiles[tile].t_ymax,
                chunk.tiles[tile].field.density0,
                chunk.tiles[tile].field.energy0,
                chunk.tiles[tile].field.pressure,
                chunk.tiles[tile].field.viscosity,
                chunk.tiles[tile].field.soundspeed,
                chunk.tiles[tile].field.density1,
                chunk.tiles[tile].field.energy1,
                chunk.tiles[tile].field.xvel0,
                chunk.tiles[tile].field.yvel0,
                chunk.tiles[tile].field.xvel1,
                chunk.tiles[tile].field.yvel1,
                chunk.tiles[tile].field.vol_flux_x,
                chunk.tiles[tile].field.vol_flux_y,
                chunk.tiles[tile].field.mass_flux_x,
                chunk.tiles[tile].field.mass_flux_y,
                &chunk.tiles[t_right - 1].t_xmin,
                &chunk.tiles[t_right - 1].t_xmax,
                &chunk.tiles[t_right - 1].t_ymin,
                &chunk.tiles[t_right - 1].t_ymax,
                chunk.tiles[t_right - 1].field.density0,
                chunk.tiles[t_right - 1].field.energy0,
                chunk.tiles[t_right - 1].field.pressure,
                chunk.tiles[t_right - 1].field.viscosity,
                chunk.tiles[t_right - 1].field.soundspeed,
                chunk.tiles[t_right - 1].field.density1,
                chunk.tiles[t_right - 1].field.energy1,
                chunk.tiles[t_right - 1].field.xvel0,
                chunk.tiles[t_right - 1].field.yvel0,
                chunk.tiles[t_right - 1].field.xvel1,
                chunk.tiles[t_right - 1].field.yvel1,
                chunk.tiles[t_right - 1].field.vol_flux_x,
                chunk.tiles[t_right - 1].field.vol_flux_y,
                chunk.tiles[t_right - 1].field.mass_flux_x,
                chunk.tiles[t_right - 1].field.mass_flux_y,
                fields,
                &depth);
        }
    }
}

#endif

#if defined(USE_KOKKOS) || defined(USE_OPENCL) || defined(USE_CUDA)

void update_tile_halo(int* fields, int depth)
{
    // TODO
}

#endif