
void kernel update_halo_1_kernel(
    int x_min, int x_max,
    int y_min, int y_max,
    constant int* chunk_neighbours,
    constant int* tile_neighbours,
    field_2d_t density0,
    field_2d_t density1,
    field_2d_t energy0,
    field_2d_t energy1,
    field_2d_t pressure,
    field_2d_t viscosity,
    field_2d_t soundspeed,
    field_2d_t xvel0,
    field_2d_t yvel0,
    field_2d_t xvel1,
    field_2d_t yvel1,
    field_2d_t vol_flux_x,
    field_2d_t mass_flux_x,
    field_2d_t vol_flux_y,
    field_2d_t mass_flux_y,
    constant int* fields,
    int depth)
{
    int j = get_global_id(0) + (x_min - depth);
    int k = get_global_id(1) + 1;

    if (j <= x_max + depth && k <= depth)
        update_halo_kernel_1(
            j, k,
            x_min,
            x_max,
            y_min,
            y_max,
            chunk_neighbours,
            tile_neighbours,
            density0,
            density1,
            energy0,
            energy1,
            pressure,
            viscosity,
            soundspeed,
            xvel0,
            yvel0,
            xvel1,
            yvel1,
            vol_flux_x,
            mass_flux_x,
            vol_flux_y,
            mass_flux_y,
            fields,
            depth);
}

void kernel update_halo_2_kernel(
    int x_min, int x_max,
    int y_min, int y_max,
    constant int* chunk_neighbours,
    constant int* tile_neighbours,
    field_2d_t density0,
    field_2d_t density1,
    field_2d_t energy0,
    field_2d_t energy1,
    field_2d_t pressure,
    field_2d_t viscosity,
    field_2d_t soundspeed,
    field_2d_t xvel0,
    field_2d_t yvel0,
    field_2d_t xvel1,
    field_2d_t yvel1,
    field_2d_t vol_flux_x,
    field_2d_t mass_flux_x,
    field_2d_t vol_flux_y,
    field_2d_t mass_flux_y,
    constant int* fields,
    int depth)
{
    int j = get_global_id(0) + 1;
    int k = get_global_id(1) + (y_min - depth);

    if (j <= depth && k <= y_max + depth)
        update_halo_kernel_2(
            j, k,
            x_min,
            x_max,
            y_min,
            y_max,
            chunk_neighbours,
            tile_neighbours,
            density0,
            density1,
            energy0,
            energy1,
            pressure,
            viscosity,
            soundspeed,
            xvel0,
            yvel0,
            xvel1,
            yvel1,
            vol_flux_x,
            mass_flux_x,
            vol_flux_y,
            mass_flux_y,
            fields,
            depth);
}
