
void kernel update_halo_1_kernel(
    int x_min, int x_max,
    int y_min, int y_max,
    constant int* chunk_neighbours,
    constant int* tile_neighbours,
    global double* density0,
    global double* density1,
    global double* energy0,
    global double* energy1,
    global double* pressure,
    global double* viscosity,
    global double* soundspeed,
    global double* xvel0,
    global double* yvel0,
    global double* xvel1,
    global double* yvel1,
    global double* vol_flux_x,
    global double* mass_flux_x,
    global double* vol_flux_y,
    global double* mass_flux_y,
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
    global double* density0,
    global double* density1,
    global double* energy0,
    global double* energy1,
    global double* pressure,
    global double* viscosity,
    global double* soundspeed,
    global double* xvel0,
    global double* yvel0,
    global double* xvel1,
    global double* yvel1,
    global double* vol_flux_x,
    global double* mass_flux_x,
    global double* vol_flux_y,
    global double* mass_flux_y,
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
