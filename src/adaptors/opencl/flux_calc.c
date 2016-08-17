

void kernel flux_calc_x_kernel_(
    int x_min, int x_max,
    int y_min, int y_max,
    double dt,
    const_field_2d_t xarea,
    const_field_2d_t xvel0,
    const_field_2d_t xvel1,
    field_2d_t vol_flux_x)
{
    int k = get_global_id(1) + y_min;
    int j = get_global_id(0) + x_min;

    if (j <= x_max && k <= y_max)
        flux_calc_x_kernel(
            j, k,
            x_min, x_max,
            y_min, y_max,
            dt,
            xarea,
            xvel0,
            xvel1,
            vol_flux_x);
}

void kernel flux_calc_y_kernel_(
    int x_min, int x_max,
    int y_min, int y_max,
    double dt,
    const_field_2d_t yarea,
    const_field_2d_t yvel0,
    const_field_2d_t yvel1,
    field_2d_t vol_flux_y)
{
    int k = get_global_id(1) + y_min;
    int j = get_global_id(0) + x_min;

    if (j <= x_max && k <= y_max)
        flux_calc_y_kernel(
            j, k,
            x_min, x_max,
            y_min, y_max,
            dt,
            yarea,
            yvel0,
            yvel1,
            vol_flux_y);
}