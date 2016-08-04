

void kernel flux_calc_x_kernel_(
    int x_min, int x_max,
    int y_min, int y_max,
    double dt,
    const global double* xarea,
    const global double* xvel0,
    const global double* xvel1,
    global double* vol_flux_x)
{
    int k = get_global_id(1) + y_min;
    int j = get_global_id(0) + x_min;

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
    const global double* yarea,
    const global double* yvel0,
    const global double* yvel1,
    global double* vol_flux_y)
{
    int k = get_global_id(1) + y_min;
    int j = get_global_id(0) + x_min;

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