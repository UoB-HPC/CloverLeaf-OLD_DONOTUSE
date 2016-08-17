

void kernel viscosity_kernel(
    int x_min, int x_max,
    int y_min, int y_max,
    const global double* celldx,
    const global double* celldy,
    const global double* density0,
    const global double* pressure,
    global double*       viscosity,
    const global double* xvel0,
    const global double* yvel0)
{
    int k = get_global_id(1) + y_min;
    int j = get_global_id(0) + x_min;

    if (j <= x_max && k <= y_max)
        viscosity_kernel_c_(
            j, k,
            x_min, x_max,
            y_min, y_max,
            celldx,
            celldy,
            density0,
            pressure,
            viscosity,
            xvel0,
            yvel0);
}
