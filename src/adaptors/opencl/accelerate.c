

void kernel accelerate_kernel(
    int x_min, int x_max,
    int y_min, int y_max,
    const global double* xarea,
    const global double* yarea,
    const global double* volume,
    const global double* density0 ,
    const global double* pressure ,
    const global double* viscosity,
    global double* xvel0,
    global double* yvel0,
    global double* xvel1,
    global double* yvel1,
    double dt)
{
    int j = get_global_id(0) + x_min;
    int k = get_global_id(1) + y_min;

    if (j <= x_max && k <= y_max)
        accelerate_kernel_c_(
            j, k,
            x_min, x_max,
            y_min, y_max,
            xarea,
            yarea,
            volume,
            density0,
            pressure,
            viscosity,
            xvel0,
            yvel0,
            xvel1,
            yvel1,
            dt);
}