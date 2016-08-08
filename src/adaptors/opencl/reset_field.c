
void kernel reset_field_kernel(
    int x_min, int x_max,
    int y_min, int y_max,
    global double*       density0,
    const global double* density1,
    global double*       energy0,
    const global double* energy1,
    global double*       xvel0,
    const global double* xvel1,
    global double*       yvel0,
    const global double* yvel1)
{
    int k = get_global_id(1) + y_min;
    int j = get_global_id(0) + x_min;

    reset_field_kernel_c_(
        j, k,
        x_min, x_max, y_min, y_max,
        density0,
        density1,
        energy0,
        energy1,
        xvel0,
        xvel1,
        yvel0,
        yvel1);
}
