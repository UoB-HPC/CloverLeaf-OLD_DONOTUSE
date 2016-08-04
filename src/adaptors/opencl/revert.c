
void kernel revert_kernel(
    int x_min, int x_max,
    int y_min, int y_max,
    global double* density0,
    global double* density1,
    global double* energy0,
    global double* energy1)
{
    int k = get_global_id(1) + y_min;
    int j = get_global_id(0) + x_min;

    revert_kernel_c_(
        j, k,
        x_min, x_max, y_min, y_max,
        density0,
        density1,
        energy0,
        energy1);
}
