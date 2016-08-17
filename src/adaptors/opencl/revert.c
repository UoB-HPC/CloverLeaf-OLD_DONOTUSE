
void kernel revert_kernel(
    int x_min, int x_max,
    int y_min, int y_max,
    const_field_2d_t density0,
    field_2d_t density1,
    const_field_2d_t energy0,
    field_2d_t energy1)
{
    int k = get_global_id(1) + y_min;
    int j = get_global_id(0) + x_min;

    if (j <= x_max && k <= y_max)
        revert_kernel_c_(
            j, k,
            x_min, x_max, y_min, y_max,
            density0,
            density1,
            energy0,
            energy1);
}
