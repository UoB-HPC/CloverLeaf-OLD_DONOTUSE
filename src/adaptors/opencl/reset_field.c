
void kernel reset_field_kernel(
    int x_min, int x_max,
    int y_min, int y_max,
    field_2d_t       density0,
    const_field_2d_t density1,
    field_2d_t       energy0,
    const_field_2d_t energy1,
    field_2d_t       xvel0,
    const_field_2d_t xvel1,
    field_2d_t       yvel0,
    const_field_2d_t yvel1)
{
    int k = get_global_id(1) + y_min;
    int j = get_global_id(0) + x_min;

    if (j <= x_max + 1 && k <= y_max + 1)
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
