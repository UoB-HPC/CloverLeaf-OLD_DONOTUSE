

void kernel ideal_gas_kernel(
    int x_min, int x_max,
    int y_min, int y_max,
    const_field_2d_t density,
    const_field_2d_t energy,
    field_2d_t pressure,
    field_2d_t soundspeed)
{
    int k = get_global_id(1) + y_min;
    int j = get_global_id(0) + x_min;

    if (j <= x_max && k <= y_max)
        ideal_gas_kernel_c_(
            j, k,
            x_min, x_max,
            y_min, y_max,
            density,
            energy,
            pressure,
            soundspeed);
}