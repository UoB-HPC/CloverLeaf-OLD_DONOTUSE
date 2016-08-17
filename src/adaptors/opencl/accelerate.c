

void kernel accelerate_kernel(
    int x_min, int x_max,
    int y_min, int y_max,
    const_field_2d_t xarea,
    const_field_2d_t yarea,
    const_field_2d_t volume,
    const_field_2d_t density0 ,
    const_field_2d_t pressure ,
    const_field_2d_t viscosity,
    field_2d_t xvel0,
    field_2d_t yvel0,
    field_2d_t xvel1,
    field_2d_t yvel1,
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