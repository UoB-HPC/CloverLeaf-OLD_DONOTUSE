

void kernel pdv_kernel(
    int x_min, int x_max,
    int y_min, int y_max,
    double dt,
    const_field_2d_t xarea,
    const_field_2d_t yarea,
    const_field_2d_t volume,
    const_field_2d_t density0,
    field_2d_t       density1,
    const_field_2d_t energy0,
    field_2d_t       energy1,
    const_field_2d_t pressure,
    const_field_2d_t viscosity,
    const_field_2d_t xvel0,
    const_field_2d_t xvel1,
    const_field_2d_t yvel0,
    const_field_2d_t yvel1,
    field_2d_t       volume_change,
    int predict)
{
    int k = get_global_id(1) + y_min;
    int j = get_global_id(0) + x_min;

    if (j <= x_max && k <= y_max) {
        if (predict == 0) {
            pdv_kernel_predict_c_(
                j, k,
                x_min, x_max, y_min, y_max,
                dt,
                xarea,
                yarea,
                volume,
                density0,
                density1,
                energy0,
                energy1,
                pressure,
                viscosity,
                xvel0,
                xvel1,
                yvel0,
                yvel1,
                volume_change);
        } else {
            pdv_kernel_no_predict_c_(
                j, k,
                x_min, x_max, y_min, y_max,
                dt,
                xarea,
                yarea,
                volume,
                density0,
                density1,
                energy0,
                energy1,
                pressure,
                viscosity,
                xvel0,
                xvel1,
                yvel0,
                yvel1,
                volume_change);
        }
    }
}