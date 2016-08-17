

void kernel pdv_kernel(
    int x_min, int x_max,
    int y_min, int y_max,
    double dt,
    const global double* xarea,
    const global double* yarea,
    const global double* volume,
    const global double* density0,
    global double*       density1,
    const global double* energy0,
    global double*       energy1,
    const global double* pressure,
    const global double* viscosity,
    const global double* xvel0,
    const global double* xvel1,
    const global double* yvel0,
    const global double* yvel1,
    global double*       volume_change,
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