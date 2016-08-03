

void kernel ideal_gas_kernel(
    int x_min, int x_max,
    int y_min, int y_max,
    const global double* density,
    const global double* energy,
    global double* pressure,
    global double* soundspeed)
{
    int k = get_global_id(1) + y_min;
    int j = get_global_id(0) + x_min;

    ideal_gas_kernel_c_(
        j, k,
        x_min, x_max,
        y_min, y_max,
        density,
        energy,
        pressure,
        soundspeed);
}