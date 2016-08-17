

void kernel calc_dt_kernel(
    int x_min, int x_max,
    int y_min, int y_max,
    const global double* xarea,
    const global double* yarea,
    const global double* celldx,
    const global double* celldy,
    const global double* volume,
    const global double* density0,
    const global double* energy0 ,
    const global double* pressure,
    const global double* viscosity,
    const global double* soundspeed,
    const global double* xvel0,
    const global double* yvel0,
    global double* dtmin,
    local double* temp)
{
    int k = get_global_id(1) + y_min;
    int j = get_global_id(0) + x_min;

    int x = get_local_id(0),
        y = get_local_id(1);
    int lid = x + get_local_size(0) * y;
    int gid = get_group_id(0) + get_num_groups(0) * get_group_id(1);

    int lsize = get_local_size(0) * get_local_size(1);

    temp[lid] = 1000.0;
    if (j <= x_max && k <= y_max) {
        double val = calc_dt_kernel_c_(
                         j, k,
                         x_min, x_max,
                         y_min, y_max,
                         xarea,
                         yarea,
                         celldx,
                         celldy,
                         volume,
                         density0,
                         energy0 ,
                         pressure,
                         viscosity,
                         soundspeed,
                         xvel0,
                         yvel0,
                         dtmin);

        temp[lid] = val;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = lsize / 2; s > 0; s /= 2) {
        if (lid < s) {
            if (temp[lid + s] < temp[lid])
                temp[lid] = temp[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        dtmin[gid] = temp[0];
    }
}
