
void kernel field_summary_kernel(
    int x_min, int x_max,
    int y_min, int y_max,
    const_field_2d_t volume,
    const_field_2d_t density0,
    const_field_2d_t energy0,
    const_field_2d_t pressure,
    const_field_2d_t xvel0,
    const_field_2d_t yvel0,
    field_2d_t g_vol,
    field_2d_t g_mass,
    field_2d_t g_ie,
    field_2d_t g_ke,
    field_2d_t g_press,
    local double* t_vol,
    local double* t_mass,
    local double* t_ie,
    local double* t_ke,
    local double* t_press)
{
    int k = get_global_id(1) + y_min;
    int j = get_global_id(0) + x_min;

    int x = get_local_id(0),
        y = get_local_id(1);
    int lid = x + get_local_size(0) * y;
    int gid = get_group_id(0) + get_num_groups(0) * get_group_id(1);

    int lsize = get_local_size(0) * get_local_size(1);

    t_vol[lid] =
        t_mass[lid] =
            t_ie[lid] =
                t_ke[lid] =
                    t_press[lid] =
                        0.0;

    if (j <= x_max && k <= y_max) {
        field_summary_kernel_(
            j, k,
            x_min, x_max,
            y_min, y_max,
            volume,
            density0, energy0,
            pressure,
            xvel0, yvel0,
            &t_vol[lid],
            &t_mass[lid],
            &t_ie[lid],
            &t_ke[lid],
            &t_press[lid]);
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = lsize / 2; s > 0; s /= 2) {
        if (lid < s) {
            t_vol[lid]   += t_vol[lid + s];
            t_mass[lid]  += t_mass[lid + s];
            t_ie[lid]    += t_ie[lid + s];
            t_ke[lid]    += t_ke[lid + s];
            t_press[lid] += t_press[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        g_vol[gid]   = t_vol[0];
        g_mass[gid]  = t_mass[0];
        g_ie[gid]    = t_ie[0];
        g_ke[gid]    = t_ke[0];
        g_press[gid] = t_press[0];
    }
}