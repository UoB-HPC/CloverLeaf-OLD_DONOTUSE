

void kernel ms1_kernel(
    int x_min, int x_max,
    int y_min, int y_max,
    field_2d_t       pre_vol,
    field_2d_t       post_vol ,
    const_field_2d_t volume ,
    const_field_2d_t vol_flux_x ,
    const_field_2d_t vol_flux_y)
{
    int k = get_global_id(1) + y_min - 2;
    int j = get_global_id(0) + x_min - 2;

    if (j <= x_max + 2 && k <= y_max + 2)
        ms1(
            j, k,
            x_min, x_max, y_min, y_max,
            pre_vol,
            post_vol,
            volume,
            vol_flux_x,
            vol_flux_y);
}


void kernel ms2_kernel(
    int x_min, int x_max,
    int y_min, int y_max,
    field_2d_t       pre_vol,
    field_2d_t       post_vol ,
    const_field_2d_t volume ,
    const_field_2d_t vol_flux_x ,
    const_field_2d_t vol_flux_y)
{
    int k = get_global_id(1) + y_min - 2;
    int j = get_global_id(0) + x_min - 2;

    if (j <= x_max + 2 && k <= y_max + 2)
        ms2(
            j, k,
            x_min, x_max, y_min, y_max,
            pre_vol,
            post_vol,
            volume,
            vol_flux_x,
            vol_flux_y);
}

void kernel ms3_kernel(
    int x_min, int x_max,
    int y_min, int y_max,
    field_2d_t       pre_vol,
    field_2d_t       post_vol ,
    const_field_2d_t volume ,
    const_field_2d_t vol_flux_x ,
    const_field_2d_t vol_flux_y)
{
    int k = get_global_id(1) + y_min - 2;
    int j = get_global_id(0) + x_min - 2;

    if (j <= x_max + 2 && k <= y_max + 2)
        ms3(
            j, k,
            x_min, x_max, y_min, y_max,
            pre_vol,
            post_vol,
            volume,
            vol_flux_x,
            vol_flux_y);
}

void kernel ms4_kernel(
    int x_min, int x_max,
    int y_min, int y_max,
    field_2d_t       pre_vol,
    field_2d_t       post_vol ,
    const_field_2d_t volume ,
    const_field_2d_t vol_flux_x ,
    const_field_2d_t vol_flux_y)
{
    int k = get_global_id(1) + y_min - 2;
    int j = get_global_id(0) + x_min - 2;

    if (j <= x_max + 2 && k <= y_max + 2)
        ms4(
            j, k,
            x_min, x_max, y_min, y_max,
            pre_vol,
            post_vol,
            volume,
            vol_flux_x,
            vol_flux_y);
}

void kernel dx1_kernel(
    int x_min, int x_max,
    int y_min, int y_max,
    field_2d_t node_flux,
    const_field_2d_t mass_flux_x)
{
    int k = get_global_id(1) + y_min;
    int j = get_global_id(0) + x_min - 2;

    if (j <= x_max + 2 && k <= y_max + 1)
        dx1(
            j, k,
            x_min, x_max, y_min, y_max,
            node_flux,
            mass_flux_x);
}

void kernel dy1_kernel(
    int x_min, int x_max,
    int y_min, int y_max,
    field_2d_t node_flux,
    const_field_2d_t mass_flux_y)
{
    int k = get_global_id(1) + y_min - 2;
    int j = get_global_id(0) + x_min;

    if (j <= x_max + 1 && k <= y_max + 2)
        dy1(
            j, k,
            x_min, x_max, y_min, y_max,
            node_flux,
            mass_flux_y);
}

void kernel dx2_kernel(
    int x_min, int x_max,
    int y_min, int y_max,
    field_2d_t node_mass_post,
    field_2d_t node_mass_pre,
    const_field_2d_t density1,
    const_field_2d_t post_vol,
    const_field_2d_t node_flux)
{
    int k = get_global_id(1) + y_min;
    int j = get_global_id(0) + x_min - 1;

    if (j <= x_max + 1 && k <= y_max + 1)
        dx2(
            j, k,
            x_min, x_max, y_min, y_max,
            node_mass_post,
            node_mass_pre,
            density1,
            post_vol,
            node_flux);
}

void kernel dy2_kernel(
    int x_min, int x_max,
    int y_min, int y_max,
    field_2d_t node_mass_post,
    field_2d_t node_mass_pre,
    const_field_2d_t density1,
    const_field_2d_t post_vol,
    const_field_2d_t node_flux)
{
    int k = get_global_id(1) + y_min - 1;
    int j = get_global_id(0) + x_min;

    if (j <= x_max + 1 && k <= y_max + 1)
        dy2(
            j, k,
            x_min, x_max, y_min, y_max,
            node_mass_post,
            node_mass_pre,
            density1,
            post_vol,
            node_flux);
}

void kernel dx3_kernel(
    int x_min, int x_max, int y_min, int y_max,
    field_2d_t mom_flux,
    const_field_2d_t node_flux,
    const_field_2d_t node_mass_pre,
    const_field_2d_t celldx,
    const_field_2d_t vel1)
{
    int k = get_global_id(1) + y_min;
    int j = get_global_id(0) + x_min - 1;

    if (j <= x_max + 1 && k <= y_max + 1)
        dx3(
            j, k,
            x_min, x_max, y_min, y_max,
            mom_flux,
            node_flux,
            node_mass_pre,
            celldx,
            vel1);
}

void kernel dy3_kernel(
    int x_min, int x_max, int y_min, int y_max,
    field_2d_t mom_flux,
    const_field_2d_t node_flux,
    const_field_2d_t node_mass_pre,
    const_field_2d_t celldy,
    const_field_2d_t vel1)
{
    int k = get_global_id(1) + y_min - 1;
    int j = get_global_id(0) + x_min;

    if (j <= x_max + 1 && k <= y_max + 1)
        dy3(
            j, k,
            x_min, x_max, y_min, y_max,
            mom_flux,
            node_flux,
            node_mass_pre,
            celldy,
            vel1);
}

void kernel dx4_kernel(
    int x_min, int x_max, int y_min, int y_max,
    field_2d_t vel1,
    const_field_2d_t node_mass_pre,
    const_field_2d_t mom_flux,
    const_field_2d_t node_mass_post)
{
    int k = get_global_id(1) + y_min;
    int j = get_global_id(0) + x_min;

    if (j <= x_max && k <= y_max)
        dx4(
            j, k,
            x_min, x_max, y_min, y_max,
            vel1,
            node_mass_pre,
            mom_flux,
            node_mass_post);
}

void kernel dy4_kernel(
    int x_min, int x_max, int y_min, int y_max,
    field_2d_t vel1,
    const_field_2d_t node_mass_pre,
    const_field_2d_t mom_flux,
    const_field_2d_t node_mass_post)
{
    int k = get_global_id(1) + y_min;
    int j = get_global_id(0) + x_min;

    if (j <= x_max  && k <= y_max)
        dy4(
            j, k,
            x_min, x_max, y_min, y_max,
            vel1,
            node_mass_pre,
            mom_flux,
            node_mass_post);
}
