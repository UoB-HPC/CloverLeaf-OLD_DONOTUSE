
void kernel xsweep_kernel(
    int x_min, int x_max, int y_min, int y_max,
    global double* pre_vol,
    global double* post_vol,
    const global double* volume,
    const global double* vol_flux_x,
    const global double* vol_flux_y,
    int sweep_number)
{
    int k = get_global_id(1) + y_min - 2;
    int j = get_global_id(0) + x_min - 2;

    xsweep(
        j, k,
        x_min, x_max,
        y_min, y_max,
        pre_vol,
        post_vol,
        volume,
        vol_flux_x,
        vol_flux_y,
        sweep_number);
}

void kernel ysweep_kernel(
    int x_min, int x_max, int y_min, int y_max,
    global double* pre_vol,
    global double* post_vol,
    const global double* volume,
    const global double* vol_flux_x,
    const global double* vol_flux_y,
    int sweep_number)
{
    int k = get_global_id(1) + y_min - 2;
    int j = get_global_id(0) + x_min - 2;

    ysweep(
        j, k,
        x_min, x_max,
        y_min, y_max,
        pre_vol,
        post_vol,
        volume,
        vol_flux_x,
        vol_flux_y,
        sweep_number);
}


void kernel xcomp1_kernel(
    int x_min, int x_max, int y_min, int y_max,
    global double* mass_flux_x,
    global double* ener_flux,
    const global double* vol_flux_x,
    const global double* pre_vol,
    const global double* density1,
    const global double* energy1,
    const global double* vertexdx)
{
    int k = get_global_id(1) + y_min;
    int j = get_global_id(0) + x_min;

    xcomp1(
        j, k,
        x_min, x_max,
        y_min, y_max,
        mass_flux_x,
        ener_flux,
        vol_flux_x,
        pre_vol,
        density1,
        energy1,
        vertexdx);
}


void kernel ycomp1_kernel(
    int x_min, int x_max, int y_min, int y_max,
    global double* mass_flux_y,
    global double* ener_flux,
    const global double* vol_flux_y,
    const global double* pre_vol,
    const global double* density1,
    const global double* energy1,
    const global double* vertexdy)
{
    int k = get_global_id(1) + y_min;
    int j = get_global_id(0) + x_min;

    ycomp1(
        j, k,
        x_min, x_max,
        y_min, y_max,
        mass_flux_y,
        ener_flux,
        vol_flux_y,
        pre_vol,
        density1,
        energy1,
        vertexdy);
}



void kernel xcomp2_kernel(
    int x_min, int x_max, int y_min, int y_max,
    global double* pre_mass,
    global double* post_mass,
    global double* post_ener,
    global double* advec_vol,
    global double* density1,
    global double* energy1,
    const global double* pre_vol,
    const global double* mass_flux_x,
    const global double* ener_flux,
    const global double* vol_flux_x)
{
    int k = get_global_id(1) + y_min;
    int j = get_global_id(0) + x_min;

    xcomp2(
        j, k,
        x_min, x_max,
        y_min, y_max,
        pre_mass,
        post_mass,
        post_ener,
        advec_vol,
        density1,
        energy1,
        pre_vol,
        mass_flux_x,
        ener_flux,
        vol_flux_x);
}


void kernel ycomp2_kernel(
    int x_min, int x_max, int y_min, int y_max,
    global double* pre_mass,
    global double* post_mass,
    global double* post_ener,
    global double* advec_vol,
    global double* density1,
    global double* energy1,
    const global double* pre_vol,
    const global double* mass_flux_y,
    const global double* ener_flux,
    const global double* vol_flux_y)
{
    int k = get_global_id(1) + y_min;
    int j = get_global_id(0) + x_min;

    ycomp2(
        j, k,
        x_min, x_max,
        y_min, y_max,
        pre_mass,
        post_mass,
        post_ener,
        advec_vol,
        density1,
        energy1,
        pre_vol,
        mass_flux_y,
        ener_flux,
        vol_flux_y);
}

