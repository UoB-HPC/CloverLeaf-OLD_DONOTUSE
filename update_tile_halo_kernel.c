#include "definitions_c.h"


void update_tile_halo_l_kernel_c_(
    int *xmin, int *xmax, int *ymin, int *ymax,
    double *density0,
    double *energy0,
    double *pressure,
    double *viscosity,
    double *soundspeed,
    double *density1,
    double *energy1,
    double *xvel0,
    double *yvel0,
    double *xvel1,
    double *yvel1,
    double *vol_flux_x,
    double *vol_flux_y,
    double *mass_flux_x,
    double *mass_flux_y,
    int *leftxmin, int *leftxmax, int *leftymin, int *leftymax,
    double *left_density0,
    double *left_energy0,
    double *left_pressure,
    double *left_viscosity,
    double *left_soundspeed,
    double *left_density1,
    double *left_energy1,
    double *left_xvel0,
    double *left_yvel0,
    double *left_xvel1,
    double *left_yvel1,
    double *left_vol_flux_x,
    double *left_vol_flux_y,
    double *left_mass_flux_x,
    double *left_mass_flux_y,
    int *fields,
    int *_depth)
{
    int x_min  = *xmin,
        x_max = *xmax,
        y_min = *ymin,
        y_max = *ymax;
    int left_xmin = *leftxmin,
        left_xmax = *leftxmax,
        left_ymin = *leftymin,
        left_ymax = *leftymax;
    int depth = *_depth;

    // Density 0
    if (fields[FIELD_DENSITY0] == 1) {

        for (int  k  =  y_min - depth;  k  <=  y_max + depth;  k ++) {
            for (int  j  =  1;  j  <=  depth;  j ++) {
                density0[FTNREF2D(x_min - j,  k, x_max + 4, x_min - 2, y_min - 2)] =
                    left_density0[FTNREF2D(left_xmax + 1 - j,  k, left_xmax + 4, left_xmin - 2, left_ymin - 2)];
            }
        }

    }

    // Density 1
    if (fields[FIELD_DENSITY1] == 1) {

        for (int  k  =  y_min - depth;  k  <=  y_max + depth;  k ++) {
            for (int  j  =  1;  j  <=  depth;  j ++) {
                density1[FTNREF2D(x_min - j,  k, x_max + 4, x_min - 2, y_min - 2)] =
                    left_density1[FTNREF2D(left_xmax + 1 - j,  k, left_xmax + 4, left_xmin - 2, left_ymin - 2)];
            }
        }

    }


    // Energy 0
    if (fields[FIELD_ENERGY0] == 1) {

        for (int  k  =  y_min - depth;  k  <=  y_max + depth;  k ++) {
            for (int  j  =  1;  j  <=  depth;  j ++) {
                energy0[FTNREF2D(x_min - j,  k, x_max + 4, x_min - 2, y_min - 2)] =
                    left_energy0[FTNREF2D(left_xmax + 1 - j,  k, left_xmax + 4, left_xmin - 2, left_ymin - 2)];
            }
        }

    }

    // Energy 1
    if (fields[FIELD_DENSITY1] == 1) {

        for (int  k  =  y_min - depth;  k  <=  y_max + depth;  k ++) {
            for (int  j  =  1;  j  <=  depth;  j ++) {
                energy1[FTNREF2D(x_min - j,  k, x_max + 4, x_min - 2, y_min - 2)] =
                    left_energy1[FTNREF2D(left_xmax + 1 - j,  k, left_xmax + 4, left_xmin - 2, left_ymin - 2)];
            }
        }

    }


    // Pressure
    if (fields[FIELD_PRESSURE] == 1) {

        for (int  k  =  y_min - depth;  k  <=  y_max + depth;  k ++) {
            for (int  j  =  1;  j  <=  depth;  j ++) {
                pressure[FTNREF2D(x_min - j,  k, x_max + 4, x_min - 2, y_min - 2)] =
                    left_pressure[FTNREF2D(left_xmax + 1 - j,  k, left_xmax + 4, left_xmin - 2, left_ymin - 2)];
            }
        }

    }

    // Viscocity
    if (fields[FIELD_VISCOSITY] == 1) {

        for (int  k  =  y_min - depth;  k  <=  y_max + depth;  k ++) {
            for (int  j  =  1;  j  <=  depth;  j ++) {
                viscosity[FTNREF2D(x_min - j,  k, x_max + 4, x_min - 2, y_min - 2)] =
                    left_viscosity[FTNREF2D(left_xmax + 1 - j,  k, left_xmax + 4, left_xmin - 2, left_ymin - 2)];
            }
        }

    }

    // Soundspeed
    if (fields[FIELD_SOUNDSPEED] == 1) {

        for (int  k  =  y_min - depth;  k  <=  y_max + depth;  k ++) {
            for (int  j  =  1;  j  <=  depth;  j ++) {
                soundspeed[FTNREF2D(x_min - j,  k, x_max + 4, x_min - 2, y_min - 2)] =
                    left_soundspeed[FTNREF2D(left_xmax + 1 - j,  k, left_xmax + 4, left_xmin - 2, left_ymin - 2)];
            }
        }

    }


    // XVEL 0
    if (fields[FIELD_XVEL0] == 1) {

        for (int  k  =  y_min - depth;  k  <=  y_max + 1 + depth;  k ++) {
            for (int  j  =  1;  j  <=  depth;  j ++) {
                xvel0[FTNREF2D(x_min - j,  k, x_max + 5, x_min - 2, y_min - 2)] =
                    left_xvel0[FTNREF2D(left_xmax + 1 - j,  k, left_xmax + 5, left_xmin - 2, left_ymin - 2)];
            }
        }

    }

    // XVEL 1
    if (fields[FIELD_XVEL1] == 1) {

        for (int  k  =  y_min - depth;  k  <=  y_max + 1 + depth;  k ++) {
            for (int  j  =  1;  j  <=  depth;  j ++) {
                xvel1[FTNREF2D(x_min - j,  k, x_max + 5, x_min - 2, y_min - 2)] =
                    left_xvel1[FTNREF2D(left_xmax + 1 - j,  k, left_xmax + 5, left_xmin - 2, left_ymin - 2)];
            }
        }

    }

    // YVEL 0
    if (fields[FIELD_YVEL0] == 1) {

        for (int  k  =  y_min - depth;  k  <=  y_max + 1 + depth;  k ++) {
            for (int  j  =  1;  j  <=  depth;  j ++) {
                yvel0[FTNREF2D(x_min - j,  k, x_max + 5, x_min - 2, y_min - 2)] =
                    left_yvel0[FTNREF2D(left_xmax + 1 - j,  k, left_xmax + 5, left_xmin - 2, left_ymin - 2)];
            }
        }

    }

    // YVEL 1
    if (fields[FIELD_YVEL1] == 1) {

        for (int  k  =  y_min - depth;  k  <=  y_max + 1 + depth;  k ++) {
            for (int  j  =  1;  j  <=  depth;  j ++) {
                yvel1[FTNREF2D(x_min - j,  k, x_max + 5, x_min - 2, y_min - 2)] =
                    left_yvel1[FTNREF2D(left_xmax + 1 - j,  k, left_xmax + 5, left_xmin - 2, left_ymin - 2)];
            }
        }

    }

    // VOL_FLUX_X
    if (fields[FIELD_VOL_FLUX_X] == 1) {

        for (int  k  =  y_min - depth;  k  <=  y_max + depth;  k ++) {
            for (int  j  =  1;  j  <=  depth;  j ++) {
                vol_flux_x[FTNREF2D(x_min - j,  k, x_max + 5, x_min - 2, y_min - 2)] =
                    left_vol_flux_x[FTNREF2D(left_xmax + 1 - j,  k, left_xmax + 5, left_xmin - 2, left_ymin - 2)];
            }
        }

    }

    // MASS_FLUX_X
    if (fields[FIELD_MASS_FLUX_X] == 1) {

        for (int  k  =  y_min - depth;  k  <=  y_max + depth;  k ++) {
            for (int  j  =  1;  j  <=  depth;  j ++) {
                mass_flux_x[FTNREF2D(x_min - j,  k, x_max + 5, x_min - 2, y_min - 2)] =
                    left_mass_flux_x[FTNREF2D(left_xmax + 1 - j,  k, left_xmax + 5, left_xmin - 2, left_ymin - 2)];
            }
        }

    }

    // VOL_FLUX_Y
    if (fields[FIELD_VOL_FLUX_Y] == 1) {

        for (int  k  =  y_min - depth;  k  <=  y_max + 1 + depth;  k ++) {
            for (int  j  =  1;  j  <=  depth;  j ++) {
                vol_flux_y[FTNREF2D(x_min - j,  k, x_max + 4, x_min - 2, y_min - 2)] =
                    left_vol_flux_y[FTNREF2D(left_xmax + 1 - j,  k, left_xmax + 4, left_xmin - 2, left_ymin - 2)];
            }
        }

    }

    // MASS_FLUX_Y
    if (fields[FIELD_MASS_FLUX_Y] == 1) {

        for (int  k  =  y_min - depth;  k  <=  y_max + 1 + depth;  k ++) {
            for (int  j  =  1;  j  <=  depth;  j ++) {
                mass_flux_y[FTNREF2D(x_min - j,  k, x_max + 4, x_min - 2, y_min - 2)] =
                    left_mass_flux_y[FTNREF2D(left_xmax + 1 - j,  k, left_xmax + 4, left_xmin - 2, left_ymin - 2)];
            }
        }

    }
}
void update_tile_halo_r_kernel_c_(
    int *xmin, int *xmax, int *ymin, int *ymax,
    double *density0,
    double *energy0,
    double *pressure,
    double *viscosity,
    double *soundspeed,
    double *density1,
    double *energy1,
    double *xvel0,
    double *yvel0,
    double *xvel1,
    double *yvel1,
    double *vol_flux_x,
    double *vol_flux_y,
    double *mass_flux_x,
    double *mass_flux_y,
    int *rightxmin, int *rightxmax, int *rightymin, int *rightymax,
    double *right_density0,
    double *right_energy0,
    double *right_pressure,
    double *right_viscosity,
    double *right_soundspeed,
    double *right_density1,
    double *right_energy1,
    double *right_xvel0,
    double *right_yvel0,
    double *right_xvel1,
    double *right_yvel1,
    double *right_vol_flux_x,
    double *right_vol_flux_y,
    double *right_mass_flux_x,
    double *right_mass_flux_y,
    int *fields,
    int *_depth)
{
    int x_min  = *xmin,
        x_max = *xmax,
        y_min = *ymin,
        y_max = *ymax;
    int right_xmin = *rightxmin,
        right_xmax = *rightxmax,
        right_ymin = *rightymin,
        right_ymax = *rightymax;
    int depth = *_depth;

    // Density 0
    if (fields[FIELD_DENSITY0] == 1) {

        for (int  k  =  y_min - depth;  k  <=  y_max + depth;  k ++) {
            for (int  j  =  1;  j  <=  depth;  j ++) {
                density0[FTNREF2D(x_max + j,  k, x_max + 4, x_min - 2, y_min - 2)] =
                    right_density0[FTNREF2D(right_xmin - 1 + j,  k, right_xmax + 4, right_xmin - 2, right_ymin - 2)];
            }
        }

    }

    // Density 1
    if (fields[FIELD_DENSITY1] == 1) {

        for (int  k  =  y_min - depth;  k  <=  y_max + depth;  k ++) {
            for (int  j  =  1;  j  <=  depth;  j ++) {
                density1[FTNREF2D(x_max + j,  k, x_max + 4, x_min - 2, y_min - 2)] =
                    right_density1[FTNREF2D(right_xmin - 1 + j,  k, right_xmax + 4, right_xmin - 2, right_ymin - 2)];
            }
        }

    }


    // Energy 0
    if (fields[FIELD_ENERGY0] == 1) {

        for (int  k  =  y_min - depth;  k  <=  y_max + depth;  k ++) {
            for (int  j  =  1;  j  <=  depth;  j ++) {
                energy0[FTNREF2D(x_max + j,  k, x_max + 4, x_min - 2, y_min - 2)] =
                    right_energy0[FTNREF2D(right_xmin - 1 + j,  k, right_xmax + 4, right_xmin - 2, right_ymin - 2)];
            }
        }

    }

    // Energy 1
    if (fields[FIELD_DENSITY1] == 1) {

        for (int  k  =  y_min - depth;  k  <=  y_max + depth;  k ++) {
            for (int  j  =  1;  j  <=  depth;  j ++) {
                energy1[FTNREF2D(x_max + j,  k, x_max + 4, x_min - 2, y_min - 2)] =
                    right_energy1[FTNREF2D(right_xmin - 1 + j,  k, right_xmax + 4, right_xmin - 2, right_ymin - 2)];
            }
        }

    }


    // Pressure
    if (fields[FIELD_PRESSURE] == 1) {

        for (int  k  =  y_min - depth;  k  <=  y_max + depth;  k ++) {
            for (int  j  =  1;  j  <=  depth;  j ++) {
                pressure[FTNREF2D(x_max + j,  k, x_max + 4, x_min - 2, y_min - 2)] =
                    right_pressure[FTNREF2D(right_xmin - 1 + j,  k, right_xmax + 4, right_xmin - 2, right_ymin - 2)];
            }
        }

    }

    // Viscocity
    if (fields[FIELD_VISCOSITY] == 1) {

        for (int  k  =  y_min - depth;  k  <=  y_max + depth;  k ++) {
            for (int  j  =  1;  j  <=  depth;  j ++) {
                viscosity[FTNREF2D(x_max + j,  k, x_max + 4, x_min - 2, y_min - 2)] =
                    right_viscosity[FTNREF2D(right_xmin - 1 + j,  k, right_xmax + 4, right_xmin - 2, right_ymin - 2)];
            }
        }

    }

    // Soundspeed
    if (fields[FIELD_SOUNDSPEED] == 1) {

        for (int  k  =  y_min - depth;  k  <=  y_max + depth;  k ++) {
            for (int  j  =  1;  j  <=  depth;  j ++) {
                soundspeed[FTNREF2D(x_max + j,  k, x_max + 4, x_min - 2, y_min - 2)] =
                    right_soundspeed[FTNREF2D(right_xmin - 1 + j,  k, right_xmax + 4, right_xmin - 2, right_ymin - 2)];
            }
        }

    }


    // XVEL 0
    if (fields[FIELD_XVEL0] == 1) {

        for (int  k  =  y_min - depth;  k  <=  y_max + 1 + depth;  k ++) {
            for (int  j  =  1;  j  <=  depth;  j ++) {
                xvel0[FTNREF2D(x_max + 1 + j,  k, x_max + 5, x_min - 2, y_min - 2)] =
                    right_xvel0[FTNREF2D(right_xmin + 1 - 1 + j,  k, right_xmax + 5, right_xmin - 2, right_ymin - 2)];
            }
        }

    }

    // XVEL 1
    if (fields[FIELD_XVEL1] == 1) {

        for (int  k  =  y_min - depth;  k  <=  y_max + 1 + depth;  k ++) {
            for (int  j  =  1;  j  <=  depth;  j ++) {
                xvel1[FTNREF2D(x_max + 1 + j,  k, x_max + 5, x_min - 2, y_min - 2)] =
                    right_xvel1[FTNREF2D(right_xmin + 1 - 1 + j,  k, right_xmax + 5, right_xmin - 2, right_ymin - 2)];
            }
        }

    }

    // YVEL 0
    if (fields[FIELD_YVEL0] == 1) {

        for (int  k  =  y_min - depth;  k  <=  y_max + 1 + depth;  k ++) {
            for (int  j  =  1;  j  <=  depth;  j ++) {
                yvel0[FTNREF2D(x_max + 1 + j,  k, x_max + 5, x_min - 2, y_min - 2)] =
                    right_yvel0[FTNREF2D(right_xmin + 1 - 1 + j,  k, right_xmax + 5, right_xmin - 2, right_ymin - 2)];
            }
        }

    }

    // YVEL 1
    if (fields[FIELD_YVEL1] == 1) {

        for (int  k  =  y_min - depth;  k  <=  y_max + 1 + depth;  k ++) {
            for (int  j  =  1;  j  <=  depth;  j ++) {
                yvel1[FTNREF2D(x_max + 1 + j,  k, x_max + 5, x_min - 2, y_min - 2)] =
                    right_yvel1[FTNREF2D(right_xmin + 1 - 1 + j,  k, right_xmax + 5, right_xmin - 2, right_ymin - 2)];
            }
        }

    }

    // VOL_FLUX_X
    if (fields[FIELD_VOL_FLUX_X] == 1) {

        for (int  k  =  y_min - depth;  k  <=  y_max + depth;  k ++) {
            for (int  j  =  1;  j  <=  depth;  j ++) {
                vol_flux_x[FTNREF2D(x_max + 1 + j,  k, x_max + 5, x_min - 2, y_min - 2)] =
                    right_vol_flux_x[FTNREF2D(right_xmin + 1 - 1 + j,  k, right_xmax + 5, right_xmin - 2, right_ymin - 2)];
            }
        }

    }

    // MASS_FLUX_X
    if (fields[FIELD_MASS_FLUX_X] == 1) {

        for (int  k  =  y_min - depth;  k  <=  y_max + depth;  k ++) {
            for (int  j  =  1;  j  <=  depth;  j ++) {
                mass_flux_x[FTNREF2D(x_max + 1 + j,  k, x_max + 5, x_min - 2, y_min - 2)] =
                    right_mass_flux_x[FTNREF2D(right_xmin + 1 - 1 + j,  k, right_xmax + 5, right_xmin - 2, right_ymin - 2)];
            }
        }

    }

    // VOL_FLUX_Y
    if (fields[FIELD_VOL_FLUX_Y] == 1) {

        for (int  k  =  y_min - depth;  k  <=  y_max + 1 + depth;  k ++) {
            for (int  j  =  1;  j  <=  depth;  j ++) {
                vol_flux_y[FTNREF2D(x_max + j,  k, x_max + 4, x_min - 2, y_min - 2)] =
                    right_vol_flux_y[FTNREF2D(right_xmin - 1 + j,  k, right_xmax + 4, right_xmin - 2, right_ymin - 2)];
            }
        }

    }

    // MASS_FLUX_Y
    if (fields[FIELD_MASS_FLUX_Y] == 1) {

        for (int  k  =  y_min - depth;  k  <=  y_max + 1 + depth;  k ++) {
            for (int  j  =  1;  j  <=  depth;  j ++) {
                mass_flux_y[FTNREF2D(x_max + j,  k, x_max + 4, x_min - 2, y_min - 2)] =
                    right_mass_flux_y[FTNREF2D(right_xmin - 1 + j,  k, right_xmax + 4, right_xmin - 2, right_ymin - 2)];
            }
        }

    }
}
void update_tile_halo_t_kernel_c_(
    int *xmin, int *xmax, int *ymin, int *ymax,
    double *density0,
    double *energy0,
    double *pressure,
    double *viscosity,
    double *soundspeed,
    double *density1,
    double *energy1,
    double *xvel0,
    double *yvel0,
    double *xvel1,
    double *yvel1,
    double *vol_flux_x,
    double *vol_flux_y,
    double *mass_flux_x,
    double *mass_flux_y,
    int *leftxmin, int *leftxmax, int *leftymin, int *leftymax,
    double *left_density0,
    double *left_energy0,
    double *left_pressure,
    double *left_viscosity,
    double *left_soundspeed,
    double *left_density1,
    double *left_energy1,
    double *left_xvel0,
    double *left_yvel0,
    double *left_xvel1,
    double *left_yvel1,
    double *left_vol_flux_x,
    double *left_vol_flux_y,
    double *left_mass_flux_x,
    double *left_mass_flux_y,
    int *fields,
    int *_depth)
{
    int x_min  = *xmin,
        x_max = *xmax,
        y_min = *ymin,
        y_max = *ymax;
    int left_xmin = *leftxmin,
        left_xmax = *leftxmax,
        left_ymin = *leftymin,
        left_ymax = *leftymax;
    int depth = *_depth;

    // Density 0
    if (fields[FIELD_DENSITY0] == 1) {

        for (int  k  =  y_min - depth;  k  <=  y_max + depth;  k ++) {
            for (int  j  =  1;  j  <=  depth;  j ++) {
                density0[FTNREF2D(x_min - j,  k, x_max + 4, x_min - 2, y_min - 2)] =
                    left_density0[FTNREF2D(left_xmax + 1 - j,  k, left_xmax + 4, left_xmin - 2, left_ymin - 2)];
            }
        }

    }

    // Density 1
    if (fields[FIELD_DENSITY1] == 1) {

        for (int  k  =  y_min - depth;  k  <=  y_max + depth;  k ++) {
            for (int  j  =  1;  j  <=  depth;  j ++) {
                density1[FTNREF2D(x_min - j,  k, x_max + 4, x_min - 2, y_min - 2)] =
                    left_density1[FTNREF2D(left_xmax + 1 - j,  k, left_xmax + 4, left_xmin - 2, left_ymin - 2)];
            }
        }

    }


    // Energy 0
    if (fields[FIELD_ENERGY0] == 1) {

        for (int  k  =  y_min - depth;  k  <=  y_max + depth;  k ++) {
            for (int  j  =  1;  j  <=  depth;  j ++) {
                energy0[FTNREF2D(x_min - j,  k, x_max + 4, x_min - 2, y_min - 2)] =
                    left_energy0[FTNREF2D(left_xmax + 1 - j,  k, left_xmax + 4, left_xmin - 2, left_ymin - 2)];
            }
        }

    }

    // Energy 1
    if (fields[FIELD_DENSITY1] == 1) {

        for (int  k  =  y_min - depth;  k  <=  y_max + depth;  k ++) {
            for (int  j  =  1;  j  <=  depth;  j ++) {
                energy1[FTNREF2D(x_min - j,  k, x_max + 4, x_min - 2, y_min - 2)] =
                    left_energy1[FTNREF2D(left_xmax + 1 - j,  k, left_xmax + 4, left_xmin - 2, left_ymin - 2)];
            }
        }

    }


    // Pressure
    if (fields[FIELD_PRESSURE] == 1) {

        for (int  k  =  y_min - depth;  k  <=  y_max + depth;  k ++) {
            for (int  j  =  1;  j  <=  depth;  j ++) {
                pressure[FTNREF2D(x_min - j,  k, x_max + 4, x_min - 2, y_min - 2)] =
                    left_pressure[FTNREF2D(left_xmax + 1 - j,  k, left_xmax + 4, left_xmin - 2, left_ymin - 2)];
            }
        }

    }

    // Viscocity
    if (fields[FIELD_VISCOSITY] == 1) {

        for (int  k  =  y_min - depth;  k  <=  y_max + depth;  k ++) {
            for (int  j  =  1;  j  <=  depth;  j ++) {
                viscosity[FTNREF2D(x_min - j,  k, x_max + 4, x_min - 2, y_min - 2)] =
                    left_viscosity[FTNREF2D(left_xmax + 1 - j,  k, left_xmax + 4, left_xmin - 2, left_ymin - 2)];
            }
        }

    }

    // Soundspeed
    if (fields[FIELD_SOUNDSPEED] == 1) {

        for (int  k  =  y_min - depth;  k  <=  y_max + depth;  k ++) {
            for (int  j  =  1;  j  <=  depth;  j ++) {
                soundspeed[FTNREF2D(x_min - j,  k, x_max + 4, x_min - 2, y_min - 2)] =
                    left_soundspeed[FTNREF2D(left_xmax + 1 - j,  k, left_xmax + 4, left_xmin - 2, left_ymin - 2)];
            }
        }

    }


    // XVEL 0
    if (fields[FIELD_XVEL0] == 1) {

        for (int  k  =  y_min - depth;  k  <=  y_max + 1 + depth;  k ++) {
            for (int  j  =  1;  j  <=  depth;  j ++) {
                xvel0[FTNREF2D(x_min - j,  k, x_max + 5, x_min - 2, y_min - 2)] =
                    left_xvel0[FTNREF2D(left_xmax + 1 - j,  k, left_xmax + 5, left_xmin - 2, left_ymin - 2)];
            }
        }

    }

    // XVEL 1
    if (fields[FIELD_XVEL1] == 1) {

        for (int  k  =  y_min - depth;  k  <=  y_max + 1 + depth;  k ++) {
            for (int  j  =  1;  j  <=  depth;  j ++) {
                xvel1[FTNREF2D(x_min - j,  k, x_max + 5, x_min - 2, y_min - 2)] =
                    left_xvel1[FTNREF2D(left_xmax + 1 - j,  k, left_xmax + 5, left_xmin - 2, left_ymin - 2)];
            }
        }

    }

    // YVEL 0
    if (fields[FIELD_YVEL0] == 1) {

        for (int  k  =  y_min - depth;  k  <=  y_max + 1 + depth;  k ++) {
            for (int  j  =  1;  j  <=  depth;  j ++) {
                yvel0[FTNREF2D(x_min - j,  k, x_max + 5, x_min - 2, y_min - 2)] =
                    left_yvel0[FTNREF2D(left_xmax + 1 - j,  k, left_xmax + 5, left_xmin - 2, left_ymin - 2)];
            }
        }

    }

    // YVEL 1
    if (fields[FIELD_YVEL1] == 1) {

        for (int  k  =  y_min - depth;  k  <=  y_max + 1 + depth;  k ++) {
            for (int  j  =  1;  j  <=  depth;  j ++) {
                yvel1[FTNREF2D(x_min - j,  k, x_max + 5, x_min - 2, y_min - 2)] =
                    left_yvel1[FTNREF2D(left_xmax + 1 - j,  k, left_xmax + 5, left_xmin - 2, left_ymin - 2)];
            }
        }

    }

    // VOL_FLUX_X
    if (fields[FIELD_VOL_FLUX_X] == 1) {

        for (int  k  =  y_min - depth;  k  <=  y_max + depth;  k ++) {
            for (int  j  =  1;  j  <=  depth;  j ++) {
                vol_flux_x[FTNREF2D(x_min - j,  k, x_max + 5, x_min - 2, y_min - 2)] =
                    left_vol_flux_x[FTNREF2D(left_xmax + 1 - j,  k, left_xmax + 5, left_xmin - 2, left_ymin - 2)];
            }
        }

    }

    // MASS_FLUX_X
    if (fields[FIELD_MASS_FLUX_X] == 1) {

        for (int  k  =  y_min - depth;  k  <=  y_max + depth;  k ++) {
            for (int  j  =  1;  j  <=  depth;  j ++) {
                mass_flux_x[FTNREF2D(x_min - j,  k, x_max + 5, x_min - 2, y_min - 2)] =
                    left_mass_flux_x[FTNREF2D(left_xmax + 1 - j,  k, left_xmax + 5, left_xmin - 2, left_ymin - 2)];
            }
        }

    }

    // VOL_FLUX_Y
    if (fields[FIELD_VOL_FLUX_Y] == 1) {

        for (int  k  =  y_min - depth;  k  <=  y_max + 1 + depth;  k ++) {
            for (int  j  =  1;  j  <=  depth;  j ++) {
                vol_flux_y[FTNREF2D(x_min - j,  k, x_max + 5, x_min - 2, y_min - 2)] =
                    left_vol_flux_y[FTNREF2D(left_xmax + 1 - j,  k, left_xmax + 5, left_xmin - 2, left_ymin - 2)];
            }
        }

    }

    // MASS_FLUX_Y
    if (fields[FIELD_MASS_FLUX_Y] == 1) {

        for (int  k  =  y_min - depth;  k  <=  y_max + 1 + depth;  k ++) {
            for (int  j  =  1;  j  <=  depth;  j ++) {
                mass_flux_y[FTNREF2D(x_min - j,  k, x_max + 5, x_min - 2, y_min - 2)] =
                    left_mass_flux_y[FTNREF2D(left_xmax + 1 - j,  k, left_xmax + 5, left_xmin - 2, left_ymin - 2)];
            }
        }

    }
}
void update_tile_halo_b_kernel_c_(
    int *xmin, int *xmax, int *ymin, int *ymax,
    double *density0,
    double *energy0,
    double *pressure,
    double *viscosity,
    double *soundspeed,
    double *density1,
    double *energy1,
    double *xvel0,
    double *yvel0,
    double *xvel1,
    double *yvel1,
    double *vol_flux_x,
    double *vol_flux_y,
    double *mass_flux_x,
    double *mass_flux_y,
    int *leftxmin, int *leftxmax, int *leftymin, int *leftymax,
    double *left_density0,
    double *left_energy0,
    double *left_pressure,
    double *left_viscosity,
    double *left_soundspeed,
    double *left_density1,
    double *left_energy1,
    double *left_xvel0,
    double *left_yvel0,
    double *left_xvel1,
    double *left_yvel1,
    double *left_vol_flux_x,
    double *left_vol_flux_y,
    double *left_mass_flux_x,
    double *left_mass_flux_y,
    int *fields,
    int *_depth)
{
    int x_min  = *xmin,
        x_max = *xmax,
        y_min = *ymin,
        y_max = *ymax;
    int left_xmin = *leftxmin,
        left_xmax = *leftxmax,
        left_ymin = *leftymin,
        left_ymax = *leftymax;
    int depth = *_depth;

    // Density 0
    if (fields[FIELD_DENSITY0] == 1) {

        for (int  k  =  y_min - depth;  k  <=  y_max + depth;  k ++) {
            for (int  j  =  1;  j  <=  depth;  j ++) {
                density0[FTNREF2D(x_min - j,  k, x_max + 4, x_min - 2, y_min - 2)] =
                    left_density0[FTNREF2D(left_xmax + 1 - j,  k, left_xmax + 4, left_xmin - 2, left_ymin - 2)];
            }
        }

    }

    // Density 1
    if (fields[FIELD_DENSITY1] == 1) {

        for (int  k  =  y_min - depth;  k  <=  y_max + depth;  k ++) {
            for (int  j  =  1;  j  <=  depth;  j ++) {
                density1[FTNREF2D(x_min - j,  k, x_max + 4, x_min - 2, y_min - 2)] =
                    left_density1[FTNREF2D(left_xmax + 1 - j,  k, left_xmax + 4, left_xmin - 2, left_ymin - 2)];
            }
        }

    }


    // Energy 0
    if (fields[FIELD_ENERGY0] == 1) {

        for (int  k  =  y_min - depth;  k  <=  y_max + depth;  k ++) {
            for (int  j  =  1;  j  <=  depth;  j ++) {
                energy0[FTNREF2D(x_min - j,  k, x_max + 4, x_min - 2, y_min - 2)] =
                    left_energy0[FTNREF2D(left_xmax + 1 - j,  k, left_xmax + 4, left_xmin - 2, left_ymin - 2)];
            }
        }

    }

    // Energy 1
    if (fields[FIELD_DENSITY1] == 1) {

        for (int  k  =  y_min - depth;  k  <=  y_max + depth;  k ++) {
            for (int  j  =  1;  j  <=  depth;  j ++) {
                energy1[FTNREF2D(x_min - j,  k, x_max + 4, x_min - 2, y_min - 2)] =
                    left_energy1[FTNREF2D(left_xmax + 1 - j,  k, left_xmax + 4, left_xmin - 2, left_ymin - 2)];
            }
        }

    }


    // Pressure
    if (fields[FIELD_PRESSURE] == 1) {

        for (int  k  =  y_min - depth;  k  <=  y_max + depth;  k ++) {
            for (int  j  =  1;  j  <=  depth;  j ++) {
                pressure[FTNREF2D(x_min - j,  k, x_max + 4, x_min - 2, y_min - 2)] =
                    left_pressure[FTNREF2D(left_xmax + 1 - j,  k, left_xmax + 4, left_xmin - 2, left_ymin - 2)];
            }
        }

    }

    // Viscocity
    if (fields[FIELD_VISCOSITY] == 1) {

        for (int  k  =  y_min - depth;  k  <=  y_max + depth;  k ++) {
            for (int  j  =  1;  j  <=  depth;  j ++) {
                viscosity[FTNREF2D(x_min - j,  k, x_max + 4, x_min - 2, y_min - 2)] =
                    left_viscosity[FTNREF2D(left_xmax + 1 - j,  k, left_xmax + 4, left_xmin - 2, left_ymin - 2)];
            }
        }

    }

    // Soundspeed
    if (fields[FIELD_SOUNDSPEED] == 1) {

        for (int  k  =  y_min - depth;  k  <=  y_max + depth;  k ++) {
            for (int  j  =  1;  j  <=  depth;  j ++) {
                soundspeed[FTNREF2D(x_min - j,  k, x_max + 4, x_min - 2, y_min - 2)] =
                    left_soundspeed[FTNREF2D(left_xmax + 1 - j,  k, left_xmax + 4, left_xmin - 2, left_ymin - 2)];
            }
        }

    }


    // XVEL 0
    if (fields[FIELD_XVEL0] == 1) {

        for (int  k  =  y_min - depth;  k  <=  y_max + 1 + depth;  k ++) {
            for (int  j  =  1;  j  <=  depth;  j ++) {
                xvel0[FTNREF2D(x_min - j,  k, x_max + 5, x_min - 2, y_min - 2)] =
                    left_xvel0[FTNREF2D(left_xmax + 1 - j,  k, left_xmax + 5, left_xmin - 2, left_ymin - 2)];
            }
        }

    }

    // XVEL 1
    if (fields[FIELD_XVEL1] == 1) {

        for (int  k  =  y_min - depth;  k  <=  y_max + 1 + depth;  k ++) {
            for (int  j  =  1;  j  <=  depth;  j ++) {
                xvel1[FTNREF2D(x_min - j,  k, x_max + 5, x_min - 2, y_min - 2)] =
                    left_xvel1[FTNREF2D(left_xmax + 1 - j,  k, left_xmax + 5, left_xmin - 2, left_ymin - 2)];
            }
        }

    }

    // YVEL 0
    if (fields[FIELD_YVEL0] == 1) {

        for (int  k  =  y_min - depth;  k  <=  y_max + 1 + depth;  k ++) {
            for (int  j  =  1;  j  <=  depth;  j ++) {
                yvel0[FTNREF2D(x_min - j,  k, x_max + 5, x_min - 2, y_min - 2)] =
                    left_yvel0[FTNREF2D(left_xmax + 1 - j,  k, left_xmax + 5, left_xmin - 2, left_ymin - 2)];
            }
        }

    }

    // YVEL 1
    if (fields[FIELD_YVEL1] == 1) {

        for (int  k  =  y_min - depth;  k  <=  y_max + 1 + depth;  k ++) {
            for (int  j  =  1;  j  <=  depth;  j ++) {
                yvel1[FTNREF2D(x_min - j,  k, x_max + 5, x_min - 2, y_min - 2)] =
                    left_yvel1[FTNREF2D(left_xmax + 1 - j,  k, left_xmax + 5, left_xmin - 2, left_ymin - 2)];
            }
        }

    }

    // VOL_FLUX_X
    if (fields[FIELD_VOL_FLUX_X] == 1) {

        for (int  k  =  y_min - depth;  k  <=  y_max + depth;  k ++) {
            for (int  j  =  1;  j  <=  depth;  j ++) {
                vol_flux_x[FTNREF2D(x_min - j,  k, x_max + 5, x_min - 2, y_min - 2)] =
                    left_vol_flux_x[FTNREF2D(left_xmax + 1 - j,  k, left_xmax + 5, left_xmin - 2, left_ymin - 2)];
            }
        }

    }

    // MASS_FLUX_X
    if (fields[FIELD_MASS_FLUX_X] == 1) {

        for (int  k  =  y_min - depth;  k  <=  y_max + depth;  k ++) {
            for (int  j  =  1;  j  <=  depth;  j ++) {
                mass_flux_x[FTNREF2D(x_min - j,  k, x_max + 5, x_min - 2, y_min - 2)] =
                    left_mass_flux_x[FTNREF2D(left_xmax + 1 - j,  k, left_xmax + 5, left_xmin - 2, left_ymin - 2)];
            }
        }

    }

    // VOL_FLUX_Y
    if (fields[FIELD_VOL_FLUX_Y] == 1) {

        for (int  k  =  y_min - depth;  k  <=  y_max + 1 + depth;  k ++) {
            for (int  j  =  1;  j  <=  depth;  j ++) {
                vol_flux_y[FTNREF2D(x_min - j,  k, x_max + 5, x_min - 2, y_min - 2)] =
                    left_vol_flux_y[FTNREF2D(left_xmax + 1 - j,  k, left_xmax + 5, left_xmin - 2, left_ymin - 2)];
            }
        }

    }

    // MASS_FLUX_Y
    if (fields[FIELD_MASS_FLUX_Y] == 1) {

        for (int  k  =  y_min - depth;  k  <=  y_max + 1 + depth;  k ++) {
            for (int  j  =  1;  j  <=  depth;  j ++) {
                mass_flux_y[FTNREF2D(x_min - j,  k, x_max + 5, x_min - 2, y_min - 2)] =
                    left_mass_flux_y[FTNREF2D(left_xmax + 1 - j,  k, left_xmax + 5, left_xmin - 2, left_ymin - 2)];
            }
        }

    }
}