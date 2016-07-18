// #include "definitions_c.h"
#include "ftocmacros.h"
#include <string.h>
#include <stdio.h>


void debug_kernel_(
    // char *postfix,
    int *id,
    int *xmin, int *xmax, int *ymin, int *ymax,
    double *density0,
    double *density1,
    double *energy0,
    double *energy1,
    double *pressure,
    double *viscosity,
    double *soundspeed,
    double *xvel0,
    double *xvel1,
    double *yvel0,
    double *yvel1,
    double *vol_flux_x,
    double *vol_flux_y,
    double *mass_flux_x,
    double *mass_flux_y,
    double *xarea,
    double *yarea,
    double *volume
)
{
    int x_min = *xmin,
        x_max = *xmax,
        y_min = *ymin,
        y_max = *ymax;
    char str[100] = "";
    sprintf(str, "%s.%d", "out", *id);
    FILE *f;
    f = fopen(str, "w");

    // exit(0);
    for (int j = x_min - 2; j <= x_max + 2; j++) {
        for (int k = y_min - 2; k <= y_max + 2; k++) {
            fprintf(f, "d0(%d, %d) = %.10e\n", j, k, density0[FTNREF2D(j  , k, x_max + 4, x_min - 2, y_min - 2)]);
        }
    }
    for (int j = x_min - 2; j <= x_max + 2; j++) {
        for (int k = y_min - 2; k <= y_max + 2; k++) {
            fprintf(f, "d1(%d, %d) = %.10e\n", j, k, density1[FTNREF2D(j  , k, x_max + 4, x_min - 2, y_min - 2)]);
        }
    }
    for (int j = x_min - 2; j <= x_max + 2; j++) {
        for (int k = y_min - 2; k <= y_max + 2; k++) {
            fprintf(f, "e0(%d, %d) = %.10e\n", j, k, energy0[FTNREF2D(j  , k, x_max + 4, x_min - 2, y_min - 2)]);
        }
    }
    for (int j = x_min - 2; j <= x_max + 2; j++) {
        for (int k = y_min - 2; k <= y_max + 2; k++) {
            fprintf(f, "e1(%d, %d) = %.10e\n", j, k, energy1[FTNREF2D(j  , k, x_max + 4, x_min - 2, y_min - 2)]);
        }
    }
    for (int j = x_min - 2; j <= x_max + 2; j++) {
        for (int k = y_min - 2; k <= y_max + 2; k++) {
            fprintf(f, "p(%d, %d) = %.10e\n", j, k, pressure[FTNREF2D(j  , k, x_max + 4, x_min - 2, y_min - 2)]);
        }
    }
    for (int j = x_min - 2; j <= x_max + 2; j++) {
        for (int k = y_min - 2; k <= y_max + 2; k++) {
            fprintf(f, "v(%d, %d) = %.10e\n", j, k, viscosity[FTNREF2D(j  , k, x_max + 4, x_min - 2, y_min - 2)]);
        }
    }
    for (int j = x_min - 2; j <= x_max + 2; j++) {
        for (int k = y_min - 2; k <= y_max + 2; k++) {
            fprintf(f, "s(%d, %d) = %.10e\n", j, k, soundspeed[FTNREF2D(j  , k, x_max + 4, x_min - 2, y_min - 2)]);
        }
    }

    for (int j = x_min - 2; j <= x_max + 2; j++) {
        for (int k = y_min - 2; k <= y_max + 2; k++) {
            fprintf(f, "x0(%d, %d) = %.10e\n", j, k, xvel0[FTNREF2D(j  , k, x_max + 5, x_min - 2, y_min - 2)]);
        }
    }
    for (int j = x_min - 2; j <= x_max + 2; j++) {
        for (int k = y_min - 2; k <= y_max + 2; k++) {
            fprintf(f, "x1(%d, %d) = %.10e\n", j, k, xvel1[FTNREF2D(j  , k, x_max + 5, x_min - 2, y_min - 2)]);
        }
    }
    for (int j = x_min - 2; j <= x_max + 2; j++) {
        for (int k = y_min - 2; k <= y_max + 2; k++) {
            fprintf(f, "y0(%d, %d) = %.10e\n", j, k, yvel0[FTNREF2D(j  , k, x_max + 5, x_min - 2, y_min - 2)]);
        }
    }
    for (int j = x_min - 2; j <= x_max + 2; j++) {
        for (int k = y_min - 2; k <= y_max + 2; k++) {
            fprintf(f, "y1(%d, %d) = %.10e\n", j, k, yvel1[FTNREF2D(j  , k, x_max + 5, x_min - 2, y_min - 2)]);
        }
    }

    for (int j = x_min - 2; j <= x_max + 2; j++) {
        for (int k = y_min - 2; k <= y_max + 2; k++) {
            fprintf(f, "vx(%d, %d) = %.10e\n", j, k, vol_flux_x[FTNREF2D(j  , k, x_max + 4, x_min - 2, y_min - 2)]);
        }
    }
    for (int j = x_min - 2; j <= x_max + 2; j++) {
        for (int k = y_min - 2; k <= y_max + 2; k++) {
            fprintf(f, "vy(%d, %d) = %.10e\n", j, k, vol_flux_y[FTNREF2D(j  , k, x_max + 5, x_min - 2, y_min - 2)]);
        }
    }
    for (int j = x_min - 2; j <= x_max + 2; j++) {
        for (int k = y_min - 2; k <= y_max + 2; k++) {
            fprintf(f, "mx(%d, %d) = %.10e\n", j, k, mass_flux_x[FTNREF2D(j  , k, x_max + 4, x_min - 2, y_min - 2)]);
        }
    }
    for (int j = x_min - 2; j <= x_max + 2; j++) {
        for (int k = y_min - 2; k <= y_max + 2; k++) {
            fprintf(f, "my(%d, %d) = %.10e\n", j, k, mass_flux_y[FTNREF2D(j  , k, x_max + 5, x_min - 2, y_min - 2)]);
        }
    }

    for (int j = x_min - 2; j <= x_max + 2; j++) {
        for (int k = y_min - 2; k <= y_max + 2; k++) {
            fprintf(f, "xa(%d, %d) = %.10e\n", j, k, xarea[FTNREF2D(j  , k, x_max + 5, x_min - 2, y_min - 2)]);
        }
    }
    for (int j = x_min - 2; j <= x_max + 2; j++) {
        for (int k = y_min - 2; k <= y_max + 2; k++) {
            fprintf(f, "ya(%d, %d) = %.10e\n", j, k, yarea[FTNREF2D(j  , k, x_max + 4, x_min - 2, y_min - 2)]);
        }
    }

    for (int j = x_min - 2; j <= x_max + 2; j++) {
        for (int k = y_min - 2; k <= y_max + 2; k++) {
            fprintf(f, "vol(%d, %d) = %.10e\n", j, k, volume[FTNREF2D(j  , k, x_max + 4, x_min - 2, y_min - 2)]);
        }
    }
    fclose(f);
}
