#include "calc_dt.h"
#include "definitions_c.h"
#include "kernels/calc_dt_kernel_c.c"
#include "string.h"

void calc_dt(int tile,
             double *local_dt,
             char* local_control,
             double *xl_pos,
             double *yl_pos,
             int *jldt,
             int *kldt)
{
    // *local_dt = g_big;
    int small = 0,
        l_control;
    double mindt = g_big;
    double lmin = g_big;

    #pragma omp parallel reduction(min:lmin)
    {

#ifdef USE_KOKKOS
        Kokkos::parallel_reduce(
            (chunk.tiles[tile].t_ymax) - (chunk.tiles[tile].t_ymin) + 1,
        KOKKOS_LAMBDA (const int& i, double & lmin) {
            int k = i + (chunk.tiles[tile].t_ymin);
#else
        for (int k = chunk.tiles[tile].t_ymin; k <= chunk.tiles[tile].t_ymax; k++) {
#endif
            for (int j = chunk.tiles[tile].t_xmin; j <= chunk.tiles[tile].t_xmax; j++) {
                double t = calc_dt_kernel_c_(
                               j, k,
                               chunk.tiles[tile].t_xmin,
                               chunk.tiles[tile].t_xmax,
                               chunk.tiles[tile].t_ymin,
                               chunk.tiles[tile].t_ymax,
                               chunk.tiles[tile].field.xarea,
                               chunk.tiles[tile].field.yarea,
                               chunk.tiles[tile].field.celldx,
                               chunk.tiles[tile].field.celldy,
                               chunk.tiles[tile].field.volume,
                               chunk.tiles[tile].field.density0,
                               chunk.tiles[tile].field.energy0,
                               chunk.tiles[tile].field.pressure,
                               chunk.tiles[tile].field.viscosity,
                               chunk.tiles[tile].field.soundspeed,
                               chunk.tiles[tile].field.xvel0,
                               chunk.tiles[tile].field.yvel0
                           );
                if (t < lmin)
                    lmin = t;
            }
        }
#ifdef USE_KOKKOS
        , mindt);
#else
        mindt = lmin;
#endif

    }
    *local_dt = mindt;

#ifdef USE_KOKKOS
    Kokkos::fence();
#endif


    double jk_control = 1.1;
    // Extract the mimimum timestep information
    // dtl_control = 10.01 * (jk_control - (int)(jk_control));
    jk_control = jk_control - (jk_control - (int)(jk_control));
    *jldt = 1; //MOD(INT(jk_control),x_max)
    *kldt = 1; //1+(jk_control/x_max)
    //xl_pos=cellx[FTNREF1D(jldt,xmin-2)];
    //yl_pos=celly[FTNREF1D(jldt,ymin-2)];

    if (*local_dt < dtmin) small = 1;

    l_control = 1;
    int x_min = chunk.tiles[tile].t_xmin,
        x_max = chunk.tiles[tile].t_xmax,
        y_min = chunk.tiles[tile].t_ymin,
        y_max = chunk.tiles[tile].t_ymax;
    if (small != 0) {
        printf("Timestep information:\n");
        printf("j, k                 :%i %i \n", *jldt, *kldt);
        printf("x, y                 :%f %f \n", *xl_pos, *yl_pos);
        printf("timestep : %f\n", *local_dt);
        printf("Cell velocities;\n");
        printf("%f %f \n", chunk.tiles[tile].field.xvel0[FTNREF2D(*jldt  , *kldt  , x_max + 5, x_min - 2, y_min - 2)], chunk.tiles[tile].field.yvel0[FTNREF2D(*jldt  , *kldt  , x_max + 5, x_min - 2, y_min - 2)]);
        printf("%f %f \n", chunk.tiles[tile].field.xvel0[FTNREF2D(*jldt + 1, *kldt  , x_max + 5, x_min - 2, y_min - 2)], chunk.tiles[tile].field.yvel0[FTNREF2D(*jldt + 1, *kldt  , x_max + 5, x_min - 2, y_min - 2)]);
        printf("%f %f \n", chunk.tiles[tile].field.xvel0[FTNREF2D(*jldt + 1, *kldt + 1, x_max + 5, x_min - 2, y_min - 2)], chunk.tiles[tile].field.yvel0[FTNREF2D(*jldt + 1, *kldt + 1, x_max + 5, x_min - 2, y_min - 2)]);
        printf("%f %f \n", chunk.tiles[tile].field.xvel0[FTNREF2D(*jldt  , *kldt + 1, x_max + 5, x_min - 2, y_min - 2)], chunk.tiles[tile].field.yvel0[FTNREF2D(*jldt  , *kldt + 1, x_max + 5, x_min - 2, y_min - 2)]);
        printf("density, energy, pressure, soundspeed \n");
        printf("%f %f %f %f \n",
               chunk.tiles[tile].field.density0[FTNREF2D(*jldt, *kldt, x_max + 4, x_min - 2, y_min - 2)],
               chunk.tiles[tile].field.energy0[FTNREF2D(*jldt, *kldt, x_max + 4, x_min - 2, y_min - 2)],
               chunk.tiles[tile].field.pressure[FTNREF2D(*jldt, *kldt, x_max + 4, x_min - 2, y_min - 2)],
               chunk.tiles[tile].field.soundspeed[FTNREF2D(*jldt, *kldt, x_max + 4, x_min - 2, y_min - 2)]);
    }
    if (l_control ==  1) strcpy(local_control, "sound");
    if (l_control ==  2) strcpy(local_control, "xvel");
    if (l_control ==  3) strcpy(local_control, "yvel");
    if (l_control ==  4) strcpy(local_control, "div");
}
