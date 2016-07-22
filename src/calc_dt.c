#include "calc_dt.h"
#include "definitions_c.h"
#include "kernels/calc_dt_kernel_c.c"
#include "string.h"

void calc_dt(int tile,
             double* local_dt,
             char* local_control,
             double* xl_pos,
             double* yl_pos,
             int* jldt,
             int* kldt)
{
    // *local_dt = g_big;
    int l_control;
    // double lmin = g_big;

    #pragma omp parallel
    {
        DOUBLEFOR(
            chunk.tiles[tile].t_ymin,
            chunk.tiles[tile].t_ymax,
            chunk.tiles[tile].t_xmin,
        chunk.tiles[tile].t_xmax, {
            calc_dt_kernel_c_(
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
                chunk.tiles[tile].field.yvel0,
                chunk.tiles[tile].field.work_array1
            );
        });
    }
    calc_dt_min_val(chunk.tiles[tile].t_xmin,
                    chunk.tiles[tile].t_xmax,
                    chunk.tiles[tile].t_ymin,
                    chunk.tiles[tile].t_ymax,
                    chunk.tiles[tile].field.work_array1,
                    local_dt);
    // *local_dt = mindt;


    double jk_control = 1.1;
    // Extract the mimimum timestep information
    // dtl_control = 10.01 * (jk_control - (int)(jk_control));
    jk_control = jk_control - (jk_control - (int)(jk_control));
    *jldt = 1; //MOD(INT(jk_control),x_max)
    *kldt = 1; //1+(jk_control/x_max)
    //xl_pos=cellx[FTNREF1D(jldt,xmin-2)];
    //yl_pos=celly[FTNREF1D(jldt,ymin-2)];


    l_control = 1;
    int x_min = chunk.tiles[tile].t_xmin,
        x_max = chunk.tiles[tile].t_xmax,
        y_min = chunk.tiles[tile].t_ymin;
    // y_max = chunk.tiles[tile].t_ymax;

    if (*local_dt < dtmin) {
        printf("Timestep information:\n");
        printf("j, k                 :%i %i \n", *jldt, *kldt);
        printf("x, y                 :%f %f \n", *xl_pos, *yl_pos);
        printf("timestep : %f\n", *local_dt);
        printf("Cell velocities;\n");
        printf("%f %f \n", XVEL0(chunk.tiles[tile].field.xvel0, *jldt  ,  *kldt), YVEL0(chunk.tiles[tile].field.yvel0, *jldt  ,  *kldt));
        printf("%f %f \n", XVEL0(chunk.tiles[tile].field.xvel0, *jldt + 1,  *kldt), YVEL0(chunk.tiles[tile].field.yvel0, *jldt + 1,  *kldt));
        printf("%f %f \n", XVEL0(chunk.tiles[tile].field.xvel0, *jldt + 1,  *kldt + 1), YVEL0(chunk.tiles[tile].field.yvel0, *jldt + 1,  *kldt + 1));
        printf("%f %f \n", XVEL0(chunk.tiles[tile].field.xvel0, *jldt  ,  *kldt + 1), YVEL0(chunk.tiles[tile].field.yvel0, *jldt  ,  *kldt + 1));
        printf("density, energy, pressure, soundspeed \n");
        printf("%f %f %f %f \n",
               DENSITY0(chunk.tiles[tile].field.density0, *jldt,  *kldt),
               ENERGY0(chunk.tiles[tile].field.energy0, *jldt,  *kldt),
               PRESSURE(chunk.tiles[tile].field.pressure, *jldt,  *kldt),
               SOUNDSPEED(chunk.tiles[tile].field.soundspeed, *jldt,  *kldt));
    }
    if (l_control ==  1) strcpy(local_control, "sound");
    if (l_control ==  2) strcpy(local_control, "xvel");
    if (l_control ==  3) strcpy(local_control, "yvel");
    if (l_control ==  4) strcpy(local_control, "div");
}
