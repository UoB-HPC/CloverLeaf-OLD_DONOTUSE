#ifndef DEFINITIONS_C_H
#define DEFINITIONS_C_H

#include <stdbool.h>
#include "data_c.h"
// #include <functional>

#ifdef USE_KOKKOS
#include <Kokkos_Core.hpp>
#endif

struct state_type {
    bool defined;
    double density,
           energy,
           xvel,
           yvel;
    int geometry;
    double xmin,
           xmax,
           ymin,
           ymax,
           radius;
};

struct grid_type {
    double xmin,
           ymin,
           xmax,
           ymax;
    int x_cells,
        y_cells;
};

struct profiler_type {
    double timestep           ,
           acceleration       ,
           PdV                ,
           cell_advection     ,
           mom_advection      ,
           viscosity          ,
           ideal_gas          ,
           visit              ,
           summary            ,
           reset              ,
           revert             ,
           flux               ,
           tile_halo_exchange ,
           self_halo_exchange ,
           mpi_halo_exchange;
};

struct field_type {
    double * density0; double * density1;
    double * energy0; double * energy1;
    double * pressure;
    double * viscosity;
    double * soundspeed;
    double * xvel0; double * xvel1;
    double * yvel0; double * yvel1;
    double * vol_flux_x; double * mass_flux_x;
    double * vol_flux_y; double * mass_flux_y;
    //node_flux; stepbymass; volume_change; pre_vo
    double * work_array1;
    //node_mass_post; post_vol
    double * work_array2;
    //node_mass_pre; pre_mass
    double * work_array3;
    //advec_vel; post_mass
    double * work_array4;
    //mom_flux; advec_vol
    double * work_array5;
    //pre_vol; post_ener
    double * work_array6;
    //post_vol; ener_flux
    double * work_array7;
    double * cellx;
    double * celly;
    double * vertexx;
    double * vertexy;
    double * celldx;
    double * celldy;
    double * vertexdx;
    double * vertexdy;
    double * volume;
    double * xarea;
    double * yarea;
};

struct tile_type {
    struct field_type field;
    int tile_neighbours[4];
    int external_tile_mask[4];
    int t_xmin, t_xmax, t_ymin, t_ymax;
    int t_left, t_right, t_bottom, t_top;
};


struct chunk_type {
    int task;
    int chunk_neighbours[4];
    double *left_rcv_buffer, *right_rcv_buffer, *bottom_rcv_buffer, *top_rcv_buffer,
           *left_snd_buffer, *right_snd_buffer, *bottom_snd_buffer, *top_snd_buffer;
    struct tile_type *tiles;
    int x_min,
        y_min,
        x_max,
        y_max;

    int left,
        right,
        bottom,
        top,
        left_boundary,
        right_boundary,
        bottom_boundary,
        top_boundary;
};

extern struct state_type *states;

extern int number_of_states;

extern int step;

extern bool advect_x;


extern int tiles_per_chunk;

extern int error_condition;

extern int test_problem;
extern bool complete;

extern bool use_fortran_kernels;
extern bool use_C_kernels;
extern bool use_OA_kernels;

extern bool profiler_on; // Internal code profiler to make comparisons across systems easier

extern struct profiler_type profiler;
extern double end_time;
extern int end_step;

extern double dtold,
       dt,
       _time,
       dtinit,
       dtmin,
       dtmax,
       dtrise,
       dtu_safe,
       dtv_safe,
       dtc_safe,
       dtdiv_safe,
       dtc,
       dtu,
       dtv,
       dtdiv;
extern int visit_frequency,
       summary_frequency;

extern int jdt, kdt;
extern struct chunk_type chunk;

extern int number_of_chunks;

extern struct grid_type grid;

#define BOSSPRINT(...) if(parallel.boss) fprintf(__VA_ARGS__)


#ifdef USE_KOKKOS

#define DOUBLEFOR(k_from, k_to, j_from, j_to, body) \
    Kokkos::parallel_for((k_to) - (k_from) + 1, KOKKOS_LAMBDA (const int& i) { \
            int k = i + (k_from); \
        _Pragma("ivdep") \
        for(int j = (j_from); j <= (j_to); j++) { \
            body ;\
        } \
    });

#else

#define DOUBLEFOR(k_from, k_to, j_from, j_to, body) \
    _Pragma("omp for") \
    for(int k = (k_from); k <= (k_to); k++) { \
        _Pragma("ivdep") \
        for(int j = (j_from); j <= (j_to); j++) { \
            body ;\
        } \
    }

#endif

#endif
