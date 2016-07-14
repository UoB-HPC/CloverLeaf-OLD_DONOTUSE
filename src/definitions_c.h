#ifndef DEFINITIONS_C_H
#define DEFINITIONS_C_H

#include <stdbool.h>
#include "data_c.h"
// #include <functional>

#ifdef USE_KOKKOS

#include <Kokkos_Core.hpp>

#endif


#ifdef USE_KOKKOS
#include "kokkosdefs.h"
#else
#include "openmpdefs.h"
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
    double* left_rcv_buffer, *right_rcv_buffer, *bottom_rcv_buffer, *top_rcv_buffer,
            *left_snd_buffer, *right_snd_buffer, *bottom_snd_buffer, *top_snd_buffer;
    struct tile_type* tiles;
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


extern struct state_type* states;

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


#define FTNREF2D(i_index, j_index, i_size, i_lb, j_lb) \
    ((i_size) * (j_index - (j_lb)) + \
        (i_index) - (i_lb))

#define T1ACCESS(d, i, j)         d[FTNREF2D(i, j, x_max + 4, x_min - 2, y_min - 2)]
#define T2ACCESS(d, i, j)         d[FTNREF2D(i, j, x_max + 5, x_min - 2, y_min - 2)]

#define VOLUME(d, i, j)        T1ACCESS(d, i, j)
#define XAREA(d, i, j)         T2ACCESS(d, i, j)
#define YAREA(d, i, j)         T1ACCESS(d, i, j)


#endif
