#ifndef DEFINITIONS_C_H
#define DEFINITIONS_C_H

#include <stdbool.h>
#include "data_c.h"

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
  double *density0, *density1,
         *energy0,  *energy1,
         *pressure,
         *viscosity,
         *soundspeed,
         *xvel0, *xvel1,
         *yvel0, *yvel1,
         *vol_flux_x, *mass_flux_x,
         *vol_flux_y, *mass_flux_y,
         *work_array1, //node_flux, stepbymass, volume_change, pre_vo
         *work_array2, //node_mass_post, post_vol
         *work_array3, //node_mass_pre, pre_mass
         *work_array4, //advec_vel, post_mass
         *work_array5, //mom_flux, advec_vol
         *work_array6, //pre_vol, post_ener
         *work_array7; //post_vol, ener_flux
  double *cellx,
         *celly,
         *vertexx,
         *vertexy,
         *celldx,
         *celldy,
         *vertexdx,
         *vertexdy;
  double *volume,
         *xarea,
         *yarea;
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

#endif
