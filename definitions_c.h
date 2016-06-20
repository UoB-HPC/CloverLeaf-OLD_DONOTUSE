#ifndef DEFINITIONS_C_H
#define DEFINITIONS_C_H

#include <stdbool.h>

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
       time,
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

#endif