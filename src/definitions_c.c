#ifndef DEFINITIONS_C_C
#define DEFINITIONS_C_C

#include "data_c.h"
#include <stdbool.h>
#include "definitions_c.h"

struct state_type *states;

int number_of_states;

int step;

bool advect_x;


int tiles_per_chunk;

int error_condition;

int test_problem;
bool complete;

bool use_fortran_kernels;
bool use_C_kernels;
bool use_OA_kernels;

bool profiler_on; // Internal code profiler to make comparisons across systems easier

struct profiler_type profiler;

double end_time;
int end_step;

double dtold,
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
int visit_frequency,
    summary_frequency;

int jdt, kdt;

struct chunk_type chunk;

int number_of_chunks;

struct grid_type grid;



#endif
