#ifndef DATA_C_C
#define DATA_C_C

#include <stdbool.h>
#include <stdio.h>
#include "data_c.h"

double g_version = 1.4;

int g_ibig = 640000;

double g_small = 1.0e-16,
       g_big  = 1.0e+21;

int g_name_len_max = 255,
    g_xdir = 1,
    g_ydir = 2;


// Time step control constants
int SOUND = 1,
    X_VEL = 2,
    Y_VEL = 3,
    DIVERG = 4;

int g_rect = 1,
    g_circ = 2,
    g_point = 3;


FILE* g_in, // File for input data.
      *g_out;

int CELL_DATA   = 1,
    VERTEX_DATA = 2,
    X_FACE_DATA = 3,
    Y_FACE_DATA = 4;



struct parallel_t parallel;

int g_len_max = 500;

#endif
