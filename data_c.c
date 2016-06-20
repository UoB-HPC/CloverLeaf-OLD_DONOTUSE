#ifndef DATA_C_C
#define DATA_C_C

#include <stdbool.h>
#include <stdio.h>
#include "data_c.h"

double g_version = 1.3;

int g_ibig = 640000;

double g_small = 1.0e-16,
       g_big  = 1.0e+21;

int g_name_len_max = 255,
    g_xdir = 1,
    g_ydir = 2;

// These two need to be kept consistent with update_halo
int CHUNK_LEFT   = 1,
    CHUNK_RIGHT  = 2,
    CHUNK_BOTTOM = 3,
    CHUNK_TOP    = 4,
    EXTERNAL_FACE = -1;

int TILE_LEFT   = 1,
    TILE_RIGHT  = 2,
    TILE_BOTTOM = 3,
    TILE_TOP    = 4,
    EXTERNAL_TILE = -1;


int FIELD_DENSITY0   = 1,
    FIELD_DENSITY1   = 2,
    FIELD_ENERGY0    = 3,
    FIELD_ENERGY1    = 4,
    FIELD_PRESSURE   = 5,
    FIELD_VISCOSITY  = 6,
    FIELD_SOUNDSPEED = 7,
    FIELD_XVEL0      = 8,
    FIELD_XVEL1      = 9,
    FIELD_YVEL0      = 10,
    FIELD_YVEL1      = 11,
    FIELD_VOL_FLUX_X = 12,
    FIELD_VOL_FLUX_Y = 13,
    FIELD_MASS_FLUX_X = 14,
    FIELD_MASS_FLUX_Y = 15,
    NUM_FIELDS       = 15;

int CELL_DATA     = 1,
    VERTEX_DATA   = 2,
    X_FACE_DATA   = 3,
    y_FACE_DATA   = 4;


// Time step control constants
int SOUND = 1,
    X_VEL = 2,
    Y_VEL = 3,
    DIVERG = 4;

int g_rect = 1,
    g_circ = 2,
    g_point = 3;


FILE *g_in, // File for input data.
     *g_out;



struct parallel_t parallel;

int g_len_max = 500;

#endif
