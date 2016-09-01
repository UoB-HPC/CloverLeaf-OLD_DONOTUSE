#ifndef DATA_C_H
#define DATA_C_H

#include <stdbool.h>
#include <stdio.h>

extern double g_version;

extern int g_ibig ;

extern double g_small,
       g_big;

extern int g_name_len_max ,
       g_xdir ,
       g_ydir ;

// These two need to be kept consistent with update_halo
enum chunk_enum {
    CHUNK_LEFT = 0  ,
    CHUNK_RIGHT  ,
    CHUNK_BOTTOM ,
    CHUNK_TOP    ,
    EXTERNAL_FACE = -1
};

enum tile_enum {
    TILE_LEFT = 0,
    TILE_RIGHT  ,
    TILE_BOTTOM ,
    TILE_TOP    ,
    EXTERNAL_TILE = -1
};

enum field_enum {
    FIELD_DENSITY0 = 0,
    FIELD_DENSITY1   ,
    FIELD_ENERGY0    ,
    FIELD_ENERGY1    ,
    FIELD_PRESSURE   ,
    FIELD_VISCOSITY  ,
    FIELD_SOUNDSPEED ,
    FIELD_XVEL0      ,
    FIELD_XVEL1      ,
    FIELD_YVEL0      ,
    FIELD_YVEL1      ,
    FIELD_VOL_FLUX_X ,
    FIELD_VOL_FLUX_Y ,
    FIELD_MASS_FLUX_X ,
    FIELD_MASS_FLUX_Y ,
    NUM_FIELDS       ,
};

extern int CELL_DATA     ,
       VERTEX_DATA   ,
       X_FACE_DATA   ,
       Y_FACE_DATA   ;

// enum some_enum {
//       CELL_DATA     ,
//       VERTEX_DATA   ,
//       X_FACE_DATA   ,
//       Y_FACE_DATA
// };

// Time step control constants
extern int SOUND ,
       X_VEL ,
       Y_VEL ,
       DIVERG ;

extern int g_rect ,
       g_circ ,
       g_point ;


extern FILE* g_in, // File for input data.
       *g_out;

struct parallel_t {
    bool parallel,
         boss;
    int max_task,
        task,
        boss_task;
};

extern struct parallel_t parallel;

extern int g_len_max ;


#endif
