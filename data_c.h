#ifndef DATA_C_H
#define DATA_C_H

#include <stdbool.h>

extern double g_version;

extern int g_ibig ;

extern double g_small,
       g_big;

extern int g_name_len_max ,
       g_xdir ,
       g_ydir ;

// These two need to be kept consistent with update_halo
extern int CHUNK_LEFT   ,
       CHUNK_RIGHT  ,
       CHUNK_BOTTOM ,
       CHUNK_TOP    ,
       EXTERNAL_FACE;

extern int TILE_LEFT   ,
       TILE_RIGHT  ,
       TILE_BOTTOM ,
       TILE_TOP    ,
       EXTERNAL_TILE;


extern int FIELD_DENSITY0   ,
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
       NUM_FIELDS       ;

extern int CELL_DATA     ,
       VERTEX_DATA   ,
       X_FACE_DATA   ,
       y_FACE_DATA   ;


// Time step control constants
extern int SOUND ,
       X_VEL ,
       Y_VEL ,
       DIVERG ;

extern int g_rect ,
       g_circ ,
       g_point ;


extern int g_in, // File for input data.
       g_out;

extern struct parallel_t parallel;

extern int g_len_max ;

#endif
