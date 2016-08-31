/*Crown Copyright 2012 AWE.
*
* This file is part of CloverLeaf.
*
* CloverLeaf is free software: you can redistribute it and/or modify it under
* the terms of the GNU General Public License as published by the
* Free Software Foundation, either version 3 of the License, or (at your option)
* any later version.
*
* CloverLeaf is distributed in the hope that it will be useful, but
* WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
* details.
*
* You should have received a copy of the GNU General Public License along with
* CloverLeaf. If not, see http://www.gnu.org/licenses/. */

/**
 *@brief C mpi buffer packing kernel
 *@author Wayne Gaudin
 *@details Packs/unpacks mpi send and receive buffers
 */

#include <stdio.h>
#include <stdlib.h>
#include "ftocmacros.h"
#include <math.h>
#include "../definitions_c.h"

kernelqual void clover_pack_message_left_c_(
    int x_min, int x_max,
    int y_min, int y_max,
    field_2d_t field,
    double* __restrict__ left_snd_buffer,
    int depth, int field_type,
    int buffer_offset)
{
    int index, x_inc, y_inc;

//Pack
    for (int k = y_min - depth; k <= y_max + y_inc + depth; k++) {
#pragma ivdep
        for (int j = 1; j <= depth; j++) {
            index = buffer_offset + j + (k + depth - 1) * depth;
            if (field_type == CELL_DATA) {
                left_snd_buffer[FTNREF1D(index, 1)] = T1ACCESS(field, x_min - 1 + j, k);
            }
            if (field_type == VERTEX_DATA) {
                left_snd_buffer[FTNREF1D(index, 1)] = T2ACCESS(field, x_min + 1 - 1 + j, k);
            }
            if (field_type == X_FACE_DATA) {
                left_snd_buffer[FTNREF1D(index, 1)] = T2ACCESS(field, x_min + 1 - 1 + j, k);
            }
            if (field_type == Y_FACE_DATA) {
                left_snd_buffer[FTNREF1D(index, 1)] = T1ACCESS(field, x_min - 1 + j, k);
            }
        }
    }
}

kernelqual void clover_unpack_message_left_c_(
    int x_min, int x_max,
    int y_min, int y_max,
    double* field,
    double* left_rcv_buffer,
    int depth, int field_type,
    int buffer_offset)
{
    // int x_min = *xmin;
    // int x_max = *xmax;
    // int y_min = *ymin;
    // int y_max = *ymax;
    // int field_type = fld_typ;
    // int depth = dpth;
    // int buffer_offset = bffr_ffst;

    int index, x_inc, y_inc;

    //Unpack
    for (int k = y_min - depth; k <= y_max + y_inc + depth; k++) {
#pragma ivdep
        for (int j = 1; j <= depth; j++) {
            index = buffer_offset + j + (k + depth - 1) * depth;
            if (field_type == CELL_DATA) {
                T1ACCESS(field, x_min - j, k) = left_rcv_buffer[FTNREF1D(index, 1)];
            }
            if (field_type == VERTEX_DATA) {
                T2ACCESS(field, x_min - j, k) = left_rcv_buffer[FTNREF1D(index, 1)];
            }
            if (field_type == X_FACE_DATA) {
                T2ACCESS(field, x_min - j, k) = left_rcv_buffer[FTNREF1D(index, 1)];
            }
            if (field_type == Y_FACE_DATA) {
                T1ACCESS(field, x_min - j, k) = left_rcv_buffer[FTNREF1D(index, 1)];
            }
        }
    }
}

kernelqual void clover_pack_message_right_c_(
    int x_min, int x_max,
    int y_min, int y_max,
    double* field,
    double* right_snd_buffer,
    int depth, int field_type,
    int buffer_offset)
{
    // int x_min = *xmin;
    // int x_max = *xmax;
    // int y_min = *ymin;
    // int y_max = *ymax;
    // int field_type = fld_typ;
    // int depth = dpth;
    // int buffer_offset = bffr_ffst;

    int index, x_inc, y_inc;

    //Pack
    for (int k = y_min - depth; k <= y_max + y_inc + depth; k++) {
#pragma ivdep
        for (int j = 1; j <= depth; j++) {
            index = buffer_offset + j + (k + depth - 1) * depth;
            if (field_type == CELL_DATA) {
                right_snd_buffer[FTNREF1D(index, 1)] = T1ACCESS(field, x_max + 1 - j, k);
            }
            if (field_type == VERTEX_DATA) {
                right_snd_buffer[FTNREF1D(index, 1)] = T2ACCESS(field, x_max + 1 - j, k);
            }
            if (field_type == X_FACE_DATA) {
                right_snd_buffer[FTNREF1D(index, 1)] = T2ACCESS(field, x_max + 1 - j, k);
            }
            if (field_type == Y_FACE_DATA) {
                right_snd_buffer[FTNREF1D(index, 1)] = T1ACCESS(field, x_max + 1 - j, k);
            }
        }
    }
}

kernelqual void clover_unpack_message_right_c_(
    int x_min, int x_max,
    int y_min, int y_max,
    double* field,
    double* right_rcv_buffer,
    int depth, int field_type,
    int buffer_offset)
{
    // int x_min = *xmin;
    // int x_max = *xmax;
    // int y_min = *ymin;
    // int y_max = *ymax;
    // int field_type = fld_typ;
    // int depth = dpth;
    // int buffer_offset = bffr_ffst;

    int index, x_inc, y_inc;

//Pack
    for (int k = y_min - depth; k <= y_max + y_inc + depth; k++) {
#pragma ivdep
        for (int j = 1; j <= depth; j++) {
            index = buffer_offset + j + (k + depth - 1) * depth;
            if (field_type == CELL_DATA) {
                T1ACCESS(field, x_max + j, k) = right_rcv_buffer[FTNREF1D(index, 1)];
            }
            if (field_type == VERTEX_DATA) {
                T2ACCESS(field, x_max + j + 1, k) = right_rcv_buffer[FTNREF1D(index, 1)];
            }
            if (field_type == X_FACE_DATA) {
                T2ACCESS(field, x_max + j + 1, k) = right_rcv_buffer[FTNREF1D(index, 1)];
            }
            if (field_type == Y_FACE_DATA) {
                T1ACCESS(field, x_max + j, k) = right_rcv_buffer[FTNREF1D(index, 1)];
            }
        }
    }
}

kernelqual void clover_pack_message_top_c_(
    int x_min, int x_max,
    int y_min, int y_max,
    double* field,
    double* top_snd_buffer,
    int depth, int field_type,
    int buffer_offset)
{
    // int x_min = *xmin;
    // int x_max = *xmax;
    // int y_min = *ymin;
    // int y_max = *ymax;
    // int field_type = fld_typ;
    // int depth = dpth;
    // int buffer_offset = bffr_ffst;

    int index, x_inc;
// y_inc;

//Pack
    for (int k = 1; k <= depth; k++) {
#pragma ivdep
        for (int j = x_min - depth; j <= x_max + x_inc + depth; j++) {
            index = buffer_offset + j + (k + depth - 1) * depth;
            if (field_type == CELL_DATA) {
                // T1ACCESS(field, x_max + j, k) = right_rcv_buffer[FTNREF1D(index, 1)];
                top_snd_buffer[FTNREF1D(index, 1)] = T1ACCESS(field, j, y_max + 1 - k);
            }
            if (field_type == VERTEX_DATA) {
                // T2ACCESS(field, x_max + j + 1, k) = right_rcv_buffer[FTNREF1D(index, 1)];
                top_snd_buffer[FTNREF1D(index, 1)] = T2ACCESS(field, j, y_max + 1 - k);
            }
            if (field_type == X_FACE_DATA) {
                // T2ACCESS(field, x_max + j + 1, k) = right_rcv_buffer[FTNREF1D(index, 1)];
                top_snd_buffer[FTNREF1D(index, 1)] = T2ACCESS(field, j, y_max + 1 - k);
            }
            if (field_type == Y_FACE_DATA) {
                // T1ACCESS(field, x_max + j, k) = right_rcv_buffer[FTNREF1D(index, 1)];
                top_snd_buffer[FTNREF1D(index, 1)] = T1ACCESS(field, j, y_max + 1 - k);
            }
        }
    }

}

kernelqual void clover_pack_message_bottom_c_(
    int x_min, int x_max,
    int y_min, int y_max,
    double* field,
    double* bottom_snd_buffer,
    int depth, int field_type,
    int buffer_offset)
{
//     int x_min = *xmin;
//     int x_max = *xmax;
//     int y_min = *ymin;
// // int y_max = *ymax;
//     int field_type = fld_typ;
//     int depth = dpth;
//     int buffer_offset = bffr_ffst;

    int j, k, index, x_inc, y_inc;

//Pack
    for (int k = 1; k <= depth; k++) {
#pragma ivdep
        for (int j = x_min - depth; j <= x_max + x_inc + depth; j++) {
            index = buffer_offset + j + (k + depth - 1) * depth;
            if (field_type == CELL_DATA) {
                // T1ACCESS(field, x_max + j, k) = right_rcv_buffer[FTNREF1D(index, 1)];
                bottom_snd_buffer[FTNREF1D(index, 1)] = T1ACCESS(field, j, y_min - 1 + k);
            }
            if (field_type == VERTEX_DATA) {
                // T2ACCESS(field, x_max + j + 1, k) = right_rcv_buffer[FTNREF1D(index, 1)];
                bottom_snd_buffer[FTNREF1D(index, 1)] = T2ACCESS(field, j, y_min + 1 - 1 + k);
            }
            if (field_type == X_FACE_DATA) {
                // T2ACCESS(field, x_max + j + 1, k) = right_rcv_buffer[FTNREF1D(index, 1)];
                bottom_snd_buffer[FTNREF1D(index, 1)] = T2ACCESS(field, j, y_min - 1 + k);
            }
            if (field_type == Y_FACE_DATA) {
                // T1ACCESS(field, x_max + j, k) = right_rcv_buffer[FTNREF1D(index, 1)];
                bottom_snd_buffer[FTNREF1D(index, 1)] = T1ACCESS(field, j, y_min + 1 - 1 + k);
            }
        }
    }
}

kernelqual void clover_unpack_message_bottom_c_(
    int x_min, int x_max,
    int y_min, int y_max,
    double* field,
    double* bottom_rcv_buffer,
    int depth, int field_type,
    int buffer_offset)
{
//     int x_min = *xmin;
//     int x_max = *xmax;
//     int y_min = *ymin;
// // int y_max = *ymax;
//     int field_type = fld_typ;
//     int depth = dpth;
//     int buffer_offset = bffr_ffst;

    int index, x_inc;

//Unpack
    for (int k = 1; k <= depth; k++) {
#pragma ivdep
        for (int j = x_min - depth; j <= x_max + x_inc + depth; j++) {
            index = buffer_offset + j + (k + depth - 1) * depth;
            if (field_type == CELL_DATA) {
                T1ACCESS(field, j, y_min - k) = bottom_rcv_buffer[FTNREF1D(index, 1)];
            }
            if (field_type == VERTEX_DATA) {
                T2ACCESS(field, j, y_min - k) = bottom_rcv_buffer[FTNREF1D(index, 1)];
            }
            if (field_type == X_FACE_DATA) {
                T2ACCESS(field, j, y_min - k) = bottom_rcv_buffer[FTNREF1D(index, 1)];
            }
            if (field_type == Y_FACE_DATA) {
                T1ACCESS(field, j, y_min - k) = bottom_rcv_buffer[FTNREF1D(index, 1)];
            }
        }
    }
}

kernelqual void clover_unpack_message_top_c_(
    int x_min, int x_max, int y_min, int y_max,
    double* field,
    double* top_rcv_buffer,
    int depth, int field_type,
    int buffer_offset)

{
    int index, x_inc, y_inc;

//Unpack
    for (int k = 1; k <= depth; k++) {
#pragma ivdep
        for (int j = x_min - depth; j <= x_max + x_inc + depth; j++) {
            index = buffer_offset + j + (k + depth - 1) * depth;
            if (field_type == CELL_DATA) {
                T1ACCESS(field, j, y_max + 0 + k) = top_rcv_buffer[FTNREF1D(index, 1)];
            }
            if (field_type == VERTEX_DATA) {
                T2ACCESS(field, j, y_max + 1 + k) = top_rcv_buffer[FTNREF1D(index, 1)];
            }
            if (field_type == X_FACE_DATA) {
                T2ACCESS(field, j, y_max + 0 + k) = top_rcv_buffer[FTNREF1D(index, 1)];
            }
            if (field_type == Y_FACE_DATA) {
                T1ACCESS(field, j, y_max + 1 + k) = top_rcv_buffer[FTNREF1D(index, 1)];
            }
        }
    }
}
