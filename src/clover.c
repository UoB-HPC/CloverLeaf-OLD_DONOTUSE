#include "clover.h"
#include <mpi.h>
#include "definitions_c.h"
#include "kernels/pack_kernel_c.c"
// #ifdef USE_KOKKOS
// #include "kernels/pack_kernel_kokkos.cpp"
// #endif

void checkMPIerror(int err)
{
    if (err != MPI_SUCCESS) {

        char error_string[200];
        int length_of_error_string;

        MPI_Error_string(err, error_string, &length_of_error_string);

        exit(1);
    }
}
void clover_barrier()
{
    checkMPIerror(MPI_Barrier(MPI_COMM_WORLD));
}



void clover_abort()
{
    checkMPIerror(MPI_Abort(MPI_COMM_WORLD, 4));
}

void clover_finalize()
{
    checkMPIerror(MPI_Finalize());

}


void clover_init_comms(int argc, char** argv)
{
    int rank = 0,
        size = 1;

    checkMPIerror(MPI_Init(&argc, &argv));

    checkMPIerror(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    checkMPIerror(MPI_Comm_size(MPI_COMM_WORLD, &size));

    parallel.parallel = true;
    parallel.task = rank;

    if (rank == 0)
        parallel.boss = true;

    parallel.boss_task = 0;
    parallel.max_task = size;
}

void clover_get_num_chunks(int* count)
{
    // Should be changed so there can be more than one chunk per mpi task
    *count = parallel.max_task;

}

void clover_exchange(int* fields, int depth)
{
    MPI_Status status[4];

    // Assuming 1 patch per task, this will be changed
    MPI_Request request[4] = {0, 0, 0, 0};
    int message_count = 0;
    int left_right_offset[NUM_FIELDS],
        bottom_top_offset[NUM_FIELDS];

    int end_pack_index_left_right = 0;
    int end_pack_index_bottom_top = 0;

    for (int  field = 0; field < NUM_FIELDS; field++) {
        if (fields[field] == 1)  {
            left_right_offset[field] = end_pack_index_left_right;
            bottom_top_offset[field] = end_pack_index_bottom_top;
            end_pack_index_left_right = end_pack_index_left_right + depth * (chunk.y_max + 5);
            end_pack_index_bottom_top = end_pack_index_bottom_top + depth * (chunk.x_max + 5);
        }
    }

    if (chunk.chunk_neighbours[CHUNK_LEFT] != EXTERNAL_FACE)  {
        // do left exchanges
        // Find left hand tiles
        for (int  tile = 0; tile < tiles_per_chunk; tile++) {
            if (chunk.tiles[tile].external_tile_mask[TILE_LEFT] == 1)  {
                clover_pack_left(tile, fields, depth, left_right_offset);
            }
        }

        //send and recv messagse to the left
        clover_send_recv_message_left(chunk.left_snd_buffer,
                                      chunk.left_rcv_buffer,
                                      end_pack_index_left_right,
                                      1, 2,
                                      &request[message_count], &request[message_count + 1]);
        message_count = message_count + 2;
    }

    if (chunk.chunk_neighbours[CHUNK_RIGHT] != EXTERNAL_FACE)  {
        // do right exchanges
        for (int  tile = 0; tile < tiles_per_chunk; tile++) {
            if (chunk.tiles[tile].external_tile_mask[TILE_RIGHT] == 1)  {
                clover_pack_right(tile, fields, depth, left_right_offset);
            }
        }

        //send message to the right
        clover_send_recv_message_right(chunk.right_snd_buffer,
                                       chunk.right_rcv_buffer,
                                       end_pack_index_left_right,
                                       2, 1,
                                       &request[message_count], &request[message_count + 1]);
        message_count = message_count + 2;
    }

    //make a call to wait / sync
    checkMPIerror(MPI_Waitall(message_count, request, status));

    //unpack in left direction
    if (chunk.chunk_neighbours[CHUNK_LEFT] != EXTERNAL_FACE)  {
        for (int  tile = 0; tile < tiles_per_chunk; tile++) {
            if (chunk.tiles[tile].external_tile_mask[TILE_LEFT] == 1)  {
                clover_unpack_left(fields, tile, depth,
                                   chunk.left_rcv_buffer,
                                   left_right_offset);
            }
        }
    }


    //unpack in right direction
    if (chunk.chunk_neighbours[CHUNK_RIGHT] != EXTERNAL_FACE)  {
        for (int  tile = 0; tile < tiles_per_chunk; tile++) {
            if (chunk.tiles[tile].external_tile_mask[TILE_RIGHT] == 1)  {
                clover_unpack_right(fields, tile, depth,
                                    chunk.right_rcv_buffer,
                                    left_right_offset);
            }
        }
    }

    message_count = 0;
    request[0] = request[1] = request[2] = request[3] = 0;

    if (chunk.chunk_neighbours[CHUNK_BOTTOM] != EXTERNAL_FACE)  {
        // do bottom exchanges
        for (int  tile = 0; tile < tiles_per_chunk; tile++) {
            if (chunk.tiles[tile].external_tile_mask[TILE_BOTTOM] == 1)  {
                clover_pack_bottom(tile, fields, depth, bottom_top_offset);
            }
        }

        //send message downwards
        clover_send_recv_message_bottom(chunk.bottom_snd_buffer,
                                        chunk.bottom_rcv_buffer,
                                        end_pack_index_bottom_top,
                                        3, 4,
                                        &request[message_count], &request[message_count + 1]);
        message_count = message_count + 2;
    }

    if (chunk.chunk_neighbours[CHUNK_TOP] != EXTERNAL_FACE)  {
        // do top exchanges
        for (int  tile = 0; tile < tiles_per_chunk; tile++) {
            if (chunk.tiles[tile].external_tile_mask[TILE_TOP] == 1)  {
                clover_pack_top(tile, fields, depth, bottom_top_offset);
            }
        }

        //send message upwards
        clover_send_recv_message_top(chunk.top_snd_buffer,
                                     chunk.top_rcv_buffer,
                                     end_pack_index_bottom_top,
                                     4, 3,
                                     &request[message_count], &request[message_count + 1]);
        message_count = message_count + 2;
    }

    //need to make a call to wait / sync
    checkMPIerror(MPI_Waitall(message_count, request, status));

    //unpack in top direction
    if (chunk.chunk_neighbours[CHUNK_TOP] != EXTERNAL_FACE)  {
        for (int  tile = 0; tile < tiles_per_chunk; tile++) {
            if (chunk.tiles[tile].external_tile_mask[TILE_TOP] == 1)  {
                clover_unpack_top(fields, tile, depth,
                                  chunk.top_rcv_buffer,
                                  bottom_top_offset);
            }
        }
    }

    //unpack in bottom direction
    if (chunk.chunk_neighbours[CHUNK_BOTTOM] != EXTERNAL_FACE)  {
        for (int  tile = 0; tile < tiles_per_chunk; tile++) {
            if (chunk.tiles[tile].external_tile_mask[TILE_BOTTOM] == 1)  {
                clover_unpack_bottom(fields, tile, depth,
                                     chunk.bottom_rcv_buffer,
                                     bottom_top_offset);
            }
        }
    }
}

void clover_pack_left(int tile, int* fields, int depth, int* left_right_offset)
{

    int t_offset = (chunk.tiles[tile].t_bottom - chunk.bottom) * depth;


    if (fields[FIELD_DENSITY0] == 1)  {
        clover_pack_message_left_c_(chunk.tiles[tile].t_xmin,
                                    chunk.tiles[tile].t_xmax,
                                    chunk.tiles[tile].t_ymin,
                                    chunk.tiles[tile].t_ymax,
                                    chunk.tiles[tile].field.density0,
                                    chunk.left_snd_buffer,
                                    depth, CELL_DATA,
                                    left_right_offset[FIELD_DENSITY0] + t_offset);

    }
    if (fields[FIELD_DENSITY1] == 1)  {
        clover_pack_message_left_c_(chunk.tiles[tile].t_xmin,
                                    chunk.tiles[tile].t_xmax,
                                    chunk.tiles[tile].t_ymin,
                                    chunk.tiles[tile].t_ymax,
                                    chunk.tiles[tile].field.density1,
                                    chunk.left_snd_buffer,
                                    depth, CELL_DATA,
                                    left_right_offset[FIELD_DENSITY1] + t_offset);

    }
    if (fields[FIELD_ENERGY0] == 1)  {
        clover_pack_message_left_c_(chunk.tiles[tile].t_xmin,
                                    chunk.tiles[tile].t_xmax,
                                    chunk.tiles[tile].t_ymin,
                                    chunk.tiles[tile].t_ymax,
                                    chunk.tiles[tile].field.energy0,
                                    chunk.left_snd_buffer,
                                    depth, CELL_DATA,
                                    left_right_offset[FIELD_ENERGY0] + t_offset);

    }
    if (fields[FIELD_ENERGY1] == 1)  {
        clover_pack_message_left_c_(chunk.tiles[tile].t_xmin,
                                    chunk.tiles[tile].t_xmax,
                                    chunk.tiles[tile].t_ymin,
                                    chunk.tiles[tile].t_ymax,
                                    chunk.tiles[tile].field.energy1,
                                    chunk.left_snd_buffer,
                                    depth, CELL_DATA,
                                    left_right_offset[FIELD_ENERGY1] + t_offset);

    }
    if (fields[FIELD_PRESSURE] == 1)  {
        clover_pack_message_left_c_(chunk.tiles[tile].t_xmin,
                                    chunk.tiles[tile].t_xmax,
                                    chunk.tiles[tile].t_ymin,
                                    chunk.tiles[tile].t_ymax,
                                    chunk.tiles[tile].field.pressure,
                                    chunk.left_snd_buffer,
                                    depth, CELL_DATA,
                                    left_right_offset[FIELD_PRESSURE] + t_offset);

    }
    if (fields[FIELD_VISCOSITY] == 1)  {
        clover_pack_message_left_c_(chunk.tiles[tile].t_xmin,
                                    chunk.tiles[tile].t_xmax,
                                    chunk.tiles[tile].t_ymin,
                                    chunk.tiles[tile].t_ymax,
                                    chunk.tiles[tile].field.viscosity,
                                    chunk.left_snd_buffer,
                                    depth, CELL_DATA,
                                    left_right_offset[FIELD_VISCOSITY] + t_offset);

    }
    if (fields[FIELD_SOUNDSPEED] == 1)  {
        clover_pack_message_left_c_(chunk.tiles[tile].t_xmin,
                                    chunk.tiles[tile].t_xmax,
                                    chunk.tiles[tile].t_ymin,
                                    chunk.tiles[tile].t_ymax,
                                    chunk.tiles[tile].field.soundspeed,
                                    chunk.left_snd_buffer,
                                    depth, CELL_DATA,
                                    left_right_offset[FIELD_SOUNDSPEED] + t_offset);

    }
    if (fields[FIELD_XVEL0] == 1)  {
        clover_pack_message_left_c_(chunk.tiles[tile].t_xmin,
                                    chunk.tiles[tile].t_xmax,
                                    chunk.tiles[tile].t_ymin,
                                    chunk.tiles[tile].t_ymax,
                                    chunk.tiles[tile].field.xvel0,
                                    chunk.left_snd_buffer,
                                    depth, VERTEX_DATA,
                                    left_right_offset[FIELD_XVEL0] + t_offset);

    }
    if (fields[FIELD_XVEL1] == 1)  {
        clover_pack_message_left_c_(chunk.tiles[tile].t_xmin,
                                    chunk.tiles[tile].t_xmax,
                                    chunk.tiles[tile].t_ymin,
                                    chunk.tiles[tile].t_ymax,
                                    chunk.tiles[tile].field.xvel1,
                                    chunk.left_snd_buffer,
                                    depth, VERTEX_DATA,
                                    left_right_offset[FIELD_XVEL1] + t_offset);

    }
    if (fields[FIELD_YVEL0] == 1)  {
        clover_pack_message_left_c_(chunk.tiles[tile].t_xmin,
                                    chunk.tiles[tile].t_xmax,
                                    chunk.tiles[tile].t_ymin,
                                    chunk.tiles[tile].t_ymax,
                                    chunk.tiles[tile].field.yvel0,
                                    chunk.left_snd_buffer,
                                    depth, VERTEX_DATA,
                                    left_right_offset[FIELD_YVEL0] + t_offset);

    }
    if (fields[FIELD_YVEL1] == 1)  {
        clover_pack_message_left_c_(chunk.tiles[tile].t_xmin,
                                    chunk.tiles[tile].t_xmax,
                                    chunk.tiles[tile].t_ymin,
                                    chunk.tiles[tile].t_ymax,
                                    chunk.tiles[tile].field.yvel1,
                                    chunk.left_snd_buffer,
                                    depth, VERTEX_DATA,
                                    left_right_offset[FIELD_YVEL1] + t_offset);

    }
    if (fields[FIELD_VOL_FLUX_X] == 1)  {
        clover_pack_message_left_c_(chunk.tiles[tile].t_xmin,
                                    chunk.tiles[tile].t_xmax,
                                    chunk.tiles[tile].t_ymin,
                                    chunk.tiles[tile].t_ymax,
                                    chunk.tiles[tile].field.vol_flux_x,
                                    chunk.left_snd_buffer,
                                    depth, X_FACE_DATA,
                                    left_right_offset[FIELD_VOL_FLUX_X] + t_offset);

    }
    if (fields[FIELD_VOL_FLUX_Y] == 1)  {
        clover_pack_message_left_c_(chunk.tiles[tile].t_xmin,
                                    chunk.tiles[tile].t_xmax,
                                    chunk.tiles[tile].t_ymin,
                                    chunk.tiles[tile].t_ymax,
                                    chunk.tiles[tile].field.vol_flux_y,
                                    chunk.left_snd_buffer,
                                    depth, Y_FACE_DATA,
                                    left_right_offset[FIELD_VOL_FLUX_Y] + t_offset);

    }
    if (fields[FIELD_MASS_FLUX_X] == 1)  {
        clover_pack_message_left_c_(chunk.tiles[tile].t_xmin,
                                    chunk.tiles[tile].t_xmax,
                                    chunk.tiles[tile].t_ymin,
                                    chunk.tiles[tile].t_ymax,
                                    chunk.tiles[tile].field.mass_flux_x,
                                    chunk.left_snd_buffer,
                                    depth, X_FACE_DATA,
                                    left_right_offset[FIELD_MASS_FLUX_X] + t_offset);

    }
    if (fields[FIELD_MASS_FLUX_Y] == 1)  {
        clover_pack_message_left_c_(chunk.tiles[tile].t_xmin,
                                    chunk.tiles[tile].t_xmax,
                                    chunk.tiles[tile].t_ymin,
                                    chunk.tiles[tile].t_ymax,
                                    chunk.tiles[tile].field.mass_flux_y,
                                    chunk.left_snd_buffer,
                                    depth, Y_FACE_DATA,
                                    left_right_offset[FIELD_MASS_FLUX_Y] + t_offset);

    }



}

void clover_send_recv_message_left(double* left_snd_buffer, double* left_rcv_buffer,
                                   int total_size,
                                   int tag_send, int tag_recv,
                                   MPI_Request* req_send, MPI_Request* req_recv)
{

//     REAL(KIND=8)    :: left_snd_buffer(:), left_rcv_buffer(:)
//     INTEGER         :: left_task
//     INTEGER         :: total_size, tag_send, tag_recv, err
//     INTEGER         :: req_send, req_recv

    int left_task = chunk.chunk_neighbours[CHUNK_LEFT] - 1;

    checkMPIerror(MPI_Isend(left_snd_buffer, total_size, MPI_DOUBLE, left_task, tag_send
                            , MPI_COMM_WORLD, req_send));

    checkMPIerror(MPI_Irecv(left_rcv_buffer, total_size, MPI_DOUBLE, left_task, tag_recv
                            , MPI_COMM_WORLD, req_recv));

}

void clover_unpack_left(int* fields, int tile, int depth,
                        double* left_rcv_buffer,
                        int* left_right_offset)
{

//     USE pack_kernel_module

//     IMPLICIT NONE

//     INTEGER         :: fields(:), tile, depth, t_offset
//     INTEGER         :: left_right_offset[:]
//     REAL(KIND=8)    :: left_rcv_buffer(:)

    int t_offset = (chunk.tiles[tile].t_bottom - chunk.bottom) * depth;


    if (fields[FIELD_DENSITY0] == 1)  {
        clover_unpack_message_left_c_(chunk.tiles[tile].t_xmin,
                                      chunk.tiles[tile].t_xmax,
                                      chunk.tiles[tile].t_ymin,
                                      chunk.tiles[tile].t_ymax,
                                      chunk.tiles[tile].field.density0,
                                      chunk.left_rcv_buffer,
                                      depth, CELL_DATA,
                                      left_right_offset[FIELD_DENSITY0] + t_offset);

    }
    if (fields[FIELD_DENSITY1] == 1)  {
        clover_unpack_message_left_c_(chunk.tiles[tile].t_xmin,
                                      chunk.tiles[tile].t_xmax,
                                      chunk.tiles[tile].t_ymin,
                                      chunk.tiles[tile].t_ymax,
                                      chunk.tiles[tile].field.density1,
                                      chunk.left_rcv_buffer,
                                      depth, CELL_DATA,
                                      left_right_offset[FIELD_DENSITY1] + t_offset);

    }
    if (fields[FIELD_ENERGY0] == 1)  {
        clover_unpack_message_left_c_(chunk.tiles[tile].t_xmin,
                                      chunk.tiles[tile].t_xmax,
                                      chunk.tiles[tile].t_ymin,
                                      chunk.tiles[tile].t_ymax,
                                      chunk.tiles[tile].field.energy0,
                                      chunk.left_rcv_buffer,
                                      depth, CELL_DATA,
                                      left_right_offset[FIELD_ENERGY0] + t_offset);

    }
    if (fields[FIELD_ENERGY1] == 1)  {
        clover_unpack_message_left_c_(chunk.tiles[tile].t_xmin,
                                      chunk.tiles[tile].t_xmax,
                                      chunk.tiles[tile].t_ymin,
                                      chunk.tiles[tile].t_ymax,
                                      chunk.tiles[tile].field.energy1,
                                      chunk.left_rcv_buffer,
                                      depth, CELL_DATA,
                                      left_right_offset[FIELD_ENERGY1] + t_offset);

    }
    if (fields[FIELD_PRESSURE] == 1)  {
        clover_unpack_message_left_c_(chunk.tiles[tile].t_xmin,
                                      chunk.tiles[tile].t_xmax,
                                      chunk.tiles[tile].t_ymin,
                                      chunk.tiles[tile].t_ymax,
                                      chunk.tiles[tile].field.pressure,
                                      chunk.left_rcv_buffer,
                                      depth, CELL_DATA,
                                      left_right_offset[FIELD_PRESSURE] + t_offset);

    }
    if (fields[FIELD_VISCOSITY] == 1)  {
        clover_unpack_message_left_c_(chunk.tiles[tile].t_xmin,
                                      chunk.tiles[tile].t_xmax,
                                      chunk.tiles[tile].t_ymin,
                                      chunk.tiles[tile].t_ymax,
                                      chunk.tiles[tile].field.viscosity,
                                      chunk.left_rcv_buffer,
                                      depth, CELL_DATA,
                                      left_right_offset[FIELD_VISCOSITY] + t_offset);

    }
    if (fields[FIELD_SOUNDSPEED] == 1)  {
        clover_unpack_message_left_c_(chunk.tiles[tile].t_xmin,
                                      chunk.tiles[tile].t_xmax,
                                      chunk.tiles[tile].t_ymin,
                                      chunk.tiles[tile].t_ymax,
                                      chunk.tiles[tile].field.soundspeed,
                                      chunk.left_rcv_buffer,
                                      depth, CELL_DATA,
                                      left_right_offset[FIELD_SOUNDSPEED] + t_offset);

    }
    if (fields[FIELD_XVEL0] == 1)  {
        clover_unpack_message_left_c_(chunk.tiles[tile].t_xmin,
                                      chunk.tiles[tile].t_xmax,
                                      chunk.tiles[tile].t_ymin,
                                      chunk.tiles[tile].t_ymax,
                                      chunk.tiles[tile].field.xvel0,
                                      chunk.left_rcv_buffer,
                                      depth, VERTEX_DATA,
                                      left_right_offset[FIELD_XVEL0] + t_offset);

    }
    if (fields[FIELD_XVEL1] == 1)  {
        clover_unpack_message_left_c_(chunk.tiles[tile].t_xmin,
                                      chunk.tiles[tile].t_xmax,
                                      chunk.tiles[tile].t_ymin,
                                      chunk.tiles[tile].t_ymax,
                                      chunk.tiles[tile].field.xvel1,
                                      chunk.left_rcv_buffer,
                                      depth, VERTEX_DATA,
                                      left_right_offset[FIELD_XVEL1] + t_offset);

    }
    if (fields[FIELD_YVEL0] == 1)  {
        clover_unpack_message_left_c_(chunk.tiles[tile].t_xmin,
                                      chunk.tiles[tile].t_xmax,
                                      chunk.tiles[tile].t_ymin,
                                      chunk.tiles[tile].t_ymax,
                                      chunk.tiles[tile].field.yvel0,
                                      chunk.left_rcv_buffer,
                                      depth, VERTEX_DATA,
                                      left_right_offset[FIELD_YVEL0] + t_offset);

    }
    if (fields[FIELD_YVEL1] == 1)  {
        clover_unpack_message_left_c_(chunk.tiles[tile].t_xmin,
                                      chunk.tiles[tile].t_xmax,
                                      chunk.tiles[tile].t_ymin,
                                      chunk.tiles[tile].t_ymax,
                                      chunk.tiles[tile].field.yvel1,
                                      chunk.left_rcv_buffer,
                                      depth, VERTEX_DATA,
                                      left_right_offset[FIELD_YVEL1] + t_offset);

    }
    if (fields[FIELD_VOL_FLUX_X] == 1)  {
        clover_unpack_message_left_c_(chunk.tiles[tile].t_xmin,
                                      chunk.tiles[tile].t_xmax,
                                      chunk.tiles[tile].t_ymin,
                                      chunk.tiles[tile].t_ymax,
                                      chunk.tiles[tile].field.vol_flux_x,
                                      chunk.left_rcv_buffer,
                                      depth, X_FACE_DATA,
                                      left_right_offset[FIELD_VOL_FLUX_X] + t_offset);

    }
    if (fields[FIELD_VOL_FLUX_Y] == 1)  {
        clover_unpack_message_left_c_(chunk.tiles[tile].t_xmin,
                                      chunk.tiles[tile].t_xmax,
                                      chunk.tiles[tile].t_ymin,
                                      chunk.tiles[tile].t_ymax,
                                      chunk.tiles[tile].field.vol_flux_y,
                                      chunk.left_rcv_buffer,
                                      depth, Y_FACE_DATA,
                                      left_right_offset[FIELD_VOL_FLUX_Y] + t_offset);

    }
    if (fields[FIELD_MASS_FLUX_X] == 1)  {
        clover_unpack_message_left_c_(chunk.tiles[tile].t_xmin,
                                      chunk.tiles[tile].t_xmax,
                                      chunk.tiles[tile].t_ymin,
                                      chunk.tiles[tile].t_ymax,
                                      chunk.tiles[tile].field.mass_flux_x,
                                      chunk.left_rcv_buffer,
                                      depth, X_FACE_DATA,
                                      left_right_offset[FIELD_MASS_FLUX_X] + t_offset);

    }
    if (fields[FIELD_MASS_FLUX_Y] == 1)  {
        clover_unpack_message_left_c_(chunk.tiles[tile].t_xmin,
                                      chunk.tiles[tile].t_xmax,
                                      chunk.tiles[tile].t_ymin,
                                      chunk.tiles[tile].t_ymax,
                                      chunk.tiles[tile].field.mass_flux_y,
                                      chunk.left_rcv_buffer,
                                      depth, Y_FACE_DATA,
                                      left_right_offset[FIELD_MASS_FLUX_Y] + t_offset);

    }


}

void clover_pack_right(int tile, int* fields, int depth, int* left_right_offset)
{

    int t_offset = (chunk.tiles[tile].t_bottom - chunk.bottom) * depth;

    if (fields[FIELD_DENSITY0] == 1)  {
        clover_pack_message_right_c_(chunk.tiles[tile].t_xmin,
                                     chunk.tiles[tile].t_xmax,
                                     chunk.tiles[tile].t_ymin,
                                     chunk.tiles[tile].t_ymax,
                                     chunk.tiles[tile].field.density0,
                                     chunk.right_snd_buffer,
                                     depth, CELL_DATA,
                                     left_right_offset[FIELD_DENSITY0] + t_offset);

    }
    if (fields[FIELD_DENSITY1] == 1)  {
        clover_pack_message_right_c_(chunk.tiles[tile].t_xmin,
                                     chunk.tiles[tile].t_xmax,
                                     chunk.tiles[tile].t_ymin,
                                     chunk.tiles[tile].t_ymax,
                                     chunk.tiles[tile].field.density1,
                                     chunk.right_snd_buffer,
                                     depth, CELL_DATA,
                                     left_right_offset[FIELD_DENSITY1] + t_offset);

    }
    if (fields[FIELD_ENERGY0] == 1)  {
        clover_pack_message_right_c_(chunk.tiles[tile].t_xmin,
                                     chunk.tiles[tile].t_xmax,
                                     chunk.tiles[tile].t_ymin,
                                     chunk.tiles[tile].t_ymax,
                                     chunk.tiles[tile].field.energy0,
                                     chunk.right_snd_buffer,
                                     depth, CELL_DATA,
                                     left_right_offset[FIELD_ENERGY0] + t_offset);

    }
    if (fields[FIELD_ENERGY1] == 1)  {
        clover_pack_message_right_c_(chunk.tiles[tile].t_xmin,
                                     chunk.tiles[tile].t_xmax,
                                     chunk.tiles[tile].t_ymin,
                                     chunk.tiles[tile].t_ymax,
                                     chunk.tiles[tile].field.energy1,
                                     chunk.right_snd_buffer,
                                     depth, CELL_DATA,
                                     left_right_offset[FIELD_ENERGY1] + t_offset);

    }
    if (fields[FIELD_PRESSURE] == 1)  {
        clover_pack_message_right_c_(chunk.tiles[tile].t_xmin,
                                     chunk.tiles[tile].t_xmax,
                                     chunk.tiles[tile].t_ymin,
                                     chunk.tiles[tile].t_ymax,
                                     chunk.tiles[tile].field.pressure,
                                     chunk.right_snd_buffer,
                                     depth, CELL_DATA,
                                     left_right_offset[FIELD_PRESSURE] + t_offset);

    }
    if (fields[FIELD_VISCOSITY] == 1)  {
        clover_pack_message_right_c_(chunk.tiles[tile].t_xmin,
                                     chunk.tiles[tile].t_xmax,
                                     chunk.tiles[tile].t_ymin,
                                     chunk.tiles[tile].t_ymax,
                                     chunk.tiles[tile].field.viscosity,
                                     chunk.right_snd_buffer,
                                     depth, CELL_DATA,
                                     left_right_offset[FIELD_VISCOSITY] + t_offset);

    }
    if (fields[FIELD_SOUNDSPEED] == 1)  {
        clover_pack_message_right_c_(chunk.tiles[tile].t_xmin,
                                     chunk.tiles[tile].t_xmax,
                                     chunk.tiles[tile].t_ymin,
                                     chunk.tiles[tile].t_ymax,
                                     chunk.tiles[tile].field.soundspeed,
                                     chunk.right_snd_buffer,
                                     depth, CELL_DATA,
                                     left_right_offset[FIELD_SOUNDSPEED] + t_offset);

    }
    if (fields[FIELD_XVEL0] == 1)  {
        clover_pack_message_right_c_(chunk.tiles[tile].t_xmin,
                                     chunk.tiles[tile].t_xmax,
                                     chunk.tiles[tile].t_ymin,
                                     chunk.tiles[tile].t_ymax,
                                     chunk.tiles[tile].field.xvel0,
                                     chunk.right_snd_buffer,
                                     depth, VERTEX_DATA,
                                     left_right_offset[FIELD_XVEL0] + t_offset);

    }
    if (fields[FIELD_XVEL1] == 1)  {
        clover_pack_message_right_c_(chunk.tiles[tile].t_xmin,
                                     chunk.tiles[tile].t_xmax,
                                     chunk.tiles[tile].t_ymin,
                                     chunk.tiles[tile].t_ymax,
                                     chunk.tiles[tile].field.xvel1,
                                     chunk.right_snd_buffer,
                                     depth, VERTEX_DATA,
                                     left_right_offset[FIELD_XVEL1] + t_offset);


    }
    if (fields[FIELD_YVEL0] == 1)  {
        clover_pack_message_right_c_(chunk.tiles[tile].t_xmin,
                                     chunk.tiles[tile].t_xmax,
                                     chunk.tiles[tile].t_ymin,
                                     chunk.tiles[tile].t_ymax,
                                     chunk.tiles[tile].field.yvel0,
                                     chunk.right_snd_buffer,
                                     depth, VERTEX_DATA,
                                     left_right_offset[FIELD_YVEL0] + t_offset);
//       ELSE

    }
    if (fields[FIELD_YVEL1] == 1)  {
        clover_pack_message_right_c_(chunk.tiles[tile].t_xmin,
                                     chunk.tiles[tile].t_xmax,
                                     chunk.tiles[tile].t_ymin,
                                     chunk.tiles[tile].t_ymax,
                                     chunk.tiles[tile].field.yvel1,
                                     chunk.right_snd_buffer,
                                     depth, VERTEX_DATA,
                                     left_right_offset[FIELD_YVEL1] + t_offset);

    }
    if (fields[FIELD_VOL_FLUX_X] == 1)  {
        clover_pack_message_right_c_(chunk.tiles[tile].t_xmin,
                                     chunk.tiles[tile].t_xmax,
                                     chunk.tiles[tile].t_ymin,
                                     chunk.tiles[tile].t_ymax,
                                     chunk.tiles[tile].field.vol_flux_x,
                                     chunk.right_snd_buffer,
                                     depth, X_FACE_DATA,
                                     left_right_offset[FIELD_VOL_FLUX_X] + t_offset);

    }
    if (fields[FIELD_VOL_FLUX_Y] == 1)  {
        clover_pack_message_right_c_(chunk.tiles[tile].t_xmin,
                                     chunk.tiles[tile].t_xmax,
                                     chunk.tiles[tile].t_ymin,
                                     chunk.tiles[tile].t_ymax,
                                     chunk.tiles[tile].field.vol_flux_y,
                                     chunk.right_snd_buffer,
                                     depth, Y_FACE_DATA,
                                     left_right_offset[FIELD_VOL_FLUX_Y] + t_offset);

    }
    if (fields[FIELD_MASS_FLUX_X] == 1)  {
        clover_pack_message_right_c_(chunk.tiles[tile].t_xmin,
                                     chunk.tiles[tile].t_xmax,
                                     chunk.tiles[tile].t_ymin,
                                     chunk.tiles[tile].t_ymax,
                                     chunk.tiles[tile].field.mass_flux_x,
                                     chunk.right_snd_buffer,
                                     depth, X_FACE_DATA,
                                     left_right_offset[FIELD_MASS_FLUX_X] + t_offset);

    }
    if (fields[FIELD_MASS_FLUX_Y] == 1)  {
        clover_pack_message_right_c_(chunk.tiles[tile].t_xmin,
                                     chunk.tiles[tile].t_xmax,
                                     chunk.tiles[tile].t_ymin,
                                     chunk.tiles[tile].t_ymax,
                                     chunk.tiles[tile].field.mass_flux_y,
                                     chunk.right_snd_buffer,
                                     depth, Y_FACE_DATA,
                                     left_right_offset[FIELD_MASS_FLUX_Y] + t_offset);

    }


}

void clover_send_recv_message_right(double* right_snd_buffer, double* right_rcv_buffer,
                                    int total_size,
                                    int tag_send, int tag_recv,
                                    MPI_Request* req_send, MPI_Request* req_recv)
{

//     IMPLICIT NONE

//     REAL(KIND=8) :: right_snd_buffer(:), right_rcv_buffer(:)
//     INTEGER      :: right_task
//     INTEGER      :: total_size, tag_send, tag_recv, err
//     INTEGER      :: req_send, req_recv

    int right_task = chunk.chunk_neighbours[CHUNK_RIGHT] - 1;

    checkMPIerror(MPI_Isend(right_snd_buffer, total_size, MPI_DOUBLE, right_task, tag_send,
                            MPI_COMM_WORLD, req_send));

    checkMPIerror(MPI_Irecv(right_rcv_buffer, total_size, MPI_DOUBLE, right_task, tag_recv,
                            MPI_COMM_WORLD, req_recv));

}

void clover_unpack_right(int* fields, int tile, int depth,
                         double* right_rcv_buffer,
                         int* left_right_offset)
{

    int t_offset = (chunk.tiles[tile].t_bottom - chunk.bottom) * depth;

    if (fields[FIELD_DENSITY0] == 1)  {
        clover_unpack_message_right_c_(chunk.tiles[tile].t_xmin,
                                       chunk.tiles[tile].t_xmax,
                                       chunk.tiles[tile].t_ymin,
                                       chunk.tiles[tile].t_ymax,
                                       chunk.tiles[tile].field.density0,
                                       chunk.right_rcv_buffer,
                                       depth, CELL_DATA,
                                       left_right_offset[FIELD_DENSITY0] + t_offset);

    }

    if (fields[FIELD_DENSITY1] == 1)  {
        clover_unpack_message_right_c_(chunk.tiles[tile].t_xmin,
                                       chunk.tiles[tile].t_xmax,
                                       chunk.tiles[tile].t_ymin,
                                       chunk.tiles[tile].t_ymax,
                                       chunk.tiles[tile].field.density1,
                                       chunk.right_rcv_buffer,
                                       depth, CELL_DATA,
                                       left_right_offset[FIELD_DENSITY1] + t_offset);

    }
    if (fields[FIELD_ENERGY0] == 1)  {
        clover_unpack_message_right_c_(chunk.tiles[tile].t_xmin,
                                       chunk.tiles[tile].t_xmax,
                                       chunk.tiles[tile].t_ymin,
                                       chunk.tiles[tile].t_ymax,
                                       chunk.tiles[tile].field.energy0,
                                       chunk.right_rcv_buffer,
                                       depth, CELL_DATA,
                                       left_right_offset[FIELD_ENERGY0] + t_offset);

    }
    if (fields[FIELD_ENERGY1] == 1)  {
        clover_unpack_message_right_c_(chunk.tiles[tile].t_xmin,
                                       chunk.tiles[tile].t_xmax,
                                       chunk.tiles[tile].t_ymin,
                                       chunk.tiles[tile].t_ymax,
                                       chunk.tiles[tile].field.energy1,
                                       chunk.right_rcv_buffer,
                                       depth, CELL_DATA,
                                       left_right_offset[FIELD_ENERGY1] + t_offset);

    }
    if (fields[FIELD_PRESSURE] == 1)  {
        clover_unpack_message_right_c_(chunk.tiles[tile].t_xmin,
                                       chunk.tiles[tile].t_xmax,
                                       chunk.tiles[tile].t_ymin,
                                       chunk.tiles[tile].t_ymax,
                                       chunk.tiles[tile].field.pressure,
                                       chunk.right_rcv_buffer,
                                       depth, CELL_DATA,
                                       left_right_offset[FIELD_PRESSURE] + t_offset);

    }
    if (fields[FIELD_VISCOSITY] == 1)  {
        clover_unpack_message_right_c_(chunk.tiles[tile].t_xmin,
                                       chunk.tiles[tile].t_xmax,
                                       chunk.tiles[tile].t_ymin,
                                       chunk.tiles[tile].t_ymax,
                                       chunk.tiles[tile].field.viscosity,
                                       chunk.right_rcv_buffer,
                                       depth, CELL_DATA,
                                       left_right_offset[FIELD_VISCOSITY] + t_offset);

    }
    if (fields[FIELD_SOUNDSPEED] == 1)  {
        clover_unpack_message_right_c_(chunk.tiles[tile].t_xmin,
                                       chunk.tiles[tile].t_xmax,
                                       chunk.tiles[tile].t_ymin,
                                       chunk.tiles[tile].t_ymax,
                                       chunk.tiles[tile].field.soundspeed,
                                       chunk.right_rcv_buffer,
                                       depth, CELL_DATA,
                                       left_right_offset[FIELD_SOUNDSPEED] + t_offset);

    }
    if (fields[FIELD_XVEL0] == 1)  {
        clover_unpack_message_right_c_(chunk.tiles[tile].t_xmin,
                                       chunk.tiles[tile].t_xmax,
                                       chunk.tiles[tile].t_ymin,
                                       chunk.tiles[tile].t_ymax,
                                       chunk.tiles[tile].field.xvel0,
                                       chunk.right_rcv_buffer,
                                       depth, VERTEX_DATA,
                                       left_right_offset[FIELD_XVEL0] + t_offset);

    }
    if (fields[FIELD_XVEL1] == 1)  {
        clover_unpack_message_right_c_(chunk.tiles[tile].t_xmin,
                                       chunk.tiles[tile].t_xmax,
                                       chunk.tiles[tile].t_ymin,
                                       chunk.tiles[tile].t_ymax,
                                       chunk.tiles[tile].field.xvel1,
                                       chunk.right_rcv_buffer,
                                       depth, VERTEX_DATA,
                                       left_right_offset[FIELD_XVEL1] + t_offset);

    }
    if (fields[FIELD_YVEL0] == 1)  {
        clover_unpack_message_right_c_(chunk.tiles[tile].t_xmin,
                                       chunk.tiles[tile].t_xmax,
                                       chunk.tiles[tile].t_ymin,
                                       chunk.tiles[tile].t_ymax,
                                       chunk.tiles[tile].field.yvel0,
                                       chunk.right_rcv_buffer,
                                       depth, VERTEX_DATA,
                                       left_right_offset[FIELD_YVEL0] + t_offset);

    }
    if (fields[FIELD_YVEL1] == 1)  {
        clover_unpack_message_right_c_(chunk.tiles[tile].t_xmin,
                                       chunk.tiles[tile].t_xmax,
                                       chunk.tiles[tile].t_ymin,
                                       chunk.tiles[tile].t_ymax,
                                       chunk.tiles[tile].field.yvel1,
                                       chunk.right_rcv_buffer,
                                       depth, VERTEX_DATA,
                                       left_right_offset[FIELD_YVEL1] + t_offset);

    }
    if (fields[FIELD_VOL_FLUX_X] == 1)  {
        clover_unpack_message_right_c_(chunk.tiles[tile].t_xmin,
                                       chunk.tiles[tile].t_xmax,
                                       chunk.tiles[tile].t_ymin,
                                       chunk.tiles[tile].t_ymax,
                                       chunk.tiles[tile].field.vol_flux_x,
                                       chunk.right_rcv_buffer,
                                       depth, X_FACE_DATA,
                                       left_right_offset[FIELD_VOL_FLUX_X] + t_offset);

    }
    if (fields[FIELD_VOL_FLUX_Y] == 1)  {
        clover_unpack_message_right_c_(chunk.tiles[tile].t_xmin,
                                       chunk.tiles[tile].t_xmax,
                                       chunk.tiles[tile].t_ymin,
                                       chunk.tiles[tile].t_ymax,
                                       chunk.tiles[tile].field.vol_flux_y,
                                       chunk.right_rcv_buffer,
                                       depth, Y_FACE_DATA,
                                       left_right_offset[FIELD_VOL_FLUX_Y] + t_offset);

    }
    if (fields[FIELD_MASS_FLUX_X] == 1)  {
        clover_unpack_message_right_c_(chunk.tiles[tile].t_xmin,
                                       chunk.tiles[tile].t_xmax,
                                       chunk.tiles[tile].t_ymin,
                                       chunk.tiles[tile].t_ymax,
                                       chunk.tiles[tile].field.mass_flux_x,
                                       chunk.right_rcv_buffer,
                                       depth, X_FACE_DATA,
                                       left_right_offset[FIELD_MASS_FLUX_X] + t_offset);

    }
    if (fields[FIELD_MASS_FLUX_Y] == 1)  {
        clover_unpack_message_right_c_(chunk.tiles[tile].t_xmin,
                                       chunk.tiles[tile].t_xmax,
                                       chunk.tiles[tile].t_ymin,
                                       chunk.tiles[tile].t_ymax,
                                       chunk.tiles[tile].field.mass_flux_y,
                                       chunk.right_rcv_buffer,
                                       depth, Y_FACE_DATA,
                                       left_right_offset[FIELD_MASS_FLUX_Y] + t_offset);

    }


}

void clover_pack_top(int tile, int* fields, int depth, int* bottom_top_offset)
{

    int t_offset = (chunk.tiles[tile].t_left - chunk.left) * depth;

    if (fields[FIELD_DENSITY0] == 1)  {
        clover_pack_message_top_c_(chunk.tiles[tile].t_xmin,
                                   chunk.tiles[tile].t_xmax,
                                   chunk.tiles[tile].t_ymin,
                                   chunk.tiles[tile].t_ymax,
                                   chunk.tiles[tile].field.density0,
                                   chunk.top_snd_buffer,
                                   depth, CELL_DATA,
                                   bottom_top_offset[FIELD_DENSITY0] + t_offset);

    }
    if (fields[FIELD_DENSITY1] == 1)  {
        clover_pack_message_top_c_(chunk.tiles[tile].t_xmin,
                                   chunk.tiles[tile].t_xmax,
                                   chunk.tiles[tile].t_ymin,
                                   chunk.tiles[tile].t_ymax,
                                   chunk.tiles[tile].field.density1,
                                   chunk.top_snd_buffer,
                                   depth, CELL_DATA,
                                   bottom_top_offset[FIELD_DENSITY1] + t_offset);

    }
    if (fields[FIELD_ENERGY0] == 1)  {
        clover_pack_message_top_c_(chunk.tiles[tile].t_xmin,
                                   chunk.tiles[tile].t_xmax,
                                   chunk.tiles[tile].t_ymin,
                                   chunk.tiles[tile].t_ymax,
                                   chunk.tiles[tile].field.energy0,
                                   chunk.top_snd_buffer,
                                   depth, CELL_DATA,
                                   bottom_top_offset[FIELD_ENERGY0] + t_offset);

    }
    if (fields[FIELD_ENERGY1] == 1)  {
        clover_pack_message_top_c_(chunk.tiles[tile].t_xmin,
                                   chunk.tiles[tile].t_xmax,
                                   chunk.tiles[tile].t_ymin,
                                   chunk.tiles[tile].t_ymax,
                                   chunk.tiles[tile].field.energy1,
                                   chunk.top_snd_buffer,
                                   depth, CELL_DATA,
                                   bottom_top_offset[FIELD_ENERGY1] + t_offset);

    }
    if (fields[FIELD_PRESSURE] == 1)  {
        clover_pack_message_top_c_(chunk.tiles[tile].t_xmin,
                                   chunk.tiles[tile].t_xmax,
                                   chunk.tiles[tile].t_ymin,
                                   chunk.tiles[tile].t_ymax,
                                   chunk.tiles[tile].field.pressure,
                                   chunk.top_snd_buffer,
                                   depth, CELL_DATA,
                                   bottom_top_offset[FIELD_PRESSURE] + t_offset);

    }
    if (fields[FIELD_VISCOSITY] == 1)  {
        clover_pack_message_top_c_(chunk.tiles[tile].t_xmin,
                                   chunk.tiles[tile].t_xmax,
                                   chunk.tiles[tile].t_ymin,
                                   chunk.tiles[tile].t_ymax,
                                   chunk.tiles[tile].field.viscosity,
                                   chunk.top_snd_buffer,
                                   depth, CELL_DATA,
                                   bottom_top_offset[FIELD_VISCOSITY] + t_offset);

    }
    if (fields[FIELD_SOUNDSPEED] == 1)  {
        clover_pack_message_top_c_(chunk.tiles[tile].t_xmin,
                                   chunk.tiles[tile].t_xmax,
                                   chunk.tiles[tile].t_ymin,
                                   chunk.tiles[tile].t_ymax,
                                   chunk.tiles[tile].field.soundspeed,
                                   chunk.top_snd_buffer,
                                   depth, CELL_DATA,
                                   bottom_top_offset[FIELD_SOUNDSPEED] + t_offset);

    }
    if (fields[FIELD_XVEL0] == 1)  {
        clover_pack_message_top_c_(chunk.tiles[tile].t_xmin,
                                   chunk.tiles[tile].t_xmax,
                                   chunk.tiles[tile].t_ymin,
                                   chunk.tiles[tile].t_ymax,
                                   chunk.tiles[tile].field.xvel0,
                                   chunk.top_snd_buffer,
                                   depth, VERTEX_DATA,
                                   bottom_top_offset[FIELD_XVEL0] + t_offset);

    }
    if (fields[FIELD_XVEL1] == 1)  {
        clover_pack_message_top_c_(chunk.tiles[tile].t_xmin,
                                   chunk.tiles[tile].t_xmax,
                                   chunk.tiles[tile].t_ymin,
                                   chunk.tiles[tile].t_ymax,
                                   chunk.tiles[tile].field.xvel1,
                                   chunk.top_snd_buffer,
                                   depth, VERTEX_DATA,
                                   bottom_top_offset[FIELD_XVEL1] + t_offset);

    }
    if (fields[FIELD_YVEL0] == 1)  {
        clover_pack_message_top_c_(chunk.tiles[tile].t_xmin,
                                   chunk.tiles[tile].t_xmax,
                                   chunk.tiles[tile].t_ymin,
                                   chunk.tiles[tile].t_ymax,
                                   chunk.tiles[tile].field.yvel0,
                                   chunk.top_snd_buffer,
                                   depth, VERTEX_DATA,
                                   bottom_top_offset[FIELD_YVEL0] + t_offset);

    }
    if (fields[FIELD_YVEL1] == 1)  {
        clover_pack_message_top_c_(chunk.tiles[tile].t_xmin,
                                   chunk.tiles[tile].t_xmax,
                                   chunk.tiles[tile].t_ymin,
                                   chunk.tiles[tile].t_ymax,
                                   chunk.tiles[tile].field.yvel1,
                                   chunk.top_snd_buffer,
                                   depth, VERTEX_DATA,
                                   bottom_top_offset[FIELD_YVEL1] + t_offset);

    }
    if (fields[FIELD_VOL_FLUX_X] == 1)  {
        clover_pack_message_top_c_(chunk.tiles[tile].t_xmin,
                                   chunk.tiles[tile].t_xmax,
                                   chunk.tiles[tile].t_ymin,
                                   chunk.tiles[tile].t_ymax,
                                   chunk.tiles[tile].field.vol_flux_x,
                                   chunk.top_snd_buffer,
                                   depth, X_FACE_DATA,
                                   bottom_top_offset[FIELD_VOL_FLUX_X] + t_offset);

    }
    if (fields[FIELD_VOL_FLUX_Y] == 1)  {
        clover_pack_message_top_c_(chunk.tiles[tile].t_xmin,
                                   chunk.tiles[tile].t_xmax,
                                   chunk.tiles[tile].t_ymin,
                                   chunk.tiles[tile].t_ymax,
                                   chunk.tiles[tile].field.vol_flux_y,
                                   chunk.top_snd_buffer,
                                   depth, Y_FACE_DATA,
                                   bottom_top_offset[FIELD_VOL_FLUX_Y] + t_offset);

    }
    if (fields[FIELD_MASS_FLUX_X] == 1)  {
        clover_pack_message_top_c_(chunk.tiles[tile].t_xmin,
                                   chunk.tiles[tile].t_xmax,
                                   chunk.tiles[tile].t_ymin,
                                   chunk.tiles[tile].t_ymax,
                                   chunk.tiles[tile].field.mass_flux_x,
                                   chunk.top_snd_buffer,
                                   depth, X_FACE_DATA,
                                   bottom_top_offset[FIELD_MASS_FLUX_X] + t_offset);

    }
    if (fields[FIELD_MASS_FLUX_Y] == 1)  {
        clover_pack_message_top_c_(chunk.tiles[tile].t_xmin,
                                   chunk.tiles[tile].t_xmax,
                                   chunk.tiles[tile].t_ymin,
                                   chunk.tiles[tile].t_ymax,
                                   chunk.tiles[tile].field.mass_flux_y,
                                   chunk.top_snd_buffer,
                                   depth, Y_FACE_DATA,
                                   bottom_top_offset[FIELD_MASS_FLUX_Y] + t_offset);

    }

}

void clover_send_recv_message_top(double* top_snd_buffer, double* top_rcv_buffer,
                                  int total_size,
                                  int tag_send, int tag_recv,
                                  MPI_Request* req_send, MPI_Request* req_recv)
{

//     IMPLICIT NONE

//     REAL(KIND=8) :: top_snd_buffer(:), top_rcv_buffer(:)
//     INTEGER      :: top_task
//     INTEGER      :: total_size, tag_send, tag_recv, err
//     INTEGER      :: req_send, req_recv


    int top_task = chunk.chunk_neighbours[CHUNK_TOP] - 1;
    // printf("top_task = %d\n", top_task);

    checkMPIerror(MPI_Isend(top_snd_buffer, total_size, MPI_DOUBLE, top_task, tag_send,
                            MPI_COMM_WORLD, req_send));

    checkMPIerror(MPI_Irecv(top_rcv_buffer, total_size, MPI_DOUBLE, top_task, tag_recv,
                            MPI_COMM_WORLD, req_recv));

}

void clover_unpack_top(int* fields, int tile, int depth,
                       double* top_rcv_buffer,
                       int* bottom_top_offset)
{
    int t_offset = (chunk.tiles[tile].t_left - chunk.left) * depth;

    if (fields[FIELD_DENSITY0] == 1)  {
        clover_unpack_message_top_c_(chunk.tiles[tile].t_xmin,
                                     chunk.tiles[tile].t_xmax,
                                     chunk.tiles[tile].t_ymin,
                                     chunk.tiles[tile].t_ymax,
                                     chunk.tiles[tile].field.density0,
                                     chunk.top_rcv_buffer,
                                     depth, CELL_DATA,
                                     bottom_top_offset[FIELD_DENSITY0] + t_offset);

    }
    if (fields[FIELD_DENSITY1] == 1)  {
        clover_unpack_message_top_c_(chunk.tiles[tile].t_xmin,
                                     chunk.tiles[tile].t_xmax,
                                     chunk.tiles[tile].t_ymin,
                                     chunk.tiles[tile].t_ymax,
                                     chunk.tiles[tile].field.density1,
                                     chunk.top_rcv_buffer,
                                     depth, CELL_DATA,
                                     bottom_top_offset[FIELD_DENSITY1] + t_offset);

    }
    if (fields[FIELD_ENERGY0] == 1)  {
        clover_unpack_message_top_c_(chunk.tiles[tile].t_xmin,
                                     chunk.tiles[tile].t_xmax,
                                     chunk.tiles[tile].t_ymin,
                                     chunk.tiles[tile].t_ymax,
                                     chunk.tiles[tile].field.energy0,
                                     chunk.top_rcv_buffer,
                                     depth, CELL_DATA,
                                     bottom_top_offset[FIELD_ENERGY0] + t_offset);

    }
    if (fields[FIELD_ENERGY1] == 1)  {
        clover_unpack_message_top_c_(chunk.tiles[tile].t_xmin,
                                     chunk.tiles[tile].t_xmax,
                                     chunk.tiles[tile].t_ymin,
                                     chunk.tiles[tile].t_ymax,
                                     chunk.tiles[tile].field.energy1,
                                     chunk.top_rcv_buffer,
                                     depth, CELL_DATA,
                                     bottom_top_offset[FIELD_ENERGY1] + t_offset);

    }
    if (fields[FIELD_PRESSURE] == 1)  {
        clover_unpack_message_top_c_(chunk.tiles[tile].t_xmin,
                                     chunk.tiles[tile].t_xmax,
                                     chunk.tiles[tile].t_ymin,
                                     chunk.tiles[tile].t_ymax,
                                     chunk.tiles[tile].field.pressure,
                                     chunk.top_rcv_buffer,
                                     depth, CELL_DATA,
                                     bottom_top_offset[FIELD_PRESSURE] + t_offset);

    }
    if (fields[FIELD_VISCOSITY] == 1)  {
        clover_unpack_message_top_c_(chunk.tiles[tile].t_xmin,
                                     chunk.tiles[tile].t_xmax,
                                     chunk.tiles[tile].t_ymin,
                                     chunk.tiles[tile].t_ymax,
                                     chunk.tiles[tile].field.viscosity,
                                     chunk.top_rcv_buffer,
                                     depth, CELL_DATA,
                                     bottom_top_offset[FIELD_VISCOSITY] + t_offset);

    }
    if (fields[FIELD_SOUNDSPEED] == 1)  {
        clover_unpack_message_top_c_(chunk.tiles[tile].t_xmin,
                                     chunk.tiles[tile].t_xmax,
                                     chunk.tiles[tile].t_ymin,
                                     chunk.tiles[tile].t_ymax,
                                     chunk.tiles[tile].field.soundspeed,
                                     chunk.top_rcv_buffer,
                                     depth, CELL_DATA,
                                     bottom_top_offset[FIELD_SOUNDSPEED] + t_offset);

    }
    if (fields[FIELD_XVEL0] == 1)  {
        clover_unpack_message_top_c_(chunk.tiles[tile].t_xmin,
                                     chunk.tiles[tile].t_xmax,
                                     chunk.tiles[tile].t_ymin,
                                     chunk.tiles[tile].t_ymax,
                                     chunk.tiles[tile].field.xvel0,
                                     chunk.top_rcv_buffer,
                                     depth, VERTEX_DATA,
                                     bottom_top_offset[FIELD_XVEL0] + t_offset);

    }
    if (fields[FIELD_XVEL1] == 1)  {
        clover_unpack_message_top_c_(chunk.tiles[tile].t_xmin,
                                     chunk.tiles[tile].t_xmax,
                                     chunk.tiles[tile].t_ymin,
                                     chunk.tiles[tile].t_ymax,
                                     chunk.tiles[tile].field.xvel1,
                                     chunk.top_rcv_buffer,
                                     depth, VERTEX_DATA,
                                     bottom_top_offset[FIELD_XVEL1] + t_offset);

    }
    if (fields[FIELD_YVEL0] == 1)  {
        clover_unpack_message_top_c_(chunk.tiles[tile].t_xmin,
                                     chunk.tiles[tile].t_xmax,
                                     chunk.tiles[tile].t_ymin,
                                     chunk.tiles[tile].t_ymax,
                                     chunk.tiles[tile].field.yvel0,
                                     chunk.top_rcv_buffer,
                                     depth, VERTEX_DATA,
                                     bottom_top_offset[FIELD_YVEL0] + t_offset);

    }
    if (fields[FIELD_YVEL1] == 1)  {
        clover_unpack_message_top_c_(chunk.tiles[tile].t_xmin,
                                     chunk.tiles[tile].t_xmax,
                                     chunk.tiles[tile].t_ymin,
                                     chunk.tiles[tile].t_ymax,
                                     chunk.tiles[tile].field.yvel1,
                                     chunk.top_rcv_buffer,
                                     depth, VERTEX_DATA,
                                     bottom_top_offset[FIELD_YVEL1] + t_offset);

    }
    if (fields[FIELD_VOL_FLUX_X] == 1)  {
        clover_unpack_message_top_c_(chunk.tiles[tile].t_xmin,
                                     chunk.tiles[tile].t_xmax,
                                     chunk.tiles[tile].t_ymin,
                                     chunk.tiles[tile].t_ymax,
                                     chunk.tiles[tile].field.vol_flux_x,
                                     chunk.top_rcv_buffer,
                                     depth, X_FACE_DATA,
                                     bottom_top_offset[FIELD_VOL_FLUX_X] + t_offset);

    }
    if (fields[FIELD_VOL_FLUX_Y] == 1)  {
        clover_unpack_message_top_c_(chunk.tiles[tile].t_xmin,
                                     chunk.tiles[tile].t_xmax,
                                     chunk.tiles[tile].t_ymin,
                                     chunk.tiles[tile].t_ymax,
                                     chunk.tiles[tile].field.vol_flux_y,
                                     chunk.top_rcv_buffer,
                                     depth, Y_FACE_DATA,
                                     bottom_top_offset[FIELD_VOL_FLUX_Y] + t_offset);

    }
    if (fields[FIELD_MASS_FLUX_X] == 1)  {
        clover_unpack_message_top_c_(chunk.tiles[tile].t_xmin,
                                     chunk.tiles[tile].t_xmax,
                                     chunk.tiles[tile].t_ymin,
                                     chunk.tiles[tile].t_ymax,
                                     chunk.tiles[tile].field.mass_flux_x,
                                     chunk.top_rcv_buffer,
                                     depth, X_FACE_DATA,
                                     bottom_top_offset[FIELD_MASS_FLUX_X] + t_offset);

    }
    if (fields[FIELD_MASS_FLUX_Y] == 1)  {
        clover_unpack_message_top_c_(chunk.tiles[tile].t_xmin,
                                     chunk.tiles[tile].t_xmax,
                                     chunk.tiles[tile].t_ymin,
                                     chunk.tiles[tile].t_ymax,
                                     chunk.tiles[tile].field.mass_flux_y,
                                     chunk.top_rcv_buffer,
                                     depth, Y_FACE_DATA,
                                     bottom_top_offset[FIELD_MASS_FLUX_Y] + t_offset);

    }



}

void clover_pack_bottom(int tile, int* fields, int depth, int* bottom_top_offset)
{
    int t_offset = (chunk.tiles[tile].t_left - chunk.left) * depth;

    if (fields[FIELD_DENSITY0] == 1)  {
        clover_pack_message_bottom_c_(chunk.tiles[tile].t_xmin,
                                      chunk.tiles[tile].t_xmax,
                                      chunk.tiles[tile].t_ymin,
                                      chunk.tiles[tile].t_ymax,
                                      chunk.tiles[tile].field.density0,
                                      chunk.bottom_snd_buffer,
                                      depth, CELL_DATA,
                                      bottom_top_offset[FIELD_DENSITY0] + t_offset);
//       ELSE

    }
    if (fields[FIELD_DENSITY1] == 1)  {
        clover_pack_message_bottom_c_(chunk.tiles[tile].t_xmin,
                                      chunk.tiles[tile].t_xmax,
                                      chunk.tiles[tile].t_ymin,
                                      chunk.tiles[tile].t_ymax,
                                      chunk.tiles[tile].field.density1,
                                      chunk.bottom_snd_buffer,
                                      depth, CELL_DATA,
                                      bottom_top_offset[FIELD_DENSITY1] + t_offset);

    }
    if (fields[FIELD_ENERGY0] == 1)  {
        clover_pack_message_bottom_c_(chunk.tiles[tile].t_xmin,
                                      chunk.tiles[tile].t_xmax,
                                      chunk.tiles[tile].t_ymin,
                                      chunk.tiles[tile].t_ymax,
                                      chunk.tiles[tile].field.energy0,
                                      chunk.bottom_snd_buffer,
                                      depth, CELL_DATA,
                                      bottom_top_offset[FIELD_ENERGY0] + t_offset);

    }
    if (fields[FIELD_ENERGY1] == 1)  {
        clover_pack_message_bottom_c_(chunk.tiles[tile].t_xmin,
                                      chunk.tiles[tile].t_xmax,
                                      chunk.tiles[tile].t_ymin,
                                      chunk.tiles[tile].t_ymax,
                                      chunk.tiles[tile].field.energy1,
                                      chunk.bottom_snd_buffer,
                                      depth, CELL_DATA,
                                      bottom_top_offset[FIELD_ENERGY1] + t_offset);

    }
    if (fields[FIELD_PRESSURE] == 1)  {
        clover_pack_message_bottom_c_(chunk.tiles[tile].t_xmin,
                                      chunk.tiles[tile].t_xmax,
                                      chunk.tiles[tile].t_ymin,
                                      chunk.tiles[tile].t_ymax,
                                      chunk.tiles[tile].field.pressure,
                                      chunk.bottom_snd_buffer,
                                      depth, CELL_DATA,
                                      bottom_top_offset[FIELD_PRESSURE] + t_offset);

    }
    if (fields[FIELD_VISCOSITY] == 1)  {
        clover_pack_message_bottom_c_(chunk.tiles[tile].t_xmin,
                                      chunk.tiles[tile].t_xmax,
                                      chunk.tiles[tile].t_ymin,
                                      chunk.tiles[tile].t_ymax,
                                      chunk.tiles[tile].field.viscosity,
                                      chunk.bottom_snd_buffer,
                                      depth, CELL_DATA,
                                      bottom_top_offset[FIELD_VISCOSITY] + t_offset);

    }
    if (fields[FIELD_SOUNDSPEED] == 1)  {
        clover_pack_message_bottom_c_(chunk.tiles[tile].t_xmin,
                                      chunk.tiles[tile].t_xmax,
                                      chunk.tiles[tile].t_ymin,
                                      chunk.tiles[tile].t_ymax,
                                      chunk.tiles[tile].field.soundspeed,
                                      chunk.bottom_snd_buffer,
                                      depth, CELL_DATA,
                                      bottom_top_offset[FIELD_SOUNDSPEED] + t_offset);

    }
    if (fields[FIELD_XVEL0] == 1)  {
        clover_pack_message_bottom_c_(chunk.tiles[tile].t_xmin,
                                      chunk.tiles[tile].t_xmax,
                                      chunk.tiles[tile].t_ymin,
                                      chunk.tiles[tile].t_ymax,
                                      chunk.tiles[tile].field.xvel0,
                                      chunk.bottom_snd_buffer,
                                      depth, VERTEX_DATA,
                                      bottom_top_offset[FIELD_XVEL0] + t_offset);

    }
    if (fields[FIELD_XVEL1] == 1)  {
        clover_pack_message_bottom_c_(chunk.tiles[tile].t_xmin,
                                      chunk.tiles[tile].t_xmax,
                                      chunk.tiles[tile].t_ymin,
                                      chunk.tiles[tile].t_ymax,
                                      chunk.tiles[tile].field.xvel1,
                                      chunk.bottom_snd_buffer,
                                      depth, VERTEX_DATA,
                                      bottom_top_offset[FIELD_XVEL1] + t_offset);

    }
    if (fields[FIELD_YVEL0] == 1)  {
        clover_pack_message_bottom_c_(chunk.tiles[tile].t_xmin,
                                      chunk.tiles[tile].t_xmax,
                                      chunk.tiles[tile].t_ymin,
                                      chunk.tiles[tile].t_ymax,
                                      chunk.tiles[tile].field.yvel0,
                                      chunk.bottom_snd_buffer,
                                      depth, VERTEX_DATA,
                                      bottom_top_offset[FIELD_YVEL0] + t_offset);

    }
    if (fields[FIELD_YVEL1] == 1)  {
        clover_pack_message_bottom_c_(chunk.tiles[tile].t_xmin,
                                      chunk.tiles[tile].t_xmax,
                                      chunk.tiles[tile].t_ymin,
                                      chunk.tiles[tile].t_ymax,
                                      chunk.tiles[tile].field.yvel1,
                                      chunk.bottom_snd_buffer,
                                      depth, VERTEX_DATA,
                                      bottom_top_offset[FIELD_YVEL1] + t_offset);

    }
    if (fields[FIELD_VOL_FLUX_X] == 1)  {
        clover_pack_message_bottom_c_(chunk.tiles[tile].t_xmin,
                                      chunk.tiles[tile].t_xmax,
                                      chunk.tiles[tile].t_ymin,
                                      chunk.tiles[tile].t_ymax,
                                      chunk.tiles[tile].field.vol_flux_x,
                                      chunk.bottom_snd_buffer,
                                      depth, X_FACE_DATA,
                                      bottom_top_offset[FIELD_VOL_FLUX_X] + t_offset);

    }
    if (fields[FIELD_VOL_FLUX_Y] == 1)  {
        clover_pack_message_bottom_c_(chunk.tiles[tile].t_xmin,
                                      chunk.tiles[tile].t_xmax,
                                      chunk.tiles[tile].t_ymin,
                                      chunk.tiles[tile].t_ymax,
                                      chunk.tiles[tile].field.vol_flux_y,
                                      chunk.bottom_snd_buffer,
                                      depth, Y_FACE_DATA,
                                      bottom_top_offset[FIELD_VOL_FLUX_Y] + t_offset);

    }
    if (fields[FIELD_MASS_FLUX_X] == 1)  {
        clover_pack_message_bottom_c_(chunk.tiles[tile].t_xmin,
                                      chunk.tiles[tile].t_xmax,
                                      chunk.tiles[tile].t_ymin,
                                      chunk.tiles[tile].t_ymax,
                                      chunk.tiles[tile].field.mass_flux_x,
                                      chunk.bottom_snd_buffer,
                                      depth, X_FACE_DATA,
                                      bottom_top_offset[FIELD_MASS_FLUX_X] + t_offset);

    }
    if (fields[FIELD_MASS_FLUX_Y] == 1)  {
        clover_pack_message_bottom_c_(chunk.tiles[tile].t_xmin,
                                      chunk.tiles[tile].t_xmax,
                                      chunk.tiles[tile].t_ymin,
                                      chunk.tiles[tile].t_ymax,
                                      chunk.tiles[tile].field.mass_flux_y,
                                      chunk.bottom_snd_buffer,
                                      depth, Y_FACE_DATA,
                                      bottom_top_offset[FIELD_MASS_FLUX_Y] + t_offset);

    }


}

void clover_send_recv_message_bottom(double* bottom_snd_buffer, double* bottom_rcv_buffer,
                                     int total_size,
                                     int tag_send, int tag_recv,
                                     MPI_Request* req_send, MPI_Request* req_recv)
{
//     IMPLICIT NONE

//     REAL(KIND=8) :: bottom_snd_buffer(:), bottom_rcv_buffer(:)
//     INTEGER      :: bottom_task
//     INTEGER      :: total_size, tag_send, tag_recv, err
//     INTEGER      :: req_send, req_recv

    int bottom_task = chunk.chunk_neighbours[CHUNK_BOTTOM] - 1;

    checkMPIerror(MPI_Isend(bottom_snd_buffer, total_size, MPI_DOUBLE, bottom_task, tag_send
                            , MPI_COMM_WORLD, req_send));

    checkMPIerror(MPI_Irecv(bottom_rcv_buffer, total_size, MPI_DOUBLE, bottom_task, tag_recv
                            , MPI_COMM_WORLD, req_recv));

}

void clover_unpack_bottom(int* fields, int tile, int depth,
                          double* bottom_rcv_buffer,
                          int* bottom_top_offset)
{
    int t_offset = (chunk.tiles[tile].t_left - chunk.left) * depth;

    if (fields[FIELD_DENSITY0] == 1)  {
        clover_unpack_message_bottom_c_(chunk.tiles[tile].t_xmin,
                                        chunk.tiles[tile].t_xmax,
                                        chunk.tiles[tile].t_ymin,
                                        chunk.tiles[tile].t_ymax,
                                        chunk.tiles[tile].field.density0,
                                        chunk.bottom_rcv_buffer,
                                        depth, CELL_DATA,
                                        bottom_top_offset[FIELD_DENSITY0] + t_offset);

    }
    if (fields[FIELD_DENSITY1] == 1)  {
        clover_unpack_message_bottom_c_(chunk.tiles[tile].t_xmin,
                                        chunk.tiles[tile].t_xmax,
                                        chunk.tiles[tile].t_ymin,
                                        chunk.tiles[tile].t_ymax,
                                        chunk.tiles[tile].field.density1,
                                        chunk.bottom_rcv_buffer,
                                        depth, CELL_DATA,
                                        bottom_top_offset[FIELD_DENSITY1] + t_offset);

    }
    if (fields[FIELD_ENERGY0] == 1)  {
        clover_unpack_message_bottom_c_(chunk.tiles[tile].t_xmin,
                                        chunk.tiles[tile].t_xmax,
                                        chunk.tiles[tile].t_ymin,
                                        chunk.tiles[tile].t_ymax,
                                        chunk.tiles[tile].field.energy0,
                                        chunk.bottom_rcv_buffer,
                                        depth, CELL_DATA,
                                        bottom_top_offset[FIELD_ENERGY0] + t_offset);

    }
    if (fields[FIELD_ENERGY1] == 1)  {
        clover_unpack_message_bottom_c_(chunk.tiles[tile].t_xmin,
                                        chunk.tiles[tile].t_xmax,
                                        chunk.tiles[tile].t_ymin,
                                        chunk.tiles[tile].t_ymax,
                                        chunk.tiles[tile].field.energy1,
                                        chunk.bottom_rcv_buffer,
                                        depth, CELL_DATA,
                                        bottom_top_offset[FIELD_ENERGY1] + t_offset);

    }
    if (fields[FIELD_PRESSURE] == 1)  {
        clover_unpack_message_bottom_c_(chunk.tiles[tile].t_xmin,
                                        chunk.tiles[tile].t_xmax,
                                        chunk.tiles[tile].t_ymin,
                                        chunk.tiles[tile].t_ymax,
                                        chunk.tiles[tile].field.pressure,
                                        chunk.bottom_rcv_buffer,
                                        depth, CELL_DATA,
                                        bottom_top_offset[FIELD_PRESSURE] + t_offset);

    }
    if (fields[FIELD_VISCOSITY] == 1)  {
        clover_unpack_message_bottom_c_(chunk.tiles[tile].t_xmin,
                                        chunk.tiles[tile].t_xmax,
                                        chunk.tiles[tile].t_ymin,
                                        chunk.tiles[tile].t_ymax,
                                        chunk.tiles[tile].field.viscosity,
                                        chunk.bottom_rcv_buffer,
                                        depth, CELL_DATA,
                                        bottom_top_offset[FIELD_VISCOSITY] + t_offset);

    }
    if (fields[FIELD_SOUNDSPEED] == 1)  {
        clover_unpack_message_bottom_c_(chunk.tiles[tile].t_xmin,
                                        chunk.tiles[tile].t_xmax,
                                        chunk.tiles[tile].t_ymin,
                                        chunk.tiles[tile].t_ymax,
                                        chunk.tiles[tile].field.soundspeed,
                                        chunk.bottom_rcv_buffer,
                                        depth, CELL_DATA,
                                        bottom_top_offset[FIELD_SOUNDSPEED] + t_offset);

    }
    if (fields[FIELD_XVEL0] == 1)  {
        clover_unpack_message_bottom_c_(chunk.tiles[tile].t_xmin,
                                        chunk.tiles[tile].t_xmax,
                                        chunk.tiles[tile].t_ymin,
                                        chunk.tiles[tile].t_ymax,
                                        chunk.tiles[tile].field.xvel0,
                                        chunk.bottom_rcv_buffer,
                                        depth, VERTEX_DATA,
                                        bottom_top_offset[FIELD_XVEL0] + t_offset);

    }
    if (fields[FIELD_XVEL1] == 1)  {
        clover_unpack_message_bottom_c_(chunk.tiles[tile].t_xmin,
                                        chunk.tiles[tile].t_xmax,
                                        chunk.tiles[tile].t_ymin,
                                        chunk.tiles[tile].t_ymax,
                                        chunk.tiles[tile].field.xvel1,
                                        chunk.bottom_rcv_buffer,
                                        depth, VERTEX_DATA,
                                        bottom_top_offset[FIELD_XVEL1] + t_offset);

    }
    if (fields[FIELD_YVEL0] == 1)  {
        clover_unpack_message_bottom_c_(chunk.tiles[tile].t_xmin,
                                        chunk.tiles[tile].t_xmax,
                                        chunk.tiles[tile].t_ymin,
                                        chunk.tiles[tile].t_ymax,
                                        chunk.tiles[tile].field.yvel0,
                                        chunk.bottom_rcv_buffer,
                                        depth, VERTEX_DATA,
                                        bottom_top_offset[FIELD_YVEL0] + t_offset);

    }
    if (fields[FIELD_YVEL1] == 1)  {
        clover_unpack_message_bottom_c_(chunk.tiles[tile].t_xmin,
                                        chunk.tiles[tile].t_xmax,
                                        chunk.tiles[tile].t_ymin,
                                        chunk.tiles[tile].t_ymax,
                                        chunk.tiles[tile].field.yvel1,
                                        chunk.bottom_rcv_buffer,
                                        depth, VERTEX_DATA,
                                        bottom_top_offset[FIELD_YVEL1] + t_offset);

    }
    if (fields[FIELD_VOL_FLUX_X] == 1)  {
        clover_unpack_message_bottom_c_(chunk.tiles[tile].t_xmin,
                                        chunk.tiles[tile].t_xmax,
                                        chunk.tiles[tile].t_ymin,
                                        chunk.tiles[tile].t_ymax,
                                        chunk.tiles[tile].field.vol_flux_x,
                                        chunk.bottom_rcv_buffer,
                                        depth, X_FACE_DATA,
                                        bottom_top_offset[FIELD_VOL_FLUX_X] + t_offset);

    }
    if (fields[FIELD_VOL_FLUX_Y] == 1)  {
        clover_unpack_message_bottom_c_(chunk.tiles[tile].t_xmin,
                                        chunk.tiles[tile].t_xmax,
                                        chunk.tiles[tile].t_ymin,
                                        chunk.tiles[tile].t_ymax,
                                        chunk.tiles[tile].field.vol_flux_y,
                                        chunk.bottom_rcv_buffer,
                                        depth, Y_FACE_DATA,
                                        bottom_top_offset[FIELD_VOL_FLUX_Y] + t_offset);

    }
    if (fields[FIELD_MASS_FLUX_X] == 1)  {
        clover_unpack_message_bottom_c_(chunk.tiles[tile].t_xmin,
                                        chunk.tiles[tile].t_xmax,
                                        chunk.tiles[tile].t_ymin,
                                        chunk.tiles[tile].t_ymax,
                                        chunk.tiles[tile].field.mass_flux_x,
                                        chunk.bottom_rcv_buffer,
                                        depth, X_FACE_DATA,
                                        bottom_top_offset[FIELD_MASS_FLUX_X] + t_offset);

    }
    if (fields[FIELD_MASS_FLUX_Y] == 1)  {
        clover_unpack_message_bottom_c_(chunk.tiles[tile].t_xmin,
                                        chunk.tiles[tile].t_xmax,
                                        chunk.tiles[tile].t_ymin,
                                        chunk.tiles[tile].t_ymax,
                                        chunk.tiles[tile].field.mass_flux_y,
                                        chunk.bottom_rcv_buffer,
                                        depth, Y_FACE_DATA,
                                        bottom_top_offset[FIELD_MASS_FLUX_Y] + t_offset);

    }
}

void clover_sum(double* value)
{

    // Only sums to the master

//     IMPLICIT NONE

//     REAL(KIND=8) :: value

//     REAL(KIND=8) :: total

//     INTEGER :: err

    double total = *value;

    checkMPIerror(MPI_Reduce(value, &total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD));

    *value = total;
}

void clover_min(double* value)
{

//     IMPLICIT NONE

//     REAL(KIND=8) :: value

//     REAL(KIND=8) :: minimum

//     INTEGER :: err

    double minimum = *value;

    checkMPIerror(MPI_Allreduce(value, &minimum, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD));

    *value = minimum;

}

void clover_max(double* value)
{

    //     IMPLICIT NONE

    //     REAL(KIND=8) :: value

    //     REAL(KIND=8) :: maximum

    //     INTEGER :: err

    double maximum = *value;

    checkMPIerror(MPI_Allreduce(value, &maximum, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));

    *value = maximum;

}

void clover_allgather(double* value, double* values)
{

    //     IMPLICIT NONE

    //     REAL(KIND=8) :: value

    //     REAL(KIND=8) :: values(parallel.max_task)

    //     INTEGER :: err

    // Just to ensure it will work in serial
    values[0] = *value;

    checkMPIerror(MPI_Allgather(value, 1, MPI_DOUBLE, values, 1, MPI_DOUBLE, MPI_COMM_WORLD));

}

void clover_check_error(int* error)
{

    //     IMPLICIT NONE

    //     INTEGER :: error

    //     INTEGER :: maximum

    //     INTEGER :: err

    int maximum = *error;

    checkMPIerror(MPI_Allreduce(error, &maximum, 1, MPI_INTEGER, MPI_MAX, MPI_COMM_WORLD));

    *error = maximum;

}
