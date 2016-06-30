#ifndef CLOVER_H
#define CLOVER_H

#include <mpi.h>

void clover_barrier();
void clover_abort();
void clover_finalize();
void clover_init_comms(int argc, char **argv);
void clover_get_num_chunks(int *count);
void clover_exchange(int *fields, int depth);
void clover_pack_left(int tile, int *fields, int depth, int *left_right_offset);
void clover_send_recv_message_left(double *left_snd_buffer, double *left_rcv_buffer,
                                   int total_size,
                                   int tag_send, int tag_recv,
                                   MPI_Request *req_send, MPI_Request *req_recv);
void clover_unpack_left(int *fields, int tile, int depth,
                        double *left_rcv_buffer,
                        int *left_right_offset);
void clover_pack_right(int tile, int *fields, int depth, int *left_right_offset);
void clover_send_recv_message_right(double *right_snd_buffer, double *right_rcv_buffer,
                                    int total_size,
                                    int tag_send, int tag_recv,
                                    MPI_Request *req_send, MPI_Request *req_recv);
void clover_unpack_right(int *fields, int tile, int depth,
                         double *right_rcv_buffer,
                         int *left_right_offset);
void clover_pack_top(int tile, int *fields, int depth, int *bottom_top_offset);
void clover_send_recv_message_top(double *top_snd_buffer, double *top_rcv_buffer,
                                  int total_size,
                                  int tag_send, int tag_recv,
                                  MPI_Request *req_send, MPI_Request *req_recv);
void clover_unpack_top(int *fields, int tile, int depth,
                       double *top_rcv_buffer,
                       int *bottom_top_offset);
void clover_pack_bottom(int tile, int *fields, int depth, int *bottom_top_offset);
void clover_send_recv_message_bottom(double *bottom_snd_buffer, double *bottom_rcv_buffer,
                                     int total_size,
                                     int tag_send, int tag_recv,
                                     MPI_Request *req_send, MPI_Request *req_recv);
void clover_unpack_bottom(int *fields, int tile, int depth,
                          double * bottom_rcv_buffer,
                          int * bottom_top_offset);
void clover_sum(double *value);
void clover_min(double *value);
void clover_max(double *value);
void clover_allgather(double *value, double * values);
void clover_check_error(int *error);

#endif
