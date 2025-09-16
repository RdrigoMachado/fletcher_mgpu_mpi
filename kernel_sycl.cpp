#include "backend.hpp"

MPI_Request request[REQUESTS_SIZE];

void sync_messages(int universe_size, int process_num){
    if(0 != process_num){
        MPI_Wait(&request[SEND_LEFT],  MPI_STATUS_IGNORE);
        MPI_Wait(&request[RECV_LEFT],  MPI_STATUS_IGNORE);
    }
    if(universe_size - 1 != process_num){
        MPI_Wait(&request[SEND_RIGHT], MPI_STATUS_IGNORE);
        MPI_Wait(&request[RECV_RIGHT], MPI_STATUS_IGNORE);
    }
}

void swap_borders(int universe_size, int process_num, float *buffer, int sxsy, int sz_slice){
    int neighbour_left   = process_num - 1;
    int neighbour_right  = process_num + 1;

    int offset_in_left   = 0;
    int offset_in_right  = sxsy * (sz_slice + BOARD_SIZE);

    int offset_out_left  = sxsy * BOARD_SIZE;
    int offset_out_right = sxsy * sz_slice;

    int size = sxsy * BOARD_SIZE;

    //Send bord LEFT
    if(0 != process_num){
        MPI_Isend(&buffer + offset_out_left, size, MPI_FLOAT, neighbour_left,  BOARD_SWAP, MPI_COMM_WORLD, &request[SEND_LEFT]);
        MPI_Irecv(&buffer + offset_in_left,  size, MPI_FLOAT, neighbour_left,  BOARD_SWAP, MPI_COMM_WORLD, &request[RECV_LEFT]);
    }
    //Send bord RIGHT
    if(universe_size - 1 != process_num){
        MPI_Isend(&buffer + offset_out_right, size, MPI_FLOAT, neighbour_right, BOARD_SWAP, MPI_COMM_WORLD, &request[SEND_RIGHT]);
        MPI_Irecv(&buffer + offset_in_right,  size, MPI_FLOAT, neighbour_right, BOARD_SWAP, MPI_COMM_WORLD, &request[RECV_RIGHT]);
    }
}
