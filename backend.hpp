#pragma once

//STREAMS
#define HALO           0
#define COMPUTE        1

//GRID MATH
#define BOARD_SIZE     5
#define ABSORB         16
//MPI
#define BOARD_SWAP     100
#define REQUESTS_SIZE  4
#define SEND_LEFT      0
#define SEND_RIGHT     1
#define RECV_LEFT      2
#define RECV_RIGHT     3


#define LEFT 0
#define RIGHT 1
#define OUT 0
#define IN 1
void backend_init(int rank, int comm_size, int sx, int sy, int sz);
void backend_data_initialize(int sx, int sy, int sz);
void backend_run(int num_steps);
void backend_finalize();
