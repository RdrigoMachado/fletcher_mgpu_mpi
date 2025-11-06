#pragma once

#include <math.h>
//STREAMS
#define STREAM_HALO           0
#define STREAM_COMPUTE        1

//GRID MATH
#define HALO           5
#define ABSORB         16

// ||         |       |      |       |        ||
// || ABSORDB | BOARD | DATA | BOARD | ABSORB ||
// ||         |       |      |       |        ||


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

#define SINGLE_GRID 1
#define FIRST_GRID  0

#define FCUT        40.0
#define PICUBE      31.00627668029982017537
#define TWOSQRTPI    3.54490770181103205458
#define THREESQRTPI  5.31736155271654808184

#define BSIZE_X 32
#define BSIZE_Y 16
#define NPOP 4
#define TOTAL_X (BSIZE_X+2*NPOP)
#define TOTAL_Y (BSIZE_Y+2*NPOP)

#define ind(ix,iy,iz) (((iz)*sy+(iy))*sx+(ix))

void backend_init(int rank, int comm_size, int nx, int ny, int nz, int num_gpus);
void backend_data_initialize();
void backend_run(int num_steps, int deltaT);
void backend_finalize();
