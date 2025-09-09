#pragma once

#define RAIO 8
#define HALO 0
#define COMPUTE 1
#define BOARD_SIZE 5
void backend_init(int rank, int comm_size, int sx, int sy, int sz);
void backend_finalize();
void backend_run(int num_steps);
void backend_data_initialize();
