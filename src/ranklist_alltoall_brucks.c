/* Copyright (c) 2012, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by Adam Moody <moody20@llnl.gov> and Edgar A. Leon <leon@llnl.gov>.
 * LLNL-CODE-557516.
 * All rights reserved.
 * This file is part of the DTCMP library.
 * For details, see https://github.com/hpc/dtcmp
 * Please also read this file: LICENSE.TXT. */

#include <stdlib.h>
#include <string.h>
#include "mpi.h"
#include "dtcmp_internal.h"

int ranklist_alltoall_brucks(
  const void* sendbuf,
  void* recvbuf,
  int num,
  MPI_Datatype datatype,
  int group_rank,
  int group_ranks,
  const int* comm_ranklist,
  MPI_Comm comm)
{
  int i;

  /* TODO: feels like some of these memory copies could be avoided */

  /* get true extent of datatype so we can allocate space */
  MPI_Aint lb, extent;
  MPI_Type_get_true_extent(datatype, &lb, &extent);

  /* compute size of datatype */
  size_t size = (size_t) extent * num;
  if (size <= 0) {
    return 0;
  }

  if (group_ranks <= 0) {
    return 0;
  }

  int bufsize = size * group_ranks;
  char* send_data = dtcmp_malloc(bufsize, 0, __FILE__, __LINE__);
  char* recv_data = dtcmp_malloc(bufsize, 0, __FILE__, __LINE__);
  char* tmp_data  = dtcmp_malloc(bufsize, 0, __FILE__, __LINE__);

  /* copy our send data to our receive buffer, and rotate it so our own rank is at the top */
/*
  memcpy(recvbuf, sendbuf, ranks * size);
*/
  memcpy(tmp_data, (char*)sendbuf + group_rank * size, (group_ranks - group_rank) * size);
  memcpy(tmp_data + (group_ranks - group_rank) * size, sendbuf, group_rank * size);

  /* now run through Bruck's index algorithm to exchange data */
  MPI_Request request[2];
  MPI_Status  status[2];
  int step = 1;
  while (step < group_ranks) {
    /* determine our source and destination ranks for this step */
    int dst_index = group_rank + step;
    if (dst_index >= group_ranks) {
      dst_index -= group_ranks;
    }
    int src_index = group_rank - step;
    if (src_index < 0) {
      src_index += group_ranks;
    }

    /* determine our source and destination ranks for this step */
    int dst = comm_ranklist[dst_index];
    int src = comm_ranklist[src_index];

    /* pack our data to send and count number of bytes */
    int send_count = 0;
    char* send_ptr = send_data;
    for (i = 0; i < group_ranks; i++) {
      int mask = (i & step);
      if (mask) {
        memcpy(send_ptr, tmp_data + i * size, size);
        send_ptr += size;
        send_count += num;
      }
    }

    /* exchange messages */
    MPI_Irecv(recv_data, send_count, datatype, src, 0, comm, &request[0]);
    MPI_Isend(send_data, send_count, datatype, dst, 0, comm, &request[1]);
    MPI_Waitall(2, request, status);

    /* unpack received data into our buffer */
    void* recv_ptr = recv_data;
    for (i = 0; i < group_ranks; i++) {
      int mask = (i & step);
      if (mask) {
        memcpy(tmp_data + i * size, recv_ptr, size);
        recv_ptr += size;
      }
    }

    /* go on to the next phase of the exchange */
    step <<= 1;
  }

  /* copy our data to our receive buffer, and rotate it back so our own rank is in its proper position */
  memcpy(send_data, tmp_data + (group_rank + 1) * size, (group_ranks - group_rank - 1) * size);
  memcpy(send_data + (group_ranks - group_rank - 1) * size, tmp_data, (group_rank + 1) * size);
  for (i = 0; i < group_ranks; i++) {
    memcpy((char*)recvbuf + i * size, send_data + (group_ranks - i - 1) * size, size);
  }

  /* free off our internal data structures */
  dtcmp_free(&tmp_data);
  dtcmp_free(&recv_data);
  dtcmp_free(&send_data);

  return 0;
}
