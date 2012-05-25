/* Copyright (c) 2012, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by Adam Moody <moody20@llnl.gov>.
 * LLNL-CODE-557516.
 * All rights reserved.
 * This file is part of the DTCMP library.
 * For details, see https://github.com/hpc/dtcmp
 * Please also read this file: LICENSE.TXT. */

#include <stdlib.h>
#include <string.h>
#include "mpi.h"

/* given a comm, our rank, and a list of ranks, issue an allgather using Brucks algorithm */
int ranklist_allgather_brucks(
  const void* sendbuf,
  void* recvbuf,
  int num,
  MPI_Datatype datatype,
  int group_rank,
  int group_ranks,
  const int* comm_ranklist,
  MPI_Comm comm)
{
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

  /* free some temporary space to work with */
  int tmpbuf_size = size * group_ranks;
  char* tmpbuf = (char*) malloc(tmpbuf_size);
  if (tmpbuf == NULL) {
    /* TODO: fail */
  }

  /* copy our own data into the receive buffer */
  memcpy(tmpbuf, sendbuf, size);

  /* execute the allgather operation */
  MPI_Request request[2];
  MPI_Status status[2];
  int step = 1;
  int count_received = 1;
  while (step < group_ranks) {
    /* get indicies for left and right partners */
    int src_index = group_rank + step;
    if (src_index >= group_ranks) {
      src_index -= group_ranks;
    }
    int dst_index = group_rank - step;
    if (dst_index < 0) {
      dst_index += group_ranks;
    }

    /* get ranks for left and right partners */
    int src = comm_ranklist[src_index];
    int dst = comm_ranklist[dst_index];

    /* determine number of elements we'll be sending and receiving in this round */
    int count = step;
    if (count_received + count > group_ranks) {
      count = group_ranks - count_received;
    }

    /* receive data from source */
    MPI_Irecv(tmpbuf + count_received * size, count * num, datatype, src, 0, comm, &request[0]); 

    /* send the data to destination */
    MPI_Isend(tmpbuf, count * num, datatype, dst, 0, comm, &request[1]);

    /* wait for communication to complete */
    MPI_Waitall(2, request, status);

    /* add the count to the total number we've received */
    count_received += count;

    /* go on to next iteration */
    step <<= 1;
  }

  /* shift our data back to the proper position in receive buffer */
  memcpy((char*)recvbuf + group_rank * size, tmpbuf, (group_ranks - group_rank) * size);
  memcpy(recvbuf, (char*)tmpbuf + (group_ranks - group_rank) * size, group_rank * size);

  /* free the temporary buffer */
  if (tmpbuf != NULL) {
    free(tmpbuf);
    tmpbuf = NULL;
  }

  return 0;
}
