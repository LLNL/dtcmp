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

int ranklist_alltoallv_linear(
  const void* sendbuf,
  const int* sendcounts,
  const int* senddispls,
  void* recvbuf,
  const int* recvcounts,
  const int* recvdispls,
  MPI_Datatype datatype,
  int group_rank,
  int group_ranks,
  const int* comm_ranklist,
  MPI_Comm comm)
{
  /* TODO: we could just fire off a bunch of issends */

  /* get true extent of datatype so we can compute offset into buffers */
  MPI_Aint lb, extent;
  MPI_Type_get_extent(datatype, &lb, &extent);

  /* execute the alltoall operation */
  MPI_Request request[2];
  MPI_Status status[2];
  int dist = 0;
  int src_index = group_rank;
  int dst_index = group_rank;
  while (dist < group_ranks) {
    /* get ranks of source and destination processes */
    int src = comm_ranklist[src_index];
    int dst = comm_ranklist[dst_index];

    /* receive data from src */
    char* recv_ptr = (char*)recvbuf + recvdispls[src_index] * extent;
    int recv_count = recvcounts[src_index];
    MPI_Irecv(recv_ptr, recv_count, datatype, src, 0, comm, &request[0]);

    /* send data to dst */
    char* send_ptr = (char*)sendbuf + senddispls[dst_index] * extent;
    int send_count = sendcounts[dst_index];
    MPI_Isend(send_ptr, send_count, datatype, dst, 0, comm, &request[1]);

    /* wait for communication to complete */
    MPI_Waitall(2, request, status);

    /* get index into our send and recv arrays */
    src_index--;
    if (src_index < 0) {
      src_index += group_ranks;
    }
    dst_index++;
    if (dst_index >= group_ranks) {
      dst_index -= group_ranks;
    }

    /* go on to next iteration */
    dist++;
  }
    
  return 0;
}
