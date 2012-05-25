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
#include "dtcmp_internal.h"

int ranklist_allreduce_recursive(
  const void* sendbuf,
  void* recvbuf,
  int count,
  MPI_Datatype datatype,
  MPI_Op op,
  int group_rank,
  int group_ranks,
  const int* comm_ranklist,
  MPI_Comm comm)
{
  /* get our rank within comm */
  int comm_rank = comm_ranklist[group_rank];

  /* build type that is count consecutive entries of datatype */
  MPI_Datatype type_contig;
  MPI_Type_contiguous(count, datatype, &type_contig);

  /* get true extent of datatype */
  MPI_Aint lb, extent;
  MPI_Type_get_true_extent(type_contig, &lb, &extent);

  /* now that we have the extent, free off the type */
  MPI_Type_free(&type_contig);

  /* allocate buffer to receive partial results */
  int buf_size = extent;
  int* tmp = NULL;
  if (buf_size > 0) {
    tmp = (int*) malloc(buf_size);
    if (tmp == NULL) {
      /* TODO: fail */
    }
  }

  MPI_Request request[2];
  MPI_Status status[2];

  /* copy our data into the receive buffer */
  if (buf_size > 0) {
    MPI_Sendrecv(
      (void*)sendbuf, count, datatype, comm_rank, 0,
      recvbuf, count, datatype, comm_rank, 0,
      comm, status
    );
  }

  /* execute recursive doubling operation */
  int mask = 1;
  while (mask < group_ranks) {
    /* compute index of partner */
    int partner_index = group_rank ^ mask;
    if (partner_index < group_ranks) {
      /* get rank of partner in comm */
      int partner = comm_ranklist[partner_index];

      /* exchange data with partner */
      MPI_Irecv(tmp,     count, datatype, partner, 0, comm, &request[0]);
      MPI_Isend(recvbuf, count, datatype, partner, 0, comm, &request[1]);
      MPI_Waitall(2, request, status);

      /* reduce data (smallest rank first) */
      if (partner_index < group_rank) {
        /* reduce into partner's data and copy back to our recvbuf */
        MPI_Reduce_local(recvbuf, tmp, count, datatype, op);
        MPI_Sendrecv(
          tmp,     count, datatype, comm_rank, 0,
          recvbuf, count, datatype, comm_rank, 0,
          comm, status
        );
      } else {
        /* otherwise, we are the smaller rank, reduce partner data into ours */
        MPI_Reduce_local(tmp, recvbuf, count, datatype, op);
      }
    }

    /* prepare for next iteration */
    mask <<= 1;
  }

  return 0;
}
