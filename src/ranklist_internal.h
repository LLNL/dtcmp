/* Copyright (c) 2012, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by Adam Moody <moody20@llnl.gov> and Edgar A. Leon <leon@llnl.gov>.
 * LLNL-CODE-557516.
 * All rights reserved.
 * This file is part of the DTCMP library.
 * For details, see https://github.com/hpc/dtcmp
 * Please also read this file: LICENSE.TXT. */

#include "mpi.h"

int ranklist_allreduce_recursive(
  const void* sendbuf,
  void* recvbuf,
  int count,
  MPI_Datatype datatype,
  MPI_Op op,
  int group_rank,
  int group_ranks,
  const int* comm_ranklist,
  MPI_Comm comm
);

int ranklist_allgather_brucks(
  const void* sendbuf,
  void* recvbuf,
  int count,
  MPI_Datatype datatype,
  int group_rank,
  int group_ranks,
  const int* comm_ranklist,
  MPI_Comm comm
);

int ranklist_alltoall_brucks(
  const void* sendbuf,
  void* recvbuf,
  int count,
  MPI_Datatype datatype,
  int group_rank,
  int group_ranks,
  const int* comm_ranklist,
  MPI_Comm comm
);

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
  MPI_Comm comm
);

