/* Copyright (c) 2012, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by Adam Moody <moody20@llnl.gov>.
 * LLNL-CODE-557516.
 * All rights reserved.
 * This file is part of the DTCMP library.
 * For details, see https://github.com/hpc/dtcmp
 * Please also read this file: LICENSE.TXT. */

#include "mpi.h"
#include "dtcmp_internal.h"

/* gather all items to each node and sort locally */
int DTCMP_Sortv_cheng(
  const void* inbuf,
  void* outbuf,
  int count,
  MPI_Datatype key,
  MPI_Datatype keysat,
  DTCMP_Op cmp,
  DTCMP_Flags hints,
  MPI_Comm comm)
{
  /* get our rank and the number of ranks in this comm */
  int rank, ranks;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &ranks);

  /* allocate an array large enough to hold a list of ranks */
  int* ranklist = (int*) dtcmp_malloc(ranks * sizeof(int), 0, __FILE__, __LINE__);

  /* fill in the rank list */
  int i;
  for (i = 0; i < ranks; i++) {
    ranklist[i] = i;
  }

  /* call ranklist_cheng sort */
  DTCMP_Sortv_ranklist_cheng(
    inbuf, outbuf, count, key, keysat, cmp, hints,
    rank, ranks, ranklist, comm
  );

  /* free the ranklist */
  dtcmp_free(&ranklist);

  return DTCMP_SUCCESS;
}
