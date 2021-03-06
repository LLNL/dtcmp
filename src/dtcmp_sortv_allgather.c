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
int DTCMP_Sortv_allgather(
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

  if (ranks > 0) {
    /* allocate space to hold counts and displacements for allgatherv call */
    int* counts = (int*) dtcmp_malloc(ranks * sizeof(int), 0, __FILE__, __LINE__);
    int* displs = (int*) dtcmp_malloc(ranks * sizeof(int), 0, __FILE__, __LINE__);

    /* gather counts from all ranks */
    MPI_Allgather(&count, 1, MPI_INT, counts, 1, MPI_INT, comm);

    /* compute displacements for all ranks */
    int i;
    int myoffset = 0;
    int offset = 0;
    for (i = 0; i < ranks; i++) {
      displs[i] = offset;
      offset += counts[i];
      if (i < rank) {
        myoffset += counts[i];
      }
    }
    int total_count = offset;

    /* get true extent of keysat type */
    MPI_Aint true_lb, true_extent;
    MPI_Type_get_true_extent(keysat, &true_lb, &true_extent);

    /* allocate space to hold all items from all procs */
    size_t buf_size = total_count * true_extent;
    if (buf_size > 0) {
      char* buf = (char*) dtcmp_malloc(buf_size, 0, __FILE__, __LINE__); 

      /* gather all items, send from outbuf if IN_PLACE is specified */
      void* sendbuf = (void*) inbuf;
      if (inbuf == DTCMP_IN_PLACE) {
        sendbuf = outbuf;
      }
      char* recvbuf = buf - true_lb;
      MPI_Allgatherv(sendbuf, count, keysat, (void*)recvbuf, counts, displs, keysat, comm);

      /* sort items with local sort */
      DTCMP_Sort_local(DTCMP_IN_PLACE, recvbuf, total_count, key, keysat, cmp, hints);

      /* copy our items into outbuf */
      char* mybuf = recvbuf + myoffset * true_extent;
      DTCMP_Memcpy(outbuf, count, keysat, (void*)mybuf, count, keysat);

      /* free off our temporary buffers */
      dtcmp_free(&buf);
    }

    /* free off our temporary buffers */
    dtcmp_free(&displs);
    dtcmp_free(&counts);
  }

  return DTCMP_SUCCESS;
}
