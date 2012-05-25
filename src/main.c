/* Copyright (c) 2012, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by Adam Moody <moody20@llnl.gov>.
 * LLNL-CODE-557516.
 * All rights reserved.
 * This file is part of the DTCMP library.
 * For details, see https://github.com/hpc/dtcmp
 * Please also read this file: LICENSE.TXT. */

#include <stdlib.h>
#include <stdio.h>

#include "mpi.h"
#include "dtcmp.h"
#include "dtcmp_internal.h"

#define SIZE (5)
//#define SIZE (2)

int main(int argc, char* argv[])
{
  int i;
  int flag;

  MPI_Init(NULL, NULL);
  DTCMP_Init();

  int rank, ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);

  int inbuf[SIZE], outbuf[SIZE];
  for (i = 0; i < SIZE; i++) {
    inbuf[i] = -(i*10 + rank);
    //inbuf[i] = -(i*10);
    outbuf[i] = 0;
  }

  int size = SIZE;
  if (rank % 2 == 1) {
    size = 3;
  }
  size = rank;

//  DTCMP_Sortv_combined(inbuf, outbuf, size, MPI_INT, MPI_INT, DTCMP_OP_INT_DESCEND, MPI_COMM_WORLD);
  //DTCMP_Sort_local_combined(inbuf, outbuf, size, MPI_INT, MPI_INT, DTCMP_OP_INT_DESCEND);
//  DTCMP_Sort_combined_bitonic(inbuf, outbuf, size, MPI_INT, MPI_INT, DTCMP_OP_INT_ASCEND, MPI_COMM_WORLD);
  for (i = 0; i < size; i++) {
    printf("%d: item %d = %d\n", rank, i, outbuf[i]);
  }

  int num_groups, group_ids[SIZE], group_ranks[SIZE], group_rank[SIZE];
  DTCMP_Rankv_combined(
    size, inbuf,
    &num_groups, group_ids, group_ranks, group_rank,
    MPI_INT, MPI_INT, DTCMP_OP_INT_DESCEND, MPI_COMM_WORLD
  );
  for (i = 0; i < size; i++) {
    printf("%d: item %d = %d, groups=%d, group=%d, ranks=%d, rank=%d\n",
      rank, i, inbuf[i], num_groups, group_ids[i], group_ranks[i], group_rank[i]
    );
  }

  const char* strings[4] = {"hello", "my", "mean", "world!"};
  DTCMP_Rankv_strings(
    size, strings,
    &num_groups, group_ids, group_ranks, group_rank,
    MPI_COMM_WORLD
  );
  for (i = 0; i < size; i++) {
    printf("%d: item %d = %s, groups=%d, group=%d, ranks=%d, rank=%d\n",
      rank, i, strings[i], num_groups, group_ids[i], group_ranks[i], group_rank[i]
    );
  }

  DTCMP_Is_sorted_local(inbuf, size, MPI_INT, MPI_INT, DTCMP_OP_INT_ASCEND, &flag);
  printf("%d: flag = %d\n", rank, flag);

  int insatbuf[SIZE*2], outsatbuf[SIZE*2];
  for (i = 0; i < SIZE; i++) {
//    insatbuf[i*2+0] = -(i*10 + rank);
    insatbuf[i*2+0] = 35 - (i%2);
    insatbuf[i*2+1] = +(i*10 + rank);
    outsatbuf[i*2+0] = 0;
    outsatbuf[i*2+1] = 0;
  }

  DTCMP_Op cmp_2int;
  DTCMP_Op_create_series(DTCMP_OP_INT_ASCEND, DTCMP_OP_INT_DESCEND, &cmp_2int);

  MPI_Datatype type_2int;
  MPI_Type_contiguous(2, MPI_INT, &type_2int);
//  MPI_Type_vector(2, 1, 2, MPI_INT, &type_2int);
  MPI_Type_commit(&type_2int);

  DTCMP_Sortv_combined(insatbuf, outsatbuf, size, type_2int, type_2int, cmp_2int, MPI_COMM_WORLD);
//  DTCMP_Sortv_combined(insatbuf, outsatbuf, size, MPI_INT, type_2int, DTCMP_OP_INT_ASCEND, MPI_COMM_WORLD);
//  DTCMP_Sortv_combined_allgather(insatbuf, outsatbuf, size, MPI_INT, type_2int, DTCMP_OP_INT_ASCEND, MPI_COMM_WORLD);
//  DTCMP_Sortv_combined_sortgather_scatter(insatbuf, outsatbuf, size, MPI_INT, type_2int, DTCMP_OP_INT_ASCEND, MPI_COMM_WORLD);
//  DTCMP_Sortv_combined_sortgather_scatter(insatbuf, outsatbuf, size, type_2int, type_2int, cmp_2int, MPI_COMM_WORLD);

  for (i = 0; i < size; i++) {
    printf("%d: item %d = %d(%d)\n", rank, i, outsatbuf[i*2+0], outsatbuf[i*2+1]);
  }

  DTCMP_Is_sorted(outsatbuf, size, type_2int, type_2int, cmp_2int, MPI_COMM_WORLD, &flag);
//  DTCMP_Is_sorted(outsatbuf, size, type_2int, type_2int, DTCMP_OP_INT_ASCEND, MPI_COMM_WORLD, &flag);
  if (rank == 0) {
    printf("%d: flag = %d\n", rank, flag);
  }

  int index;
  int target[2];
  target[0] = 35 - (4%2);
  target[1] = +(4*10 + rank);

  DTCMP_Search_low_combined(target, outsatbuf, 0, size-1, MPI_INT, type_2int, DTCMP_OP_INT_DESCEND, &flag, &index);
//  DTCMP_Search_low_combined(target, outsatbuf, 0, size-1, type_2int, type_2int, cmp_2int, &flag, &index);
  printf("%d: flag %d, index %d\n", rank, flag, index);

  DTCMP_Search_high_combined(target, outsatbuf, 0, size-1, MPI_INT, type_2int, DTCMP_OP_INT_DESCEND, &flag, &index);
//  DTCMP_Search_high_combined(target, outsatbuf, 0, size-1, type_2int, type_2int, cmp_2int, &flag, &index);
  printf("%d: flag %d, index %d\n", rank, flag, index);

  MPI_Type_free(&type_2int);

  DTCMP_Op_free(&cmp_2int);

  DTCMP_Finalize();
  MPI_Finalize();

  return 0;
}
