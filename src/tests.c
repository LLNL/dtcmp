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

#define SIZE (50)

typedef int(*sort_local_fn)(const void* inbuf, void* outbuf, int size, MPI_Datatype key, MPI_Datatype keysat, DTCMP_Op op);
typedef int(*sort_fn)(const void* inbuf, void* outbuf, int size, MPI_Datatype key, MPI_Datatype keysat, DTCMP_Op op, MPI_Comm comm);
typedef int(*sortv_fn)(const void* inbuf, void* outbuf, int size, MPI_Datatype key, MPI_Datatype keysat, DTCMP_Op op, MPI_Comm comm);
typedef int(*sortz_fn)(const void* inbuf, int incount, void** outbuf, int* outcount, MPI_Datatype key, MPI_Datatype keysat, DTCMP_Op op, MPI_Comm comm, DTCMP_Handle* handle);

#define NUM_SORT_LOCAL_FNS (5)
sort_local_fn sort_local_fns[NUM_SORT_LOCAL_FNS] = {
  DTCMP_Sort_local,
  DTCMP_Sort_local_insertionsort,
  DTCMP_Sort_local_randquicksort,
  DTCMP_Sort_local_mergesort,
  DTCMP_Sort_local_qsort, /* this can only be used for basic types */
};
char* sort_local_names[NUM_SORT_LOCAL_FNS] = {
  "DTCMP_Sort_local",
  "DTCMP_Sort_local_insertionsort",
  "DTCMP_Sort_local_randquicksort",
  "DTCMP_Sort_local_mergesort",
  "DTCMP_Sort_local_qsort",
};

#define NUM_SORT_FNS (3)
sort_fn sort_fns[NUM_SORT_FNS] = {
  DTCMP_Sort,
  DTCMP_Sort_bitonic,
  DTCMP_Sort_allgather,
};
char* sort_names[NUM_SORT_FNS] = {
  "DTCMP_Sort",
  "DTCMP_Sort_bitonic",
  "DTCMP_Sort_allgather",
};

#define NUM_SORTV_FNS (3)
sortv_fn sortv_fns[NUM_SORTV_FNS] = {
  DTCMP_Sortv,
  DTCMP_Sortv_allgather,
  DTCMP_Sortv_sortgather_scatter,
};
char* sortv_names[NUM_SORTV_FNS] = {
  "DTCMP_Sortv",
  "DTCMP_Sortv_allgather",
  "DTCMP_Sortv_sortgather_scatter",
};

#define NUM_SORTZ_FNS (2)
sortz_fn sortz_fns[NUM_SORTZ_FNS] = {
  DTCMP_Sortz,
  DTCMP_Sortz_samplesort,
};
char* sortz_names[NUM_SORTZ_FNS] = {
  "DTCMP_Sortz",
  "DTCMP_Sortz_samplesort",
};

int test_sort_local(
  const char* test,
  sort_local_fn fn,
  const char* name,
  const void* inbuf,
  void* outbuf,
  int size,
  MPI_Datatype key,
  MPI_Datatype keysat,
  DTCMP_Op cmp)
{
  (*fn)(inbuf, outbuf, size, key, keysat, cmp);

  int flag;
  DTCMP_Is_sorted(outbuf, size, key, keysat, cmp, MPI_COMM_SELF, &flag);
  if (flag != 1) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    printf("ERROR DETECTED rank=%d test=%s routine=%s\n", rank, test, name);
  }
  return flag;
}

int test_sort(
  const char* test,
  sort_fn fn,
  const char* name,
  const void* inbuf,
  void* outbuf,
  int size,
  MPI_Datatype key,
  MPI_Datatype keysat,
  DTCMP_Op cmp,
  MPI_Comm comm)
{
  (*fn)(inbuf, outbuf, size, key, keysat, cmp, comm);

  int flag;
  DTCMP_Is_sorted(outbuf, size, key, keysat, cmp, comm, &flag);
  if (flag != 1) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    if (rank == 0) {
      printf("ERROR DETECTED rank=%d test=%s routine=%s\n", rank, test, name);
    }
  }
  return flag;
}

int test_sortv(
  const char* test,
  sortv_fn fn,
  const char* name,
  const void* inbuf,
  void* outbuf,
  int size,
  MPI_Datatype key,
  MPI_Datatype keysat,
  DTCMP_Op cmp,
  MPI_Comm comm)
{
  (*fn)(inbuf, outbuf, size, key, keysat, cmp, comm);

  int flag;
  DTCMP_Is_sorted(outbuf, size, key, keysat, cmp, comm, &flag);
  if (flag != 1) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    if (rank == 0) {
      printf("ERROR DETECTED rank=%d test=%s routine=%s\n", rank, test, name);
    }
  }
  return flag;
}

int test_sortz(
  const char* test,
  sortz_fn fn,
  const char* name,
  const void* inbuf,
  int size,
  MPI_Datatype key,
  MPI_Datatype keysat,
  DTCMP_Op cmp,
  MPI_Comm comm)
{
  void* outbuf;
  int outcount;
  DTCMP_Handle handle;
  (*fn)(inbuf, size, &outbuf, &outcount, key, keysat, cmp, comm, &handle);

  int flag;
  DTCMP_Is_sorted(outbuf, outcount, key, keysat, cmp, comm, &flag);
  if (flag != 1) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    if (rank == 0) {
      printf("ERROR DETECTED rank=%d test=%s routine=%s\n", rank, test, name);
    }
  }

  DTCMP_Free(&handle);
  return flag;
}

int test_all_sorts(
  const char* test,
  const void* inbuf,
  void* outbuf,
  int size,
  MPI_Datatype key,
  MPI_Datatype keysat,
  DTCMP_Op cmp,
  MPI_Comm comm)
{
  int i;

  for (i = 0; i < NUM_SORT_LOCAL_FNS; i++) {
    test_sort_local(test, sort_local_fns[i], sort_local_names[i], inbuf, outbuf, size, key, keysat, cmp);
  }

  for (i = 0; i < NUM_SORT_FNS; i++) {
    test_sort(test, sort_fns[i], sort_names[i], inbuf, outbuf, size, key, keysat, cmp, comm);
  }

  for (i = 0; i < NUM_SORTV_FNS; i++) {
    test_sortv(test, sortv_fns[i], sortv_names[i], inbuf, outbuf, size, key, keysat, cmp, comm);
  }

  for (i = 0; i < NUM_SORTZ_FNS; i++) {
    test_sortz(test, sortz_fns[i], sortz_names[i], inbuf, size, key, keysat, cmp, comm);
  }

  return 0;
}

int main(int argc, char* argv[])
{
  int i;

  MPI_Init(NULL, NULL);
  DTCMP_Init();

  int rank, ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);

/* sorting tests, check result of each with is_sorted, which is assumed to be correct code
 *   all procs contribute zero items
 *   all procs contribute one item
 *   all procs contribute same number of items > 1
 *   all procs contribute diff number of items, but at least one
 *   one proc has zero items
 *   only rank 0 has zero items
 *   only rank N-1 has zero items
 *   every other proc has zero items
 *   randomly select procs with zero items
 *   all procs have zero items
 *
 *   items already in order
 *   items in reverse order
 *
 */

  char* test;
  MPI_Datatype key, keysat;
  int size;
  DTCMP_Op op;
  MPI_Comm comm;

  int inbuf[SIZE], outbuf[SIZE];
  for (i = 0; i < SIZE; i++) {
    inbuf[i] = -(i*10 + rank);
    outbuf[i] = 0;
  }

  key    = MPI_INT;
  keysat = MPI_INT;
  comm   = MPI_COMM_WORLD;

  size = 0;
  test = "0 INT ASCEND";
  op = DTCMP_OP_INT_ASCEND;
  test_all_sorts(test, inbuf, outbuf, size, key, keysat, op, comm);

  size = 1;
  test = "1 INT/INT ASCEND";
  op = DTCMP_OP_INT_ASCEND;
  test_all_sorts(test, inbuf, outbuf, size, key, keysat, op, comm);
  test = "1 INT/INT DESCEND";
  op = DTCMP_OP_INT_DESCEND;
  test_all_sorts(test, inbuf, outbuf, size, key, keysat, op, comm);

  size = SIZE;
  test = "SIZE INT/INT ASCEND";
  op = DTCMP_OP_INT_ASCEND;
  test_all_sorts(test, inbuf, outbuf, size, key, keysat, op, comm);
  test = "SIZE INT/INT DESCEND";
  op = DTCMP_OP_INT_DESCEND;
  test_all_sorts(test, inbuf, outbuf, size, key, keysat, op, comm);

#if 0
  int insatbuf[SIZE*2], outsatbuf[SIZE*2];
  for (i = 0; i < SIZE; i++) {
    insatbuf[i*2+0] = -(i*10 + rank);
//    insatbuf[i*2+0] = 35 - (i%2);
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

  DTCMP_Sortv(insatbuf, outsatbuf, size, type_2int, type_2int, cmp_2int, MPI_COMM_WORLD);
//  DTCMP_Sortv(insatbuf, outsatbuf, size, MPI_INT, type_2int, DTCMP_OP_INT_ASCEND, MPI_COMM_WORLD);
//  DTCMP_Sortv_allgather(insatbuf, outsatbuf, size, MPI_INT, type_2int, DTCMP_OP_INT_ASCEND, MPI_COMM_WORLD);
//  DTCMP_Sortv_sortgather_scatter(insatbuf, outsatbuf, size, MPI_INT, type_2int, DTCMP_OP_INT_ASCEND, MPI_COMM_WORLD);
//  DTCMP_Sortv_sortgather_scatter(insatbuf, outsatbuf, size, type_2int, type_2int, cmp_2int, MPI_COMM_WORLD);
  int sortz_outcount;
  void* sortz_outbuf;
  DTCMP_Handle handle;
  DTCMP_Sortz(insatbuf, size, &sortz_outbuf, &sortz_outcount, type_2int, type_2int, cmp_2int, MPI_COMM_WORLD, &handle);
  DTCMP_Free(&handle);

  MPI_Type_free(&type_2int);

  DTCMP_Op_free(&cmp_2int);
#endif

  DTCMP_Finalize();
  MPI_Finalize();

  return 0;
}
