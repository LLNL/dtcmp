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

#define NUM_SORTV_FNS (4)
sortv_fn sortv_fns[NUM_SORTV_FNS] = {
  DTCMP_Sortv,
  DTCMP_Sortv_allgather,
  DTCMP_Sortv_cheng,
  DTCMP_Sortv_sortgather_scatter,
};
char* sortv_names[NUM_SORTV_FNS] = {
  "DTCMP_Sortv",
  "DTCMP_Sortv_allgather",
  "DTCMP_Sortv_cheng",
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

int test_variable_sorts(
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
 *   only rank 0 has zero items
 *   only rank N-1 has zero items
 *   every other proc has zero items
 *   ** randomly select procs with zero items
 *
 *   ** repeat all of above with duplicates
 *
 *   items already in order
 *   items in reverse order
 *   items in random order
 *
 */

  char* test;
  void* inbuf;
  void* outbuf;
  int size;
  MPI_Datatype key, keysat;
  DTCMP_Op op;
  MPI_Comm comm;


  int in_1int[SIZE], out_1int[SIZE];
  inbuf  = (void*) in_1int;
  outbuf = (void*) out_1int;

  for (i = 0; i < SIZE; i++) {
    in_1int[i] = -(i*10 + rank);
    out_1int[i] = 0;
  }

  key    = MPI_INT;
  keysat = MPI_INT;
  comm   = MPI_COMM_WORLD;

  /* test that all sorts work with count=0 on all procs */
  size = 0;
  test = "0 INT ASCEND";
  op = DTCMP_OP_INT_ASCEND;
  test_all_sorts(test, inbuf, outbuf, size, key, keysat, op, comm);

  /* test that all sorts work with count=1 on all procs */
  size = 1;
  test = "1 INT/INT ASCEND";
  op = DTCMP_OP_INT_ASCEND;
  test_all_sorts(test, inbuf, outbuf, size, key, keysat, op, comm);
  test = "1 INT/INT DESCEND";
  op = DTCMP_OP_INT_DESCEND;
  test_all_sorts(test, inbuf, outbuf, size, key, keysat, op, comm);

  /* test that all sorts work with count>1 on all procs */
  size = SIZE;
  test = "SIZE INT/INT ASCEND";
  op = DTCMP_OP_INT_ASCEND;
  test_all_sorts(test, inbuf, outbuf, size, key, keysat, op, comm);
  test = "SIZE INT/INT DESCEND";
  op = DTCMP_OP_INT_DESCEND;
  test_all_sorts(test, inbuf, outbuf, size, key, keysat, op, comm);

  for (i = 0; i < SIZE; i++) {
    in_1int[i]  = 1;
    out_1int[i] = 0;
  }

  /* test that all sorts work with count=0 on all procs */
  size = 0;
  test = "0 INT ASCEND DUP";
  op = DTCMP_OP_INT_ASCEND;
  test_all_sorts(test, inbuf, outbuf, size, key, keysat, op, comm);

  /* test that all sorts work with count=1 on all procs */
  size = 1;
  test = "1 INT/INT ASCEND DUP";
  op = DTCMP_OP_INT_ASCEND;
  test_all_sorts(test, inbuf, outbuf, size, key, keysat, op, comm);
  test = "1 INT/INT DESCEND DUP";
  op = DTCMP_OP_INT_DESCEND;
  test_all_sorts(test, inbuf, outbuf, size, key, keysat, op, comm);

  /* test that all sorts work with count>1 on all procs */
  size = SIZE;
  test = "SIZE INT/INT ASCEND DUP";
  op = DTCMP_OP_INT_ASCEND;
  test_all_sorts(test, inbuf, outbuf, size, key, keysat, op, comm);
  test = "SIZE INT/INT DESCEND DUP";
  op = DTCMP_OP_INT_DESCEND;
  test_all_sorts(test, inbuf, outbuf, size, key, keysat, op, comm);


  MPI_Datatype type_2int;
  MPI_Type_contiguous(2, MPI_INT, &type_2int);
  MPI_Type_commit(&type_2int);

  DTCMP_Op cmp_updown, cmp_downdown;
  DTCMP_Op_create_series(DTCMP_OP_INT_ASCEND, DTCMP_OP_INT_DESCEND, &cmp_updown);
  DTCMP_Op_create_series(DTCMP_OP_INT_DESCEND, DTCMP_OP_INT_DESCEND, &cmp_downdown);

  int in_2int[SIZE*2], out_2int[SIZE*2];
  inbuf  = (void*) in_2int;
  outbuf = (void*) out_2int;

  for (i = 0; i < SIZE; i++) {
    in_2int[i*2+0]  = -(i*10 + rank);
    in_2int[i*2+1]  = +(i*10 + rank);
    out_2int[i*2+0] = 0;
    out_2int[i*2+1] = 0;
  }

  key    = MPI_INT;
  keysat = type_2int;
  comm   = MPI_COMM_WORLD;

  /* test that all sorts work with count=0 on all procs */
  size = 0;
  test = "0 INT/2INT ASCEND";
  op = DTCMP_OP_INT_ASCEND;
  test_all_sorts(test, inbuf, outbuf, size, key, keysat, op, comm);

  /* test that all sorts work with count=1 on all procs */
  size = 1;
  test = "1 INT/2INT ASCEND";
  op = DTCMP_OP_INT_ASCEND;
  test_all_sorts(test, inbuf, outbuf, size, key, keysat, op, comm);
  test = "1 INT/2INT DESCEND";
  op = DTCMP_OP_INT_DESCEND;
  test_all_sorts(test, inbuf, outbuf, size, key, keysat, op, comm);

  /* test that all sorts work with count>1 on all procs */
  size = SIZE;
  test = "SIZE INT/2INT ASCEND";
  op = DTCMP_OP_INT_ASCEND;
  test_all_sorts(test, inbuf, outbuf, size, key, keysat, op, comm);
  test = "SIZE INT/2INT DESCEND";
  op = DTCMP_OP_INT_DESCEND;
  test_all_sorts(test, inbuf, outbuf, size, key, keysat, op, comm);

  key    = type_2int;
  keysat = type_2int;
  comm   = MPI_COMM_WORLD;

  /* test that all sorts work with count=0 on all procs */
  size = 0;
  test = "0 2INT/2INT ASCEND";
  op = cmp_updown;
  test_all_sorts(test, inbuf, outbuf, size, key, keysat, op, comm);

  /* test that all sorts work with count=1 on all procs */
  size = 1;
  test = "1 2INT/2INT ASCEND";
  op = cmp_updown;
  test_all_sorts(test, inbuf, outbuf, size, key, keysat, op, comm);
  test = "1 2INT/2INT DESCEND";
  op = cmp_downdown;
  test_all_sorts(test, inbuf, outbuf, size, key, keysat, op, comm);

  /* test that all sorts work with count>1 on all procs */
  size = SIZE;
  test = "SIZE 2INT/2INT ASCEND";
  op = cmp_updown;
  test_all_sorts(test, inbuf, outbuf, size, key, keysat, op, comm);
  test = "SIZE 2INT/2INT DESCEND";
  op = cmp_downdown;
  test_all_sorts(test, inbuf, outbuf, size, key, keysat, op, comm);

  key    = MPI_INT;
  keysat = type_2int;
  comm   = MPI_COMM_WORLD;

  /* test variable counts across procs where 1 <= count <= 10 */
  size = rank % 10 + 1;
  if (size > SIZE) {
    printf("Invalid size %d limit %d\n", size, SIZE);
    return 1;
  }
  test = "1-10 INT/2INT ASCEND";
  op = DTCMP_OP_INT_ASCEND;
  test_variable_sorts(test, inbuf, outbuf, size, key, keysat, op, comm);
  test = "1-10 INT/2INT DESCEND";
  op = DTCMP_OP_INT_DESCEND;
  test_variable_sorts(test, inbuf, outbuf, size, key, keysat, op, comm);

  /* test variable counts where count=0 on rank=0, 1 <= count <= 10 elsewhere */
  size = rank % 10 + 1;
  if (rank == 0) {
    size = 0;
  }
  if (size > SIZE) {
    printf("Invalid size %d limit %d\n", size, SIZE);
    return 1;
  }
  test = "0,1-10 INT/2INT ASCEND";
  op = DTCMP_OP_INT_ASCEND;
  test_variable_sorts(test, inbuf, outbuf, size, key, keysat, op, comm);
  test = "0,1-10 INT/2INT DESCEND";
  op = DTCMP_OP_INT_DESCEND;
  test_variable_sorts(test, inbuf, outbuf, size, key, keysat, op, comm);

  /* test variable counts where count=0 on rank=N-1, 1 <= count <= 10 elsewhere */
  size = rank % 10 + 1;
  if (rank == ranks-1) {
    size = 0;
  }
  if (size > SIZE) {
    printf("Invalid size %d limit %d\n", size, SIZE);
    return 1;
  }
  test = "1-10,0 INT/2INT ASCEND";
  op = DTCMP_OP_INT_ASCEND;
  test_variable_sorts(test, inbuf, outbuf, size, key, keysat, op, comm);
  test = "1-10,0 INT/2INT DESCEND";
  op = DTCMP_OP_INT_DESCEND;
  test_variable_sorts(test, inbuf, outbuf, size, key, keysat, op, comm);

  /* test variable counts where count=0 on even ranks, 1 <= count <= 10 elsewhere */
  size = rank % 10 + 1;
  if (rank % 2 == 0) {
    size = 0;
  }
  if (size > SIZE) {
    printf("Invalid size %d limit %d\n", size, SIZE);
    return 1;
  }
  test = "1-10,0 even INT/2INT ASCEND";
  op = DTCMP_OP_INT_ASCEND;
  test_variable_sorts(test, inbuf, outbuf, size, key, keysat, op, comm);
  test = "1-10,0 even INT/2INT DESCEND";
  op = DTCMP_OP_INT_DESCEND;
  test_variable_sorts(test, inbuf, outbuf, size, key, keysat, op, comm);

  for (i = 0; i < SIZE; i++) {
    in_2int[i*2+0]  = 1;
    in_2int[i*2+1]  = 2;
    out_2int[i*2+0] = 0;
    out_2int[i*2+1] = 0;
  }

  key    = MPI_INT;
  keysat = type_2int;
  comm   = MPI_COMM_WORLD;

  /* test that all sorts work with count=0 on all procs */
  size = 0;
  test = "0 INT/2INT ASCEND DUP";
  op = DTCMP_OP_INT_ASCEND;
  test_all_sorts(test, inbuf, outbuf, size, key, keysat, op, comm);

  /* test that all sorts work with count=1 on all procs */
  size = 1;
  test = "1 INT/2INT ASCEND DUP";
  op = DTCMP_OP_INT_ASCEND;
  test_all_sorts(test, inbuf, outbuf, size, key, keysat, op, comm);
  test = "1 INT/2INT DESCEND DUP";
  op = DTCMP_OP_INT_DESCEND;
  test_all_sorts(test, inbuf, outbuf, size, key, keysat, op, comm);

  /* test that all sorts work with count>1 on all procs */
  size = SIZE;
  test = "SIZE INT/2INT ASCEND DUP";
  op = DTCMP_OP_INT_ASCEND;
  test_all_sorts(test, inbuf, outbuf, size, key, keysat, op, comm);
  test = "SIZE INT/2INT DESCEND DUP";
  op = DTCMP_OP_INT_DESCEND;
  test_all_sorts(test, inbuf, outbuf, size, key, keysat, op, comm);

  key    = type_2int;
  keysat = type_2int;
  comm   = MPI_COMM_WORLD;

  /* test that all sorts work with count=0 on all procs */
  size = 0;
  test = "0 2INT/2INT ASCEND DUP";
  op = cmp_updown;
  test_all_sorts(test, inbuf, outbuf, size, key, keysat, op, comm);

  /* test that all sorts work with count=1 on all procs */
  size = 1;
  test = "1 2INT/2INT ASCEND DUP";
  op = cmp_updown;
  test_all_sorts(test, inbuf, outbuf, size, key, keysat, op, comm);
  test = "1 2INT/2INT DESCEND DUP";
  op = cmp_downdown;
  test_all_sorts(test, inbuf, outbuf, size, key, keysat, op, comm);

  /* test that all sorts work with count>1 on all procs */
  size = SIZE;
  test = "SIZE 2INT/2INT ASCEND DUP";
  op = cmp_updown;
  test_all_sorts(test, inbuf, outbuf, size, key, keysat, op, comm);
  test = "SIZE 2INT/2INT DESCEND DUP";
  op = cmp_downdown;
  test_all_sorts(test, inbuf, outbuf, size, key, keysat, op, comm);

  key    = MPI_INT;
  keysat = type_2int;
  comm   = MPI_COMM_WORLD;

  /* test variable counts across procs where 1 <= count <= 10 */
  size = rank % 10 + 1;
  if (size > SIZE) {
    printf("Invalid size %d limit %d\n", size, SIZE);
    return 1;
  }
  test = "1-10 INT/2INT ASCEND DUP";
  op = DTCMP_OP_INT_ASCEND;
  test_variable_sorts(test, inbuf, outbuf, size, key, keysat, op, comm);
  test = "1-10 INT/2INT DESCEND DUP";
  op = DTCMP_OP_INT_DESCEND;
  test_variable_sorts(test, inbuf, outbuf, size, key, keysat, op, comm);

  /* test variable counts where count=0 on rank=0, 1 <= count <= 10 elsewhere */
  size = rank % 10 + 1;
  if (rank == 0) {
    size = 0;
  }
  if (size > SIZE) {
    printf("Invalid size %d limit %d\n", size, SIZE);
    return 1;
  }
  test = "0,1-10 INT/2INT ASCEND DUP";
  op = DTCMP_OP_INT_ASCEND;
  test_variable_sorts(test, inbuf, outbuf, size, key, keysat, op, comm);
  test = "0,1-10 INT/2INT DESCEND DUP";
  op = DTCMP_OP_INT_DESCEND;
  test_variable_sorts(test, inbuf, outbuf, size, key, keysat, op, comm);

  /* test variable counts where count=0 on rank=N-1, 1 <= count <= 10 elsewhere */
  size = rank % 10 + 1;
  if (rank == ranks-1) {
    size = 0;
  }
  if (size > SIZE) {
    printf("Invalid size %d limit %d\n", size, SIZE);
    return 1;
  }
  test = "1-10,0 INT/2INT ASCEND DUP";
  op = DTCMP_OP_INT_ASCEND;
  test_variable_sorts(test, inbuf, outbuf, size, key, keysat, op, comm);
  test = "1-10,0 INT/2INT DESCEND DUP";
  op = DTCMP_OP_INT_DESCEND;
  test_variable_sorts(test, inbuf, outbuf, size, key, keysat, op, comm);

  /* test variable counts where count=0 on even ranks, 1 <= count <= 10 elsewhere */
  size = rank % 10 + 1;
  if (rank % 2 == 0) {
    size = 0;
  }
  if (size > SIZE) {
    printf("Invalid size %d limit %d\n", size, SIZE);
    return 1;
  }
  test = "1-10,0 even INT/2INT ASCEND DUP";
  op = DTCMP_OP_INT_ASCEND;
  test_variable_sorts(test, inbuf, outbuf, size, key, keysat, op, comm);
  test = "1-10,0 even INT/2INT DESCEND DUP";
  op = DTCMP_OP_INT_DESCEND;
  test_variable_sorts(test, inbuf, outbuf, size, key, keysat, op, comm);

  MPI_Type_free(&type_2int);
  DTCMP_Op_free(&cmp_downdown);
  DTCMP_Op_free(&cmp_updown);

  DTCMP_Finalize();
  MPI_Finalize();

  return 0;
}
