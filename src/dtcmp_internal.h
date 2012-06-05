/* Copyright (c) 2012, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by Adam Moody <moody20@llnl.gov>.
 * LLNL-CODE-557516.
 * All rights reserved.
 * This file is part of the DTCMP library.
 * For details, see https://github.com/hpc/dtcmp
 * Please also read this file: LICENSE.TXT. */

#ifndef DTCMP_INTERNAL_H_
#define DTCMP_INTERNAL_H_

#include <stdlib.h>
#include <stdint.h>
#include "mpi.h"
#include "dtcmp.h"
#include "dtcmp_ops.h"

/* ---------------------------------------
 * Internal types
 * --------------------------------------- */

/* we allocate one of these structs and return a pointer to it as the value for
 * our DTCMP_Op operation handle */
typedef struct {
  uint32_t magic;   /* special integer value which we can use to verify that handle appears valid */
  uint32_t type;    /* type of DTCMP handle */
  MPI_Datatype key; /* datatype of items being compared */
  DTCMP_Op_fn fn;   /* comparison function pointer */
  MPI_Aint disp;    /* byte displacement from current pointer to start of next type */
  DTCMP_Op series;  /* second comparison handle to be applied if first evaluates to equal */
} DTCMP_Handle_t;

/* ---------------------------------------
 * Globals
 * --------------------------------------- */

/* dup of MPI_COMM_SELF that we need for DTCMP_Memcpy */
extern MPI_Comm dtcmp_comm_self;

/* we call rand_r() to acquire random numbers,
 * and this keeps track of the seed between calls */
extern unsigned dtcmp_rand_seed;

/* ---------------------------------------
 * Utility functions
 * --------------------------------------- */

void* dtcmp_malloc(size_t size, size_t alignment, const char* file, int line);
void dtcmp_free(void*);

/* ---------------------------------------
 * Seach implementations
 * --------------------------------------- */

int DTCMP_Search_local_low_binary(
  const void* target,
  const void* list,
  int low,
  int high,
  MPI_Datatype key,
  MPI_Datatype keysat,
  DTCMP_Op cmp,
  int* flag,
  int* index
);

int DTCMP_Search_local_high_binary(
  const void* target,
  const void* list,
  int low,
  int high,
  MPI_Datatype key,
  MPI_Datatype keysat,
  DTCMP_Op cmp,
  int* flag,
  int* index
);

int DTCMP_Search_local_low_list_binary(
  int num,
  const void* targets,
  const void* list,
  int low,
  int high,
  MPI_Datatype key,
  MPI_Datatype keysat,
  DTCMP_Op cmp,
  int* indicies
);

/* ---------------------------------------
 * Parition implementations
 * --------------------------------------- */

int dtcmp_partition_local_memcpy(
  void* buf,
  void* scratch,
  int pivot,
  int num,
  size_t size,
  DTCMP_Op cmp
);

int DTCMP_Partition_local_dtcpy(
  void* buf,
  int count,
  int inpivot,
  int* outpivot,
  MPI_Datatype key,
  MPI_Datatype keysat,
  DTCMP_Op cmp
);

/* ---------------------------------------
 * Merge implementations
 * --------------------------------------- */

int dtcmp_merge_local_2way_memcpy(
  int num,
  const void* inbufs[],
  int counts[],
  void* outbuf,
  size_t size,
  DTCMP_Op cmp
);

int DTCMP_Merge_local_2way(
  int num,
  const void* inbufs[],
  int counts[],
  void* outbuf,
  MPI_Datatype key,
  MPI_Datatype keysat,
  DTCMP_Op cmp
);

int DTCMP_Merge_local_kway_heap(
  int k,
  const void* inbufs[],
  int counts[],
  void* outbuf,
  MPI_Datatype key,
  MPI_Datatype keysat,
  DTCMP_Op cmp
);

int dtcmp_select_local_ends(
  void* buf,
  int num,
  int k,
  void* item,
  MPI_Datatype key,
  MPI_Datatype keysat,
  DTCMP_Op cmp
);

/* ---------------------------------------
 * Local select implementations
 * --------------------------------------- */

int DTCMP_Select_local_ends(
  const void* buf,
  int num,
  int k,
  void* item,
  MPI_Datatype key,
  MPI_Datatype keysat,
  DTCMP_Op cmp
);

int DTCMP_Select_local_randpartition(
  const void* buf,
  int num,
  int k,
  void* item,
  MPI_Datatype key,
  MPI_Datatype keysat,
  DTCMP_Op cmp
);

/* ---------------------------------------
 * Local sort implementations
 * --------------------------------------- */

int DTCMP_Sort_local_insertionsort(
  const void* inbuf,
  void* outbuf,
  int count,
  MPI_Datatype key,
  MPI_Datatype keysat,
  DTCMP_Op cmp
);

int DTCMP_Sort_local_randquicksort(
  const void* inbuf, 
  void* outbuf,
  int count,
  MPI_Datatype key,
  MPI_Datatype keysat,
  DTCMP_Op cmp
);

int DTCMP_Sort_local_mergesort(
  const void* inbuf,
  void* outbuf,
  int count,
  MPI_Datatype key,
  MPI_Datatype keysat,
  DTCMP_Op cmp
);

int DTCMP_Sort_local_qsort(
  const void* inbuf,
  void* outbuf,
  int count,
  MPI_Datatype key,
  MPI_Datatype keysat,
  DTCMP_Op cmp
);

/* ---------------------------------------
 * Sort implementations
 * --------------------------------------- */

int DTCMP_Sort_allgather(
  const void* inbuf,
  void* outbuf,
  int count,
  MPI_Datatype key,
  MPI_Datatype keysat,
  DTCMP_Op cmp,
  MPI_Comm comm
);

int DTCMP_Sort_bitonic(
  const void* inbuf,
  void* outbuf,
  int count,
  MPI_Datatype key,
  MPI_Datatype keysat,
  DTCMP_Op cmp,
  MPI_Comm comm
);

/* ---------------------------------------
 * Sortv implementations
 * --------------------------------------- */

int DTCMP_Sortv_allgather(
  const void* inbuf,
  void* outbuf,
  int count,
  MPI_Datatype key,
  MPI_Datatype keysat,
  DTCMP_Op cmp,
  MPI_Comm comm
);

int DTCMP_Sortv_sortgather_scatter(
  const void* inbuf,
  void* outbuf,
  int count,
  MPI_Datatype key,
  MPI_Datatype keysat,
  DTCMP_Op cmp,
  MPI_Comm comm
);

int DTCMP_Sortv_cheng(
  const void* inbuf,
  void* outbuf,
  int count,
  MPI_Datatype key,
  MPI_Datatype keysat,
  DTCMP_Op cmp,
  int group_rank,
  int group_ranks,
  const int* comm_ranklist,
  MPI_Comm comm
);

/* ---------------------------------------
 * Rankv implementations
 * --------------------------------------- */

int DTCMP_Rankv_sort(
  int count,
  const void* buf,
  int* groups,
  int  group_id[],
  int  group_ranks[],
  int  group_rank[],
  MPI_Datatype key,
  MPI_Datatype keysat,
  DTCMP_Op cmp,
  MPI_Comm comm
);

int DTCMP_Rankv_strings_sort(
  int count,
  const char* strings[],
  int* groups,
  int  group_id[],
  int  group_ranks[],
  int  group_rank[],
  MPI_Comm comm
);

#endif /* DTCMP_INTERNAL_H_ */
