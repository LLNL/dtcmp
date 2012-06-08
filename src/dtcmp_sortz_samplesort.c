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

static int find_splitters(
  void* buf,
  int count,
  int s,
  void* splitters,
  MPI_Datatype key,
  MPI_Datatype keysat,
  DTCMP_Op cmp,
  MPI_Comm comm)
{
  int i;

  /* get my rank and number of ranks in comm */
  int rank, ranks;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &ranks);

  /* get lower bound and extent of key */
  MPI_Aint key_lb, key_extent;
  MPI_Aint key_true_lb, key_true_extent;
  MPI_Type_get_extent(key, &key_lb, &key_extent);
  MPI_Type_get_true_extent(key, &key_true_lb, &key_true_extent);

  /* get lower bound and extent of keysat */
  MPI_Aint keysat_lb, keysat_extent;
  MPI_Aint keysat_true_lb, keysat_true_extent;
  MPI_Type_get_extent(keysat, &keysat_lb, &keysat_extent);
  MPI_Type_get_true_extent(keysat, &keysat_true_lb, &keysat_true_extent);

  /* pick "s" regularly spaced samples */
  size_t samples_size = s * key_true_extent;
  char* samples = (char*) dtcmp_malloc(samples_size, 0, __FILE__, __LINE__);
  for (i = 0; i < s; i++) {
    int sample_index = (((i+1) * count) / s) - 1;
    char* src = buf + sample_index * keysat_extent;
    char* dst = samples + i * key_extent;
    DTCMP_Memcpy(dst, 1, key, src, 1, key);
  }

  /* gather samples to root */
  int num_all_samples = ranks * s;
  size_t all_samples_size = num_all_samples * key_true_extent;
  char* all_samples = NULL;
  if (rank == 0) {
    all_samples = (char*) dtcmp_malloc(all_samples_size, 0, __FILE__, __LINE__);
  }
  MPI_Gather(samples, s, key, all_samples, s, key, 0, comm);

  /* TODO: we could replace this with a merge since they are already
   * ordered from each process, or replace this with a sorting
   * gather */
  /* sort samples at root */
  if (rank == 0) {
    DTCMP_Sort_local(DTCMP_IN_PLACE, all_samples, num_all_samples, key, key, cmp);
  }

  /* pick ranks-1 splitters */
  int num_splitters = ranks - 1;
  if (rank == 0) {
    /* first splitter is s units in */
    for (i = 0; i < num_splitters; i++) {
      int splitter_index = (i+1) * s;
      char* src = all_samples + splitter_index * key_extent;
      char* dst = splitters + i * key_extent;
      DTCMP_Memcpy(dst, 1, key, src, 1, key);
    }
  }

  /* broadcast splitters to all procs */
  MPI_Bcast(splitters, num_splitters, key, 0, comm);

  /* free memory */
  dtcmp_free(&all_samples);
  dtcmp_free(&samples);

  return DTCMP_SUCCESS;
}

static int split_exchange_merge(
  void* buf,
  int count,
  void** outbuf,
  int* outcount,
  int num_splitters,
  void* splitters,
  MPI_Datatype key,
  MPI_Datatype keysat,
  DTCMP_Op cmp,
  MPI_Comm comm,
  DTCMP_Handle* handle)
{
  int i;

  /* initialize output parameters */
  *outbuf   = NULL;
  *outcount = 0;
  *handle   = DTCMP_HANDLE_NULL;

  /* get my rank and number of ranks in comm */
  int rank, ranks;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &ranks);

  /* get lower bound and extent of keysat */
  MPI_Aint keysat_lb, keysat_extent;
  MPI_Aint keysat_true_lb, keysat_true_extent;
  MPI_Type_get_extent(keysat, &keysat_lb, &keysat_extent);
  MPI_Type_get_true_extent(keysat, &keysat_true_lb, &keysat_true_extent);

  /* search for split locations */
  size_t flags_size    = num_splitters * sizeof(int);
  size_t indicies_size = num_splitters * sizeof(int);
  int* flags    = (int*) dtcmp_malloc(flags_size, 0, __FILE__, __LINE__);
  int* indicies = (int*) dtcmp_malloc(indicies_size, 0, __FILE__, __LINE__);
  DTCMP_Search_low_list_local(num_splitters, splitters, buf, 0, count-1, key, keysat, cmp, flags, indicies);

  /* allocate space to record outgoing counts and to receive incoming counts */
  size_t outgoing_counts_size = ranks * sizeof(int);
  size_t incoming_counts_size = ranks * sizeof(int);
  int* outgoing_counts = (int*) dtcmp_malloc(outgoing_counts_size, 0, __FILE__, __LINE__);
  int* incoming_counts = (int*) dtcmp_malloc(incoming_counts_size, 0, __FILE__, __LINE__);

  /* determine number of elements that we'll send to each process */
  int last_offset = 0;
  for (i = 0; i < num_splitters; i++) {
    outgoing_counts[i] = indicies[i] - last_offset;
    last_offset = indicies[i];
  }
  outgoing_counts[num_splitters] = count - last_offset;

  /* alltoall to collect counts from each processes */
  MPI_Alltoall(outgoing_counts, 1, MPI_INT, incoming_counts, 1, MPI_INT, comm);

  /* compute outgoing and incoming displacements */
  int outgoing_disps_size = ranks * sizeof(int);
  int incoming_disps_size = ranks * sizeof(int);
  int* outgoing_disps = (int*) dtcmp_malloc(outgoing_disps_size, 0, __FILE__, __LINE__);
  int* incoming_disps = (int*) dtcmp_malloc(incoming_disps_size, 0, __FILE__, __LINE__);
  int last_outgoing = 0;
  int last_incoming = 0;
  for (i = 0; i < ranks; i++) {
    outgoing_disps[i] = last_outgoing;
    last_outgoing += outgoing_counts[i];

    incoming_disps[i] = last_incoming;
    last_incoming += incoming_counts[i];
  }
  int total_incoming = last_incoming;

  /* alltoallv to collect items */
  size_t incoming_bytes = total_incoming * keysat_true_extent;
  void* incoming_buf = (void*) dtcmp_malloc(incoming_bytes, 0, __FILE__, __LINE__);
  MPI_Alltoallv(
    buf,          outgoing_counts, outgoing_disps, keysat,
    incoming_buf, incoming_counts, incoming_disps, keysat,
    comm
  );

  /* setup handle and buffer to merge data to return to user */
  void* ret_buf = NULL;
  int ret_buf_size = sizeof(DTCMP_Free_fn) + incoming_bytes;
  if (ret_buf_size > 0) {
    ret_buf = (void*) dtcmp_malloc(ret_buf_size, 0, __FILE__, __LINE__);
    if (ret_buf == NULL) {
      /* TODO: error */
    }
  }

  /* compute and allocate space to merge received data */
  char* ret_buf_tmp = (char*)ret_buf;
  void* merge_buf = NULL;
  if (ret_buf != NULL) {
    /* allocate and initialize function pointer as first item in handle struct */
    DTCMP_Free_fn* fn = (DTCMP_Free_fn*) ret_buf;
    *fn = DTCMP_Free_single;
    ret_buf_tmp += sizeof(DTCMP_Free_fn);

    merge_buf = (void*) ret_buf_tmp;
    ret_buf_tmp += incoming_bytes;
  }

  /* merge p sorted lists */
  size_t inbufs_size = ranks * sizeof(void*);
  const void** inbufs = (const void**) dtcmp_malloc(inbufs_size, 0, __FILE__, __LINE__);
  for (i = 0; i < ranks; i++) {
    inbufs[i] = (void*) ((char*)incoming_buf + incoming_disps[i] * keysat_extent);
  }
  DTCMP_Merge_local(ranks, inbufs, incoming_counts, merge_buf, key, keysat, cmp);

  /* free memory */
  dtcmp_free(&inbufs);
  dtcmp_free(&incoming_buf);
  dtcmp_free(&incoming_disps);
  dtcmp_free(&outgoing_disps);
  dtcmp_free(&incoming_counts);
  dtcmp_free(&outgoing_counts);
  dtcmp_free(&indicies);
  dtcmp_free(&flags);

  /* set return parameters */
  *outbuf   = merge_buf;
  *outcount = total_incoming;
  *handle   = ret_buf;

  return DTCMP_SUCCESS;
}

/* gather all items to each node and sort locally */
int DTCMP_Sortz_samplesort(
  const void* inbuf,
  int incount,
  void** outbuf,
  int* outcount,
  MPI_Datatype key,
  MPI_Datatype keysat,
  DTCMP_Op cmp,
  MPI_Comm comm,
  DTCMP_Handle* handle)
{
  /* TODO: compute appropriate s value */
  /* s = min(incount, 12*ceil(log(incount))) */
  int log_count = 0;
  int size = 1;
  while (size < incount) {
    size <<= 1;
    log_count++;
  }
  int s = 12 * log_count;
  if (s > incount) {
    s = incount;
  }

  /* get my rank and number of ranks in comm */
  int rank, ranks;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &ranks);

  /* get lower bound and extent of key */
  MPI_Aint key_true_lb, key_true_extent;
  MPI_Type_get_true_extent(key, &key_true_lb, &key_true_extent);

  /* get lower bound and extent of keysat */
  MPI_Aint keysat_true_lb, keysat_true_extent;
  MPI_Type_get_true_extent(keysat, &keysat_true_lb, &keysat_true_extent);

  /* copy input data to a temporary buffer */
  size_t tmpbuf_size = incount * keysat_true_extent;
  char* tmpbuf = (char*) dtcmp_malloc(tmpbuf_size, 0, __FILE__, __LINE__);
  DTCMP_Memcpy(tmpbuf, incount, keysat, inbuf, incount, keysat);

  /* local sort */
  DTCMP_Sort_local(DTCMP_IN_PLACE, tmpbuf, incount, key, keysat, cmp);

  /* pick ranks-1 splitters */
  int num_splitters = ranks - 1;
  size_t splitters_size = num_splitters * key_true_extent;
  char* splitters = (char*) dtcmp_malloc(splitters_size, 0, __FILE__, __LINE__);
  find_splitters(tmpbuf, incount, s, splitters, key, keysat, cmp, comm);

  /* split local data, exchange, and merge recevied data */
  split_exchange_merge(tmpbuf, incount, outbuf, outcount, num_splitters, splitters, key, keysat, cmp, comm, handle);

  /* free memory */
  dtcmp_free(&splitters);
  dtcmp_free(&tmpbuf);

  return DTCMP_SUCCESS;
}
