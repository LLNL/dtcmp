/* Copyright (c) 2012, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by Adam Moody <moody20@llnl.gov>.
 * LLNL-CODE-557516.
 * All rights reserved.
 * This file is part of the DTCMP library.
 * For details, see https://github.com/hpc/dtcmp
 * Please also read this file: LICENSE.TXT. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpi.h"
#include "dtcmp_internal.h"
#include "ranklist_internal.h"

/* This implementation is based on ideas from:
 *
 * "A Novel Parallel Sorting Algorithm for Contemporary Architectures", 2007
 * David R. Cheng, Viral B. Shah, John R. Gilbert, Alan Edelman
 *
 * "A Scalable MPI_Comm_split Algorithm for Exascale Computing", 2010
 * Paul Sack, William Gropp
 *
 * Notable difference from Cheng's design:
 * - When computing the median of medians, each process is responsible
 *   for finding one splitter.  We use an all-to-all to distribute
 *   the contribution from each process for each splitter.  Further
 *   this all-to-all is implemented using Bruck's algorithm and it
 *   sorts elements as part of the communication algorithm.  An allgather
 *   is used to collect the proposed splitter from each process. */

#define THRESH (1)
#define LT (0)
#define EQ (1)

/* when we compute the weighted median, we may get some median values
 * that have a zero count.  We should avoid calling the compare function
 * for these medians, as the actual value may be garbage, which may crash
 * the comparator function, so we check the count before the median. */
static int int_with_key_cmp_fn(const void* a, const void* b)
{
  /* check that the counts for both elements are not zero,
   * if a count is zero, consider that element to be higher
   * (throws zeros to back of list) */
  int count_a = *(int*) a;
  int count_b = *(int*) b;
  if (count_a != 0 && count_b != 0) {
    /* both elements have non-zero counts,
     * so compare the median values to each other */
    return 0;
  } else if (count_b != 0) {
    /* count of first element is zero (but not the second),
     * so say the first element is larger */
    return 1;
  } else if (count_a != 0) {
    /* count of second element is zero (but not the first),
     * so say the second element is larger */
    return -1;
  } else {
    /* counts of both elements are zero, they are equivalent */
    return 0;
  }
}

/* each process computes weighted median for different splitter */
static void compute_weighted_median(
  const void* my_num_with_median,
  void* out_num_with_median,
  void* scratch,
  MPI_Datatype key,
  DTCMP_Op keycmp,
  MPI_Datatype type_int_with_key,
  DTCMP_Op cmp_int_with_key,
  int group_rank,
  int group_ranks,
  const int* comm_ranklist,
  MPI_Comm comm)
{
  int i;

  /* get true extent of key */
  MPI_Aint key_true_lb, key_true_extent;
  MPI_Type_get_true_extent(key, &key_true_lb, &key_true_extent);
  size_t size_int_with_key = sizeof(int) + key_true_extent;

  /* set up pointers to scratch space */
  char* num_with_median     = (char*)scratch;
  char* all_num_with_median = (char*)scratch + size_int_with_key;

  /* scatter weights and medians to different ranks */
  ranklist_alltoall_brucks(
    my_num_with_median, all_num_with_median, 1, type_int_with_key,
    group_rank, group_ranks, comm_ranklist, comm
  );

  /* sort by medians value (ensuring that count is non-zero) */
  DTCMP_Sort_local(
    DTCMP_IN_PLACE, all_num_with_median, group_ranks,
    type_int_with_key, type_int_with_key, cmp_int_with_key
  );

  /* compute total number of elements */
  int N = 0;
  for (i = 0; i < group_ranks; i++) {
    int cnt = *(int*) (all_num_with_median + i * size_int_with_key);
    N += cnt;
  }

  /* identify the weighted median */
  i = 0;
  int before_weight = 0;
  int half_weight   = N / 2;
  char* ptr = all_num_with_median;
  void* target = (void*) (ptr + sizeof(int));
  while(i < group_ranks) {
    /* set our target to the current median
     * and initialize our current weight */
    target = (void*) (ptr + sizeof(int));
    int current_weight = *(int*) ptr;
    i++;
    ptr += size_int_with_key;

    /* add weights for any elements which equal this current median */
    if (i < group_ranks) {
      int result;
      int next_weight = *(int*) ptr;
      if (next_weight > 0) {
        void* next_target = (void*) (ptr + sizeof(int));
        result = dtcmp_op_eval(target, next_target, keycmp);
      } else {
        result = 0;
      }
      while (i < group_ranks && result == 0) {
        /* current item is equal to target, add its weight */
        current_weight += next_weight;
        i++;
        ptr += size_int_with_key;

        /* get weight and comparison result of next item if one exists */
        if (i < group_ranks) {
          next_weight = *(int*) ptr;
          if (next_weight > 0) {
            void* next_target = (void*) (ptr + sizeof(int));
            result = dtcmp_op_eval(target, next_target, keycmp);
          } else {
            result = 0;
          }
        }
      }
    }

    /* determine if the weight before and after this value are
     * each less than or equal to half */
    int after_weight = N - before_weight - current_weight;
    if (before_weight <= half_weight && after_weight <= half_weight) {
      break;
    }

    /* after was too heavy, so add current weight to before
     * and go to next value */
    before_weight += current_weight;
  }

  /* set total number of active elements,
   * and copy the median to our allgather send buffer */
  int* num = (int*) num_with_median;
  *num = N;
  memcpy(num_with_median + sizeof(int), target, key_true_extent);

  /* broadcast medians */
  ranklist_allgather_brucks(
    num_with_median, out_num_with_median, 1, type_int_with_key,
    group_rank, group_ranks, comm_ranklist, comm
  );

  return;
}

static int find_exact_splitters(
  const void* data,
  int n,
  MPI_Datatype key,
  MPI_Datatype keysat,
  DTCMP_Op cmp,
  int serial_search_threshold,
  void* splitters,
  int splitters_valid[],
  int group_rank,
  int group_ranks,
  const int* comm_ranklist,
  MPI_Comm comm)
{
  int i;

  /* get extent of keysat type so we can compute offsets into data buffer */
  MPI_Aint keysat_lb, keysat_extent;
  MPI_Type_get_extent(keysat, &keysat_lb, &keysat_extent);

  /* get true extent of key so we can allocate space to copy them into */
  MPI_Aint key_true_lb, key_true_extent;
  MPI_Type_get_true_extent(key, &key_true_lb, &key_true_extent);
  size_t size_int_with_key = sizeof(int) + key_true_extent;

  /* allocate scratch space */

  /* indicies of splitters */
  int* k = dtcmp_malloc(group_ranks * sizeof(int), 0, __FILE__, __LINE__);

  /* tracks current lowest index of each of active range */
  int* index = dtcmp_malloc(group_ranks * sizeof(int), 0, __FILE__, __LINE__);

  /* tracks number of items in each active range */
  int* num = dtcmp_malloc(group_ranks * sizeof(int), 0, __FILE__, __LINE__);

  /* flag per rank signifying its median is exact */
  int* found_exact = dtcmp_malloc(group_ranks * sizeof(int), 0, __FILE__, __LINE__);

  /* candidate median for each splitter with count from local rank */
  char* my_num_with_median = dtcmp_malloc(group_ranks * size_int_with_key, 0, __FILE__, __LINE__); 

  /* candidate splitters after each median-of-medians round */
  char* out_num_with_median = dtcmp_malloc(group_ranks * size_int_with_key, 0, __FILE__, __LINE__);

  /* number of items less than and equal to each candidate median */
  int* counts = dtcmp_malloc(2 * group_ranks * sizeof(int), 0, __FILE__, __LINE__);

  /* total number of items less than and equal to each candidate median */
  int* all_counts = dtcmp_malloc(2 * group_ranks * sizeof(int), 0, __FILE__, __LINE__);

  /* number of items we'll receive from each rank */
  int* recvcounts = dtcmp_malloc(group_ranks * sizeof(int), 0, __FILE__, __LINE__);

  /* displacement of each set of items we'll receive from each rank */
  int* recvdispls = dtcmp_malloc(group_ranks * sizeof(int), 0, __FILE__, __LINE__);

  /* number of items we'll send to each rank */
  int* sendcounts = dtcmp_malloc(group_ranks * sizeof(int), 0, __FILE__, __LINE__);

  /* displacement of each set of items we'll send to each rank */
  int* senddispls = dtcmp_malloc(group_ranks * sizeof(int), 0, __FILE__, __LINE__);

  /* scratch space used to compute median-of-medians */
  void* weighted_median_scratch = dtcmp_malloc((group_ranks + 1) * size_int_with_key, 0, __FILE__, __LINE__);

  /* compute split points based on the number of ranks each task contributes,
   * we want the task to end up with the same number of elements */

  /* gather number of elements each process is contributing
   * (borrow the index array for this task) */
  ranklist_allgather_brucks(
    &n, index, 1, MPI_INT,
    group_rank, group_ranks, comm_ranklist, comm
  );

  /* compute ranks of split points (start from 1) */
  int count = 1;
  for (i = 0; i < group_ranks; i++) {
    k[i] = count;
    count += index[i];
  }

  /* initialize our active ranges */
  for (i = 0; i < group_ranks; i++) {
    index[i] = 0;
    num[i] = n;
    found_exact[i] = 0;
  }

  /* create new comparison operation to compare number, then key,
   * don't compare key if either count is zero, since key may be garbage */
  DTCMP_Op cmp_count_nonzero, cmp_int_with_key;
  DTCMP_Op_create(MPI_INT, int_with_key_cmp_fn, &cmp_count_nonzero);
  DTCMP_Op_create_series(cmp_count_nonzero, cmp, &cmp_int_with_key);

  /* create and commit datatype to represent concatenation of int with key */
  MPI_Datatype type_int_with_key;
  dtcmp_type_concat2(MPI_INT, key, &type_int_with_key);

  while (1) {
    /* for each range, pick a median and record number of active elements
     * we have in that range */
    char* ptr = my_num_with_median;
    for (i = 0; i < group_ranks; i++) {
      /* record number of active elements for this range */
      memcpy(ptr, &num[i], sizeof(int));
      if (num[i] > 0) {
        /* if there is at least one element, copy in median element */
        int median_index = (num[i] / 2) + index[i];
        if (median_index < n) {
          /* TODO: for general key types, we need to use DTCMP_Memcpy here
           * and adjust for lb */
          const void* median = (char*)data + median_index * keysat_extent;
          memcpy(ptr + sizeof(int), median, key_true_extent);
        }
      }
      ptr += size_int_with_key;
    }

    /* for each rank, compute the weighted median, M, and total number
     * of active elements, N */
    compute_weighted_median(
      (const void*)my_num_with_median, (void*)out_num_with_median, weighted_median_scratch,
      key, cmp, type_int_with_key, cmp_int_with_key,
      group_rank, group_ranks, comm_ranklist, comm
    );

    /* stop if for each range, N is below threshold or if we found
     * an exact match */
    int can_break = 1;
    for (i = 0; i < group_ranks; i++) {
      int N = *(int*) (out_num_with_median + i * size_int_with_key);
      if (N <= 1) {
        found_exact[i] = 1;
      }
      if (N > serial_search_threshold && !found_exact[i]) {
        can_break = 0;
      }
    }
    if (can_break) {
      break;
    }

    /* compute counts of elements less-than, equal-to, and greater-than M */
    for (i = 0; i < group_ranks; i++) {
      /* if we already found the split point for this range,
       * or if we don't have any active elements, go to next */
      if (found_exact[i] || num[i] == 0) {
        counts[i*2+LT] = 0;
        counts[i*2+EQ] = 0;
        continue;
      }

      /* get proposed split point */
      void* target = out_num_with_median + i * size_int_with_key + sizeof(int);

      /* use binary search to determine indicies denoting elements
       * less-than and greater than M */
      int start_index = index[i];
      int end_index   = index[i] + num[i] - 1;
      int flag, lowest, highest;
      DTCMP_Search_low_local(target,  data, start_index, end_index, key, keysat, cmp, &flag, &lowest);
      DTCMP_Search_high_local(target, data, lowest,      end_index, key, keysat, cmp, &flag, &highest);
      counts[i*2+LT] = lowest - start_index;
      counts[i*2+EQ] = (highest + 1) - lowest;
    }
    ranklist_allreduce_recursive(
      counts, all_counts, 2 * group_ranks, MPI_INT, MPI_SUM,
      group_rank, group_ranks, comm_ranklist, comm
    );

    /* based on rank of splitter, chop down our active range */
    for (i = 0; i < group_ranks; i++) {
      /* if we already found the split point for this range, go to next */
      if (found_exact[i]) {
        continue;
      }

      if (k[i] <= all_counts[i*2+LT]) {
        /* the target is in the lower portion,
         * exclude all entries equal to or greater than */
        num[i] = counts[i*2+LT];
      } else if (k[i] > (all_counts[i*2+LT] + all_counts[i*2+EQ])){
        /* the target is in the higher portion,
         * exclude all entries equal to or less than */
        int num_lte = counts[i*2+LT] + counts[i*2+EQ];
        index[i] += num_lte;
        num[i]   -= num_lte;
        k[i] = k[i] - (all_counts[i*2+LT] + all_counts[i*2+EQ]);
      } else { /* all_counts[LT] < k && k <= (all_counts[LT] + all_counts[EQ]) */
        /* found our target exactly, we're done */
        found_exact[i] = 1;
      }
    }
  }

  /* free our comparison ops and type */
  DTCMP_Op_free(&cmp_count_nonzero);
  DTCMP_Op_free(&cmp_int_with_key);
  MPI_Type_free(&type_int_with_key);

  /* check whether all values are already exact */
  int all_exact = 1;
  for (i = 0; i < group_ranks; i++) {
    if (!found_exact[i]) {
      all_exact = 0;
      break;
    }
  }

  if (all_exact) {
    /* if we found all values exactly, then just copy them into array */
    for (i = 0; i < group_ranks; i++) {
      splitters_valid[i] = *(int*) (out_num_with_median + i * size_int_with_key);
      if (splitters_valid[i] != 0) {
        memcpy(
          (char*)splitters + i * key_true_extent,
          out_num_with_median + i * size_int_with_key + sizeof(int),
          key_true_extent
        );
      }
    }
  } else {
    /* otherwise, distribute remaining ranges to processes to sort and search */

    /* if we found exact median for any section,
     * we can avoid sending and sorting */
    for (i = 0; i < group_ranks; i++) {
      senddispls[i] = index[i];
      sendcounts[i] = num[i];
      if (found_exact[i]) {
        sendcounts[i] = 0;
      }
    }

    /* inform each rank how many elements we will be sending in alltoallv */
    ranklist_alltoall_brucks(
      sendcounts, recvcounts, 1, MPI_INT,
      group_rank, group_ranks, comm_ranklist, comm
    );

    /* build our displacement arrays for alltoallv call */
    int recvdisp = 0;
    for (i = 0; i < group_ranks; i++) {
      recvdispls[i] = recvdisp;
      recvdisp += recvcounts[i];
    }

    /* TODO: this does not handle non-contig types */
    exit(1);

#if 0
    /* allocate space to receive incoming data */
    void* recvdata = NULL;
    if (recvdisp > 0) {
      recvdata = (void*) malloc(recvdisp * keysize);
      if (recvdata == NULL) {
        /* TODO: fail */
      }
    }

    /* exchange data */
    ranklist_alltoallv_linear(
      data, sendcounts, senddispls,
      recvdata, recvcounts, recvdispls,
      key, group_rank, group_ranks, comm_ranklist, comm
    );

    /* sort elements if we received any */
    void* my_M = out_num_with_median + group_rank * size_int_with_key + sizeof(int);
    if (!found_exact[group_rank] && recvdisp > 0) {
      /* get our local rank in comm for error reporting */
      int comm_rank;
      MPI_Comm_rank(comm, &comm_rank);

      /* verify that the number we received matches the number we expcted to receive */
      int my_N = *(int*) (out_num_with_median + group_rank * size_int_with_key);
      int numrecv = recvdisp;
      if (numrecv != my_N) {
        /* TODO: shouldn't happen */
        printf("%d: ERROR: Rank %d failed to find splitter\n", comm_rank, group_rank);
      }

      /* TODO: convert this to a kway merge */
      DTCMP_Sort_local(DTCMP_IN_PLACE, recvdata, numrecv, key, key, cmp);

      /* identify kth element and broadcast */
      int k_index = k[group_rank] - 1;
      if (k_index < numrecv) {
        my_M = recvdata + k_index * keysize;
      } else {
        /* TODO: shouldn't happen */
        printf("%d: ERROR: looking for kth elementh (%d) when array is only %d long\n",
          comm_rank, k_index, numrecv
        );
      }
    }

    /* each process has now computed one splitter, gather all splitters to all procs */
    ranklist_allgather_brucks(my_M, splitters, 1, key, group_rank, group_ranks, comm_ranklist, comm);

    if (recvdata != NULL) {
      free(recvdata);
      recvdata = NULL;
    }
#endif
  }

  /* free off our scratch space */
  dtcmp_free(&k);
  dtcmp_free(&index);
  dtcmp_free(&num);
  dtcmp_free(&found_exact);
  dtcmp_free(&my_num_with_median);
  dtcmp_free(&out_num_with_median);
  dtcmp_free(&counts);
  dtcmp_free(&all_counts);
  dtcmp_free(&recvcounts);
  dtcmp_free(&recvdispls);
  dtcmp_free(&sendcounts);
  dtcmp_free(&senddispls);
  dtcmp_free(&weighted_median_scratch);

  return DTCMP_SUCCESS;
}

static int exact_split_exchange_merge(
  void* outbuf,
  int count,
  void* splitters,
  int splitters_valid[],
  MPI_Datatype key,
  MPI_Datatype keysat,
  DTCMP_Op cmp,
  int group_rank,
  int group_ranks,
  const int comm_ranklist[],
  MPI_Comm comm)
{
  int i;

  /* number of items we'll send to each proc */
  int* sendcounts = dtcmp_malloc(group_ranks * sizeof(int), 0, __FILE__, __LINE__);

  /* set all send counts to zero */
  for (i = 0; i < group_ranks; i++) {
    sendcounts[i] = 0;
  }

  /* if we have any elements, search for splitter locations and
   * compute send counts for those elements */
  if (count > 0) {
    /* get extent of key */
    MPI_Aint key_lb, key_extent;
    MPI_Type_get_extent(key, &key_lb, &key_extent);

    /* get true extent of key */
    MPI_Aint key_true_lb, key_true_extent;
    MPI_Type_get_true_extent(key, &key_true_lb, &key_true_extent);

    /* arrays to hold values in Search_low_list call */
    char* splitters_search = dtcmp_malloc(group_ranks * key_true_extent, 0, __FILE__, __LINE__);
    int* flags    = dtcmp_malloc(group_ranks * sizeof(int), 0, __FILE__, __LINE__);
    int* indicies = dtcmp_malloc(group_ranks * sizeof(int), 0, __FILE__, __LINE__);

    /* only consider valid splitters */
    int num_splitters = 0;
    for (i = 0; i < group_ranks; i++) {
      if (splitters_valid[i]) {
        char* src = (char*)splitters + i * key_extent;
        char* dst = splitters_search + num_splitters * key_extent;
        DTCMP_Memcpy(dst, 1, key, src, 1, key);
        num_splitters++;
      }
    }

    /* search for index values for these split points in our local data */
    DTCMP_Search_low_list_local(
      num_splitters, splitters_search, outbuf, 0, count-1,
      key, keysat, cmp, flags, indicies
    );

    /* compute send counts based on indicies from search */
    i = 0;
    int j = 0;
    int start_index = 0;
    while (i < group_ranks) {
      if (splitters_valid[i]) {
        /* there may be a non-zero count for this range, look for next
         * valid splitter to get index */
        start_index = indicies[j];
        j++;

        if (j < num_splitters) {
          /* there's at least one more valid splitter out there,
           * find its index */
          int k = i + 1;
          while (k < group_ranks && !splitters_valid[k]) {
            k++;
          }
          sendcounts[i] = indicies[j] - start_index;
          i = k;
        } else {
          /* no more valid splitters, assign whatever is left to
           * this range */
          sendcounts[i] = count - start_index;
          i = group_ranks;
        }
      } else {
        /* leave this range set to a 0 count */
        i++;
      }
    }

    /* free memory used for splitter search */
    dtcmp_free(&indicies);
    dtcmp_free(&flags);
    dtcmp_free(&splitters_search);
  }

  /* number of items we'll receive from each proc */
  int* recvcounts = dtcmp_malloc(group_ranks * sizeof(int), 0, __FILE__, __LINE__);

  /* inform other processes how many elements we'll be sending to each */
  ranklist_alltoall_brucks(
    sendcounts, recvcounts, 1, MPI_INT,
    group_rank, group_ranks, comm_ranklist, comm
  );

  /* displacement for each of our send ranges */
  int* senddispls = dtcmp_malloc(group_ranks * sizeof(int), 0, __FILE__, __LINE__);

  /* displacement for each of our receive ranges */
  int* recvdispls = dtcmp_malloc(group_ranks * sizeof(int), 0, __FILE__, __LINE__);

  /* build our displacement arrays for alltoallv call */
  int recvdisp = 0;
  int senddisp = 0;
  for (i = 0; i < group_ranks; i++) {
    recvdispls[i] = recvdisp;
    recvdisp += recvcounts[i];

    senddispls[i] = senddisp;
    senddisp += sendcounts[i];
  }

  /* get true extent of key with satellite */
  MPI_Aint keysat_true_lb, keysat_true_extent;
  MPI_Type_get_true_extent(keysat, &keysat_true_lb, &keysat_true_extent);

  /* merge buffer to sort items we receive from each proc */
  void* kbuf = dtcmp_malloc(count * keysat_true_extent, 0, __FILE__, __LINE__);

  /* exchange data */
  char* recvdata = kbuf;
  ranklist_alltoallv_linear(
    outbuf, sendcounts, senddispls,
    recvdata, recvcounts, recvdispls,
    keysat, group_rank, group_ranks, comm_ranklist, comm
  );

  /* pointer to set of items we receive from each proc */
  const void** kbufs = dtcmp_malloc(group_ranks * sizeof(void*), 0, __FILE__, __LINE__);

  /* number of items we receive from each proc */
  int* ksizes = dtcmp_malloc(group_ranks * sizeof(int), 0, __FILE__, __LINE__);

  /* if we received any elements, sort them */
  if (recvdisp > 0) {
    /* set up pointers and sizes to our k buffers */
    size_t koffset = 0;
    for (i = 0; i < group_ranks; i++) {
      int elements = recvcounts[i];
      kbufs[i]  = recvdata + koffset;
      ksizes[i] = elements;
      koffset += elements * keysat_true_extent;
    }
    DTCMP_Merge_local(group_ranks, kbufs, ksizes, outbuf, key, keysat, cmp);
  }

  /* free memory */
  dtcmp_free(&ksizes);
  dtcmp_free(&kbufs);
  dtcmp_free(&kbuf);
  dtcmp_free(&recvdispls);
  dtcmp_free(&senddispls);
  dtcmp_free(&recvcounts);
  dtcmp_free(&sendcounts);

  return DTCMP_SUCCESS;
}

int DTCMP_Sortv_ranklist_cheng(
  const void* inbuf,
  void* outbuf,
  int count,
  MPI_Datatype key,
  MPI_Datatype keysat,
  DTCMP_Op cmp,
  int group_rank,
  int group_ranks,
  const int comm_ranklist[],
  MPI_Comm comm)
{
  /* TODO: currently only works if each process has at least one element */

  /* TODO: incorporate scans to avoid this requirement */
  /* ensure all elements are unique */
  void* uniqbuf;
  MPI_Datatype uniqkey, uniqkeysat;
  DTCMP_Op uniqcmp;
  DTCMP_Handle uniqhandle;

  int input_not_unique = 1;
  if (input_not_unique) {
    /* determine which buffer input data is in */
    const void* tmpbuf = inbuf;
    if (inbuf == DTCMP_IN_PLACE) {
      tmpbuf = outbuf;
    }

    /* tack on rank and original index to each item and return new key
     * and keysat types along with new comparison op */
    dtcmp_uniqify(
      tmpbuf, count, key, keysat, cmp,
      &uniqbuf, &uniqkey, &uniqkeysat, &uniqcmp,
      comm, &uniqhandle
    );
  } else {
    /* move data to output buffer */
    if (inbuf != DTCMP_IN_PLACE) {
      DTCMP_Memcpy(outbuf, count, keysat, inbuf, count, keysat);
    }

    /* set unique variables to point to output buffer and existing
     * types */
    uniqbuf    = outbuf;
    uniqkey    = key;
    uniqkeysat = keysat;
    uniqcmp    = cmp;
  }

  /* local sort */
  DTCMP_Sort_local(DTCMP_IN_PLACE, uniqbuf, count, uniqkey, uniqkeysat, uniqcmp);

  /* get true extent of the keys */
  MPI_Aint key_true_lb, key_true_extent;
  MPI_Type_get_true_extent(uniqkey, &key_true_lb, &key_true_extent);

  /* hold key for each process representing exact splitter,
   * all items equal to or less than should be sent to corresponding proc */
  void* splitters = dtcmp_malloc(group_ranks * key_true_extent, 0, __FILE__, __LINE__);
  int*  valid     = dtcmp_malloc(group_ranks * sizeof(int), 0, __FILE__, __LINE__);

  /* compute global split points across all data */
  find_exact_splitters(
    uniqbuf, count, uniqkey, uniqkeysat, uniqcmp, THRESH, splitters, valid,
    group_rank, group_ranks, comm_ranklist, comm
  );

  /* now split data, exchange, and merge */
  exact_split_exchange_merge(
    uniqbuf, count, splitters, valid, uniqkey, uniqkeysat, uniqcmp,
    group_rank, group_ranks, comm_ranklist, comm
  );

  /* strip off extra data we attached to ensure items are unique */
  if (input_not_unique) {
    dtcmp_deuniqify(uniqbuf, count, uniqkey, uniqkeysat, outbuf, key, keysat, &uniqhandle);
  }

  /* free our scratch space */
  dtcmp_free(&valid);
  dtcmp_free(&splitters);

  return 0;
}
