/* Copyright (c) 2012, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by Adam Moody <moody20@llnl.gov>.
 * LLNL-CODE-557516.
 * All rights reserved.
 * This file is part of the DTCMP library.
 * For details, see https://github.com/hpc/dtcmp
 * Please also read this file: LICENSE.TXT. */

#include <string.h>
#include "mpi.h"
#include "dtcmp_internal.h"

#define DETECT_NEXT  (0)
#define DETECT_VALID (1)

#define ASSIGN_VALID  (0)
#define ASSIGN_FLAG   (1)
#define ASSIGN_NEXT   (2)

/* computes first-in-group and last-in-group flags for each local item,
 * disregarding first-in-group for first item and last-in-group for
 * last item since those can be affected by items on other procs,
 * we do this separately from detect_edges to (eventually) reuse code
 * for Rank_local */
static int detect_edges_interior(
  const void* buf,
  int num,
  MPI_Datatype item,
  DTCMP_Op cmp,
  DTCMP_Flags hints,
  int first_in_group[],
  int last_in_group[])
{
  /* get extent of item */
  MPI_Aint lb, extent;
  MPI_Type_get_extent(item, &lb, &extent);

  /* we can directly compute the edges for all elements but the endpoints */
  int i;
  const char* left_data  = (const char*)buf;
  const char* right_data = (const char*)buf + extent;
  for (i = 1; i < num; i++) {
    int result = dtcmp_op_eval(left_data, right_data, cmp);
    if (result != 0) {
      /* the left item is different from the right item,
       * mark the right item as the leader of a new group,
       * and mark the left as the last of its group */
      first_in_group[i]  = 1;
      last_in_group[i-1] = 1;
    } else {
      /* the items are equal, so the right is not the start of a group,
       * nor is the left the end of a group */
      first_in_group[i]  = 0;
      last_in_group[i-1] = 0;
    }

    /* point to the next element */
    left_data = right_data;
    right_data += extent;
  }

  return DTCMP_SUCCESS;
}

static int detect_edges(
  const void* buf,
  int num,
  MPI_Datatype item,
  DTCMP_Op cmp,
  DTCMP_Flags hints,
  int first_in_group[],
  int last_in_group[],
  MPI_Comm comm)
{
  /* if every process has at least one item, we'd just need to send values
   * to the ranks to our left and right sides O(1), but since
   * some ranks may not have any items, we must execute a double scan
   * instead O(log N) */

  /* get our rank and number of ranks in the communicator */
  int rank, ranks;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &ranks);

  /* get extent of item */
  MPI_Aint lb, extent;
  MPI_Type_get_extent(item, &lb, &extent);

  /* get true extent of item */
  MPI_Aint true_lb, true_extent;
  MPI_Type_get_true_extent(item, &true_lb, &true_extent);

  /* compute the size of the scan item, we'll send the rank the
   * process should exchange data with in the next round,
   * a flag to indicate whether the item is valid, and the item itself */
  size_t scan_buf_size = 2 * sizeof(int) + true_extent;

  /* build type for exchange */
  MPI_Datatype type_item;
  MPI_Datatype types[3];
  types[0] = MPI_INT;
  types[1] = MPI_INT;
  types[2] = item;
  dtcmp_type_concat(3, types, &type_item);

  /* declare pointers to our scan buffers */
  char* recv_left_buf  = dtcmp_malloc(scan_buf_size, 0, __FILE__, __LINE__);
  char* recv_right_buf = dtcmp_malloc(scan_buf_size, 0, __FILE__, __LINE__);
  char* send_left_buf  = dtcmp_malloc(scan_buf_size, 0, __FILE__, __LINE__);
  char* send_right_buf = dtcmp_malloc(scan_buf_size, 0, __FILE__, __LINE__);

  /* we can directly compute the edges for all elements but the endpoints */
  detect_edges_interior(buf, num, item, cmp, hints, first_in_group, last_in_group);

  /* to compute the edges for our endpoints we use left-to-right and
   * right-to-left scan operations -- we can't just do point-to-point
   * with our left and right neighbors because our left and/or right
   * neighbor may not have any elements */

  /* for the scan element, we specify:
   *   (int)  a flag indicating whether the element value is valid,
   *   (size) a copy of the element
   * Note that the next rank with the element and the next rank for
   * communication may be different since some ranks may not have a value. */

  /* to compute the leader flag for the leftmost element,
   * we execute a left-to-right scan */
  int* send_left_ints  = (int*) (send_left_buf);
  int* send_right_ints = (int*) (send_right_buf);
  int* recv_left_ints  = (int*) (recv_left_buf);
  int* recv_right_ints = (int*) (recv_right_buf);

  /* get pointers to first and last items in buffer */
  char* buf_first = (char*)buf;
  char* buf_last  = (char*)buf + (num-1) * extent;

  /* copy the value from our rightmost element into our left-to-right
   * scan buffer assume that we don't have a valid value */
  send_left_ints[DETECT_VALID]  = 0;
  send_right_ints[DETECT_VALID] = 0;
  if (true_extent > 0 && num > 0) {
    /* copy value from our leftmost element into right-to-left scan buffer */
    send_left_ints[DETECT_VALID] = 1;
    DTCMP_Memcpy(send_left_buf + 2 * sizeof(int), 1, item, buf_first, 1, item);

    /* copy value from our rightmost element into left-to-right scan buffer */
    send_right_ints[DETECT_VALID] = 1;
    DTCMP_Memcpy(send_right_buf + 2 * sizeof(int), 1, item, buf_last, 1, item);
  }

  int left_rank = rank - 1;
  if (left_rank < 0) {
    left_rank = MPI_PROC_NULL;
  }
  int right_rank = rank + 1;
  if (right_rank >= ranks) {
    right_rank = MPI_PROC_NULL;
  }

  /* execute left-to-right and right-to-left scans */
  MPI_Request request[4];
  MPI_Status  status[4];
  int made_left_comparison  = 0;
  int made_right_comparison = 0;
  while (left_rank != MPI_PROC_NULL || right_rank != MPI_PROC_NULL) {
    int k = 0;

    /* if we have a left partner, send him our left-going data
     * and recv his right-going data */
    if (left_rank != MPI_PROC_NULL) {
      /* receive right-going data from the left */
      MPI_Irecv(recv_left_buf, 1, type_item, left_rank, 0, comm, &request[k]);
      k++;

      /* inform rank to our left of the rank on our right,
       * and send him our data */
      send_left_ints[DETECT_NEXT] = right_rank;
      MPI_Isend(send_left_buf, 1, type_item, left_rank, 0, comm, &request[k]);
      k++;
    }

    /* if we have a right partner, send him our right-going data
     * and recv his left-going data */
    if (right_rank != MPI_PROC_NULL) {
      /* receive left-going data from the right */
      MPI_Irecv(recv_right_buf, 1, type_item, right_rank, 0, comm, &request[k]);
      k++;

      /* inform rank to our right of the rank on our left,
       * and send him our data */
      send_right_ints[DETECT_NEXT] = left_rank;
      MPI_Isend(send_right_buf, 1, type_item, right_rank, 0, comm, &request[k]);
      k++;
    }

    /* wait for all communication to complete */
    if (k > 0) {
      MPI_Waitall(k, request, status);
    }

    /* if we have a left partner, merge his data with our right-going data */
    if (left_rank != MPI_PROC_NULL) {
      if (true_extent > 0 && num > 0 && !made_left_comparison && recv_left_ints[DETECT_VALID]) {
        /* compare our leftmost value with the value we receive from the left,
         * if our value is different, mark it as the start of a new group */
        int result = dtcmp_op_eval(recv_left_buf + 2 * sizeof(int), buf_first, cmp);
        if (result != 0) {
          first_in_group[0] = 1;
        } else {
          first_in_group[0] = 0;
        }

        /* after we make this comparison, we don't need to again,
         * we just need to check with the first rank to our left
         * that has a valid value */
        made_left_comparison = 1;
      }

      /* get the next rank to send to on our left */
      left_rank = recv_left_ints[DETECT_NEXT];
    }

    /* if we have a right partner, merge his data with our left-going data */
    if (right_rank != MPI_PROC_NULL) {
      if (true_extent > 0 && num > 0 && !made_right_comparison && recv_right_ints[DETECT_VALID]) {
        /* compare our rightmost value with the value we receive from the right,
         * if our value is different, mark it as the end of a group */
        int result = dtcmp_op_eval(recv_right_buf + 2 * sizeof(int), buf_last, cmp);
        if (result != 0) {
          last_in_group[num-1] = 1;
        } else {
          last_in_group[num-1] = 0;
        }
        made_right_comparison = 1;
      }

      /* get the next rank to send to on our right */
      right_rank = recv_right_ints[DETECT_NEXT];
    }
  }

  /* if we have an element but never made a comparison on one side,
   * our element is the first or last in its group */
  if (num > 0) {
    /* if we never made a comparison on the left,
     * there is no value to our left, so our leftmost item is the first */
    if (! made_left_comparison) {
      first_in_group[0] = 1;
    }

    /* if we never made a comparison on the right, 
     * there is no value to our right, so our rightmost item is the last */
    if (! made_right_comparison) {
      last_in_group[num-1] = 1;
    }
  }

  /* free our temporary data */
  dtcmp_free(&send_right_buf);
  dtcmp_free(&send_left_buf);
  dtcmp_free(&recv_right_buf);
  dtcmp_free(&recv_left_buf);

  MPI_Type_free(&type_item);

  return DTCMP_SUCCESS;
}

/* Executes a segmented exclusive scan on items in buf.
 * Executes specified MPI operation on satellite data
 * for items whose keys are equal.  Overwrites satellite
 * data with result. Items are assumed to be in sorted
 * order. */
int DTCMP_Segmented_exscan(
  int count,
  const void* keybuf,
  MPI_Datatype key,
  const void* valbuf,
  void* ltrbuf,
  void* rtlbuf,
  MPI_Datatype val,
  DTCMP_Op cmp,
  DTCMP_Flags hints,
  MPI_Op op,
  MPI_Comm comm)
{
  int i, tmp_rc;
  int rc = DTCMP_SUCCESS;

  /* build items to be sorted with return address */
  int*  first_in_group = dtcmp_malloc(count * sizeof(int), 0, __FILE__, __LINE__);
  int*  last_in_group  = dtcmp_malloc(count * sizeof(int), 0, __FILE__, __LINE__);

  /* detect edges across sorted items, note we pass in the full items,
   * which includes (key,rank,index) but we just use the key cmp operation */
  tmp_rc = detect_edges(
    keybuf, count, key, cmp, hints,
    first_in_group, last_in_group, comm
  );
  if (tmp_rc != DTCMP_SUCCESS) {
    rc = tmp_rc;
  }

  /* get our rank and number of ranks in the communicator */
  int rank, ranks;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &ranks);

  /* build type for exchange */
  MPI_Datatype type_item;
  MPI_Datatype types[5];
  types[0] = MPI_INT; /* whether incoming key/value is valid */
  types[1] = MPI_INT; /* flag indicating whether the start of the current group is set */
  types[2] = MPI_INT; /* next MPI rank to talk to */
  types[3] = key;     /* user key */
  types[4] = val;     /* incoming scan value */
  dtcmp_type_concat(5, types, &type_item);

  /* get extent of key type */
  MPI_Aint key_lb, key_extent;
  MPI_Type_get_extent(key, &key_lb, &key_extent);

  /* get true extent of key type */
  MPI_Aint key_true_lb, key_true_extent;
  MPI_Type_get_true_extent(key, &key_true_lb, &key_true_extent);

  /* get extent of val type */
  MPI_Aint val_lb, val_extent;
  MPI_Type_get_extent(val, &val_lb, &val_extent);

  /* get true extent of val type */
  MPI_Aint val_true_lb, val_true_extent;
  MPI_Type_get_extent(val, &val_true_lb, &val_true_extent);

  /* compute the size of the scan item, we'll send a flag indicating
   * whether the value is valid, a flag indicating whether the value
   * is the start of a new group, the next rank a process should exchange
   * data with, and finally a copy of the keysat with current value */
  size_t scan_buf_size = 3 * sizeof(int) + key_true_extent + val_true_extent;

  /* allocate memory for scan buffers */
  void* scan_ltr_recv = dtcmp_malloc(scan_buf_size, 0, __FILE__, __LINE__);
  void* scan_ltr_low  = dtcmp_malloc(scan_buf_size, 0, __FILE__, __LINE__);
  void* scan_ltr_high = dtcmp_malloc(scan_buf_size, 0, __FILE__, __LINE__);
  void* scan_rtl_recv = dtcmp_malloc(scan_buf_size, 0, __FILE__, __LINE__);
  void* scan_rtl_low  = dtcmp_malloc(scan_buf_size, 0, __FILE__, __LINE__);
  void* scan_rtl_high = dtcmp_malloc(scan_buf_size, 0, __FILE__, __LINE__);

  /* we need a temporary to accumulate scan values */
  void* scan_tmp_val = dtcmp_malloc(val_true_extent, 0, __FILE__, __LINE__);

  /* get pointers to int portion of buffers */
  int* scan_ltr_recv_ints = (int*) scan_ltr_recv;
  int* scan_ltr_low_ints  = (int*) scan_ltr_low;
  int* scan_ltr_high_ints = (int*) scan_ltr_high;
  int* scan_rtl_recv_ints = (int*) scan_rtl_recv;
  int* scan_rtl_low_ints  = (int*) scan_rtl_low;
  int* scan_rtl_high_ints = (int*) scan_rtl_high;

  /* get pointers to keysat portion of buffers */
  char* scan_ltr_recv_key = (char*)scan_ltr_recv + 3 * sizeof(int);
  char* scan_ltr_low_key  = (char*)scan_ltr_low  + 3 * sizeof(int);
  char* scan_ltr_high_key = (char*)scan_ltr_high + 3 * sizeof(int);
  char* scan_rtl_recv_key = (char*)scan_rtl_recv + 3 * sizeof(int);
  char* scan_rtl_low_key  = (char*)scan_rtl_low  + 3 * sizeof(int);
  char* scan_rtl_high_key = (char*)scan_rtl_high + 3 * sizeof(int);

  /* TODO: this assumes the satellite datatype starts at the end of the normal
   * extent of the key type */
  /* get pointers to sat portion of buffers */
  char* scan_ltr_recv_val = (char*)scan_ltr_recv + 3 * sizeof(int) + key_true_extent;
  char* scan_ltr_low_val  = (char*)scan_ltr_low  + 3 * sizeof(int) + key_true_extent;
  char* scan_ltr_high_val = (char*)scan_ltr_high + 3 * sizeof(int) + key_true_extent;
  char* scan_rtl_recv_val = (char*)scan_rtl_recv + 3 * sizeof(int) + key_true_extent;
  char* scan_rtl_low_val  = (char*)scan_rtl_low  + 3 * sizeof(int) + key_true_extent;
  char* scan_rtl_high_val = (char*)scan_rtl_high + 3 * sizeof(int) + key_true_extent;

  /* we assume we do not start a new group */
  scan_ltr_high_ints[ASSIGN_FLAG] = 0;
  scan_rtl_high_ints[ASSIGN_FLAG] = 0;

  /* run through the elements we have locally and prepare
   * our left-to-right and right-to-left scan items */
  for (i = 0; i < count; i++) {
    /* get pointer to current value in user's input buffer */
    char* inval = (char*)valbuf + i * val_extent;
    if (first_in_group[i]) {
      /* left edge of new group, set flag */
      scan_ltr_high_ints[ASSIGN_FLAG] = 1;

      /* copy user data into high buf */
      DTCMP_Memcpy(scan_ltr_high_val, 1, val, inval, 1, val);
    } else {
      /* item is not the start of a new group */
      if (i > 0) {
        /* accumulate user's input in high buffer,
         * current = temp + current (keep order) */
        DTCMP_Memcpy(scan_tmp_val, 1, val, inval, 1, val);
        MPI_Reduce_local(scan_ltr_high_val, scan_tmp_val, 1, val, op);
        DTCMP_Memcpy(scan_ltr_high_val, 1, val, scan_tmp_val, 1, val);
      } else {
        /* copy first item into high buffer */
        DTCMP_Memcpy(scan_ltr_high_val, 1, val, inval, 1, val);
      }
    }

    /* get pointer to current value in user's input buffer */
    int j = count - 1 - i;
    char* endval = (char*)valbuf + j * val_extent;
    if (last_in_group[j]) {
      /* right edge of new group, set flag */
      scan_rtl_high_ints[ASSIGN_FLAG] = 1;

      /* copy user data into high buf */
      DTCMP_Memcpy(scan_rtl_high_val, 1, val, endval, 1, val);
    } else {
      /* item is not the start of a new group */
      if (i > 0) {
        /* accumulate user's input in high buffer,
         * current = temp + current (keep order) */
        DTCMP_Memcpy(scan_tmp_val, 1, val, endval, 1, val);
        MPI_Reduce_local(scan_rtl_high_val, scan_tmp_val, 1, val, op);
        DTCMP_Memcpy(scan_rtl_high_val, 1, val, scan_tmp_val, 1, val);
      } else {
        /* copy first item into high buffer */
        DTCMP_Memcpy(scan_rtl_high_val, 1, val, endval, 1, val);
      }
    }
  }

  /* our low values aren't valid unless we receive a valid message */
  scan_ltr_low_ints[ASSIGN_VALID] = 0;
  scan_rtl_low_ints[ASSIGN_VALID] = 0;

  /* our high scan values are valid if we have at least one item */
  scan_ltr_high_ints[ASSIGN_VALID] = 0;
  scan_rtl_high_ints[ASSIGN_VALID] = 0;
  if (count > 0) {
    /* if we have at least one item, our high values are valid */
    scan_ltr_high_ints[ASSIGN_VALID] = 1;
    scan_rtl_high_ints[ASSIGN_VALID] = 1;

    /* copy key of last item into left-to-right scan message */
    char* lastkey = (char*)keybuf + (count - 1) * key_extent;
    DTCMP_Memcpy(scan_ltr_high_key, 1, key, lastkey, 1, key);

    /* copy key of first item into right-to-left scan message */
    char* firstkey = (char*)keybuf;
    DTCMP_Memcpy(scan_rtl_high_key, 1, key, firstkey, 1, key);
  }

  /* get first rank to our left if we have one */
  int left_rank  = rank - 1;
  if (left_rank < 0) {
    left_rank = MPI_PROC_NULL;
  }

  /* get first rank to our right if we have one */
  int right_rank = rank + 1;
  if (right_rank >= ranks) {
    right_rank = MPI_PROC_NULL;
  }

  /* execute exclusive scan in both directions */
  MPI_Request request[4];
  MPI_Status  status[4];
  while (left_rank != MPI_PROC_NULL || right_rank != MPI_PROC_NULL) {
    /* select our left and right partners for this iteration */
    int k = 0;

    /* send and receive data with left partner */
    if (left_rank != MPI_PROC_NULL) {
      MPI_Irecv(scan_ltr_recv, 1, type_item, left_rank, 0, comm, &request[k]);
      k++;

      scan_rtl_high_ints[ASSIGN_NEXT] = right_rank;
      MPI_Isend(scan_rtl_high, 1, type_item, left_rank, 0, comm, &request[k]);
      k++;
    }

    /* send and receive data with right partner */
    if (right_rank != MPI_PROC_NULL) {
      MPI_Irecv(scan_rtl_recv, 1, type_item, right_rank, 0, comm, &request[k]);
      k++;

      scan_ltr_high_ints[ASSIGN_NEXT] = left_rank;
      MPI_Isend(scan_ltr_high, 1, type_item, right_rank, 0, comm, &request[k]);
      k++;
    }

    /* wait for communication to finsih */
    if (k > 0) {
      MPI_Waitall(k, request, status);
    }

    /* reduce data from left partner */
    if (left_rank != MPI_PROC_NULL) {
      if (scan_ltr_recv_ints[ASSIGN_VALID]) {
        /* accumulate totals at left end in left-to-right scan */
        if (scan_ltr_low_ints[ASSIGN_FLAG] != 1) {
          scan_ltr_low_ints[ASSIGN_FLAG] = scan_ltr_recv_ints[ASSIGN_FLAG];
          if (scan_ltr_low_ints[ASSIGN_VALID]) {
            /* accumulate received value into ours */
            MPI_Reduce_local(scan_ltr_recv_val, scan_ltr_low_val, 1, val, op);
          } else {
            /* if our low value is not valid, copy the one received */
            scan_ltr_low_ints[ASSIGN_VALID] = 1;
            DTCMP_Memcpy(scan_ltr_low_key, 1, key, scan_ltr_recv_key, 1, key);
            DTCMP_Memcpy(scan_ltr_low_val, 1, val, scan_ltr_recv_val, 1, val);
          }
        }

        /* accumulate total at right end in left-to-right scan */
        if (scan_ltr_high_ints[ASSIGN_FLAG] != 1) {
          scan_ltr_high_ints[ASSIGN_FLAG] = scan_ltr_recv_ints[ASSIGN_FLAG];
          if (scan_ltr_high_ints[ASSIGN_VALID]) {
            /* accumulate received value into ours */
            MPI_Reduce_local(scan_ltr_recv_val, scan_ltr_high_val, 1, val, op);
          } else {
            /* if our high value is not valid, copy the one received */
            scan_ltr_high_ints[ASSIGN_VALID] = 1;
            DTCMP_Memcpy(scan_ltr_high_key, 1, key, scan_ltr_recv_key, 1, key);
            DTCMP_Memcpy(scan_ltr_high_val, 1, val, scan_ltr_recv_val, 1, val);
          }
        }
      }

      /* get the next rank on the left */
      left_rank = scan_ltr_recv_ints[ASSIGN_NEXT];
    }

    /* reduce data from right partner */
    if (right_rank != MPI_PROC_NULL) {
      if (scan_rtl_recv_ints[ASSIGN_VALID]) {
        /* accumulate totals at right end in right-to-left scan */
        if (scan_rtl_low_ints[ASSIGN_FLAG] != 1) {
          scan_rtl_low_ints[ASSIGN_FLAG] = scan_rtl_recv_ints[ASSIGN_FLAG];
          if (scan_rtl_low_ints[ASSIGN_VALID]) {
            /* accumulate received value into ours */
            MPI_Reduce_local(scan_rtl_recv_val, scan_rtl_low_val, 1, val, op);
          } else {
            /* if our low value is not valid, copy the one received */
            scan_rtl_low_ints[ASSIGN_VALID] = 1;
            DTCMP_Memcpy(scan_rtl_low_key, 1, key, scan_rtl_recv_key, 1, key);
            DTCMP_Memcpy(scan_rtl_low_val, 1, val, scan_rtl_recv_val, 1, val);
          }
        }

        /* accumulate total at left end in right-to-left scan */
        if (scan_rtl_high_ints[ASSIGN_FLAG] != 1) {
          scan_rtl_high_ints[ASSIGN_FLAG] = scan_rtl_recv_ints[ASSIGN_FLAG];
          if (scan_rtl_high_ints[ASSIGN_VALID]) {
            /* accumulate received value into ours */
            MPI_Reduce_local(scan_rtl_recv_val, scan_rtl_high_val, 1, val, op);
          } else {
            /* if our high value is not valid, copy the one received */
            scan_rtl_high_ints[ASSIGN_VALID] = 1;
            DTCMP_Memcpy(scan_rtl_high_key, 1, key, scan_rtl_recv_key, 1, key);
            DTCMP_Memcpy(scan_rtl_high_val, 1, val, scan_rtl_recv_val, 1, val);
          }
        }
      }

      /* get the next rank on the left */
      right_rank = scan_rtl_recv_ints[ASSIGN_NEXT];
    }
  }

  /* if low scan value is valid, initialize our high value */
  if (scan_ltr_low_ints[ASSIGN_VALID]) {
    DTCMP_Memcpy(scan_ltr_high_val, 1, val, scan_ltr_low_val, 1, val);
  }
  if (scan_rtl_low_ints[ASSIGN_VALID]) {
    DTCMP_Memcpy(scan_rtl_high_val, 1, val, scan_rtl_low_val, 1, val);
  }

  /* finally set user's output buffers */
  for (i = 0; i < count; i++) {
    /* get pointer to user's value */
    char* inval = (char*)valbuf + i * val_extent;
    if (first_in_group[i]) {
      /* reinit our accumulator with user's data if we start a new group */
      DTCMP_Memcpy(scan_ltr_high_val, 1, val, inval, 1, val);
    } else {
      /* otherwise, copy scan result to user's buffer */
      char* ltrval = (char*)ltrbuf + i * val_extent;
      DTCMP_Memcpy(ltrval, 1, val, scan_ltr_high_val, 1, val);
      
      /* add users data to accumulation,
       * be careful with the order here */
      DTCMP_Memcpy(scan_tmp_val, 1, val, inval, 1, val);
      MPI_Reduce_local(scan_ltr_high_val, scan_tmp_val, 1, val, op);
      DTCMP_Memcpy(scan_ltr_high_val, 1, val, scan_tmp_val, 1, val);
    }

    /* get pointer to user's value */
    int j = count - 1 - i;
    char* endval = (char*)valbuf + j * val_extent;
    if (last_in_group[j]) {
      /* reinit our accumulator with user's data if we start a new group */
      DTCMP_Memcpy(scan_rtl_high_val, 1, val, endval, 1, val);
    } else {
      /* otherwise, copy scan result to user's buffer */
      char* rtlval = (char*)rtlbuf + j * val_extent;
      DTCMP_Memcpy(rtlval, 1, val, scan_rtl_high_val, 1, val);
      
      /* add users data to accumulation,
       * be careful with the order here */
      DTCMP_Memcpy(scan_tmp_val, 1, val, endval, 1, val);
      MPI_Reduce_local(scan_rtl_high_val, scan_tmp_val, 1, val, op);
      DTCMP_Memcpy(scan_rtl_high_val, 1, val, scan_tmp_val, 1, val);
    }
  }

  /* TODO: complete right-to-left scan */

  /* free buffers */
  dtcmp_free(&scan_tmp_val);

  dtcmp_free(&scan_ltr_high);
  dtcmp_free(&scan_ltr_low);
  dtcmp_free(&scan_ltr_recv);
  dtcmp_free(&scan_rtl_high);
  dtcmp_free(&scan_rtl_low);
  dtcmp_free(&scan_rtl_recv);

  MPI_Type_free(&type_item);

  dtcmp_free(&first_in_group);
  dtcmp_free(&last_in_group);

  return rc;
}
