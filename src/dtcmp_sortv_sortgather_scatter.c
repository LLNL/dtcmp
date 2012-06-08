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

#define MIN (0)
#define MAX (1)
#define SUM (2)
#define LEV (3)

/* issue a reduce to root of tree to determine total number of elements
 * each rank will receive from its children to compute total it will send
 * to its parent, then issue a sorting gather to root of tree where at
 * each step parent merges sorted list it currently has with the incoming
 * sorted lists from each of its children, so that entire list is received
 * and sorted as the last step at the root, then scatter the sorted
 * elements back to children passing down same number they passed up */

int DTCMP_Sortv_sortgather_scatter(
  const void* inbuf,
  void* outbuf,
  int count,
  MPI_Datatype key,
  MPI_Datatype keysat,
  DTCMP_Op cmp,
  MPI_Comm comm)
{
  int iter, dist;

  /* determine our rank and the number of ranks in our group */
  int rank, ranks;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &ranks);

  /* don't bother with all of this mess if we only have a single rank,
   * we use this check because later we'll allocate memory of log(P)
   * and if P=1, then log(P)=0 and we don't want to malloc(0) */
  if (ranks < 2) {
    return DTCMP_Sort_local(inbuf, outbuf, count, key, keysat, cmp);
  }

  /* get extent of keysat type */
  MPI_Aint true_lb, true_extent;
  MPI_Type_get_true_extent(keysat, &true_lb, &true_extent);

  /* given extent of datatype and maximum amount of memory,
   * compute maximum number of elements we can hold */
  uint64_t max_mem = 100*1024*1024;
//  int threshold = max_mem / true_extent;
  uint64_t threshold = 3;

  /* determine the potential number of tasks we will communicate
   * with on our left and right sides */
  int list_size = 0;
  dist = 1;
  while (dist < ranks) {
    list_size++;
    dist <<= 1;
  }

  /* declare pointers for list of left and right ranks, as well as,
   * number of elements we'll receive from each child */
  int* left_list       = dtcmp_malloc(list_size * sizeof(int), 0, __FILE__, __LINE__);
  int* right_list      = dtcmp_malloc(list_size * sizeof(int), 0, __FILE__, __LINE__);
  uint64_t* count_list = dtcmp_malloc(list_size * sizeof(uint64_t), 0, __FILE__, __LINE__);

  /* now identify our children and parent in the tree,
   * and compute number of elements we'll receive from each child */
  int left_rank;
  int right_rank;
  int left_size  = 0;
  int right_size = 0;
  int count_size = 0;

  /* initialize our data for reduction */
  uint64_t reduce[4];
  reduce[MIN] = count; /* minimum count across all ranks */
  reduce[MAX] = count; /* maximum count across all ranks */
  reduce[SUM] = count; /* sum of counts across all ranks */
  reduce[LEV] = 0;     /* min iteration at which some rank has max # elems */

  /* send counts up tree to determine how many elements each
   * child will contribute, remember the iteration in which we send */
  int send_iteration = -1;
  iter = 0;
  dist = 1;
  while (dist < ranks) {
    /* get the rank dist hops to our left */
    left_rank = rank - dist;
    if (left_rank >= 0) {
      left_list[left_size] = left_rank;
      left_size++;
    }

    /* get the rank dist hops to our right */
    right_rank = rank + dist;
    if (right_rank < ranks) {
      right_list[right_size] = right_rank;
      right_size++;
    }

    /* if we have not already sent our data for the gather operation,
     * check whether we need to send or receive data in this iteration */
    if (send_iteration == -1) {
      /* determine whether we should send or receive in this round */
      int send = rank & dist;
      if (send) {
        /* send reduction data to parent and
         * remember the iteration in which we send */
        MPI_Send(reduce, 4, MPI_UINT64_T, left_rank, 0, comm);
        send_iteration = iter;
      } else {
        /* compute the rank we'll receive from in this round */
        int recv_rank = rank + dist;
        if (recv_rank < ranks) {
          /* receive reduction data from this child */
          uint64_t recv_reduce[4];
          MPI_Status recv_status;
          MPI_Recv(recv_reduce, 4, MPI_UINT64_T, right_rank, 0, comm, &recv_status);

          /* record the total number of elements from each child */
          count_list[count_size] = recv_reduce[SUM];
          count_size++;

          /* compute minimum and maximum counts across all ranks */
          if (recv_reduce[MIN] < reduce[MIN]) {
            reduce[MIN] = recv_reduce[MIN];
          }
          if (recv_reduce[MAX] > reduce[MAX]) {
            reduce[MAX] = recv_reduce[MAX];
          }

          /* compute running sum of all elements from our children */
          reduce[SUM] += recv_reduce[SUM];

          /* check whether any task has reached its threshold, and if
           * so record the lowest iteration in which happens */
          if (reduce[LEV] == 0) {
            if (recv_reduce[LEV] != 0) {
              /* if we haven't reached the threshold but our child has,
               * then set our level value to our child's */
              reduce[LEV] = recv_reduce[LEV];
            } else if (reduce[SUM] > threshold) {
              /* otherwise, if we have reached the threshold,
               * record the current iteration plus one, since 0 means NULL */ 
              reduce[LEV] = iter + 1;
            }
          }
        }
      }
    }

    /* go on to next step */
    dist <<= 1;
    iter++;
  }

  /* broacast totals back down the tree */
  int received = 0;
  if (rank == 0) {
    received = 1;

    /* if we never reached the threshold, set the reduce level to signal
     * that the gather will go all the way up the tree */
    if (reduce[LEV] == 0) {
      reduce[LEV] = iter + 1;
    }
  }
  while (dist > 1) {
    iter--;
    dist >>= 1;

    /* determine whether we should send or receive in this round */
    if (!received) {
      int receive = (iter == send_iteration);
      if (receive) {
        /* get global rank we'll be receiving from in this step */
        left_rank = left_list[iter];

        /* receive totals */
        MPI_Status recv_status;
        MPI_Recv(reduce, 4, MPI_UINT64_T, left_rank, 0, comm, &recv_status);
        received = 1;
      }
    } else {
      /* we've received our data, now we forward it during each step */
      int send_rank = rank + dist;
      if (send_rank < ranks) {
        /* get global rank we'll be sending to in this step */
        right_rank = right_list[iter];
 
        /* send data and update our offset for the next send */
        MPI_Send(reduce, 4, MPI_UINT64_T, right_rank, 0, comm);
      }
    }
  }

  /* given that we'll gather items up tree for reduce[LEV] steps,
   * total up number of elements we'll receive from our children */
  iter = 0;
  uint64_t elem_count = count;
  int block_size = 1;
  while (iter < reduce[LEV]) {
    if (iter < send_iteration || send_iteration == -1) {
      elem_count += count_list[iter];
    }
    block_size <<= 1;
    iter++;
  }

  /* declare pointers for temporary buffers to receive elements */
  void* merge_buf = dtcmp_malloc(elem_count * true_extent, 0, __FILE__, __LINE__);
  void* send_buf  = dtcmp_malloc(elem_count * true_extent, 0, __FILE__, __LINE__);
  void* recv_buf  = dtcmp_malloc(elem_count * true_extent, 0, __FILE__, __LINE__);

  /* copy our input data to send buffer buffer and sort it */
  char* buf = (void*) inbuf;
  if (inbuf == DTCMP_IN_PLACE) {
    buf = outbuf;
  }
  DTCMP_Memcpy(send_buf, count, keysat, (void*)buf, count, keysat);
  DTCMP_Sort_local(DTCMP_IN_PLACE, send_buf, count, key, keysat, cmp);

  /* gather data to sorters (and sort the data as it is gathered) */
  dist = 1;
  iter = 0;
  int sent = 0;
  int send_count = count;
  while (!sent && iter < reduce[LEV]) {
    /* determine whether we should send or receive in this round */
    int send = (iter == send_iteration);
    if (send) {
      left_rank = left_list[iter];
      if (send_count > 0) {
        MPI_Send(send_buf, send_count, keysat, left_rank, 0, comm);
      }
      sent = 1;
    } else {
      /* compute the rank we'll receive from in this round */
      int recv_rank = rank + dist;
      if (recv_rank < ranks) {
        /* determine the number of entries we'll receive from this rank */
        right_rank = right_list[iter];
        int recv_count = (int)count_list[iter];
        if (recv_count > 0) {
          /* receive the data */
          MPI_Status recv_status;
          MPI_Recv(
            recv_buf, recv_count, keysat,
            right_rank, 0, comm, &recv_status
          );

          /* merge our send and recv buffers into merge buffer */
          const void* inbufs[2];
          inbufs[0] = send_buf;
          inbufs[1] = recv_buf;
          int counts[2];
          counts[0] = send_count;
          counts[1] = recv_count;
          DTCMP_Merge_local(2, inbufs, counts, merge_buf, key, keysat, cmp);

          /* swap our send buffer with our merge buffer */
          void* tmp_buf = send_buf;
          send_buf = merge_buf;
          merge_buf = tmp_buf;

          /* add the number of received elements to our send count */
          send_count += recv_count;
        }
      }
    }

    /* go on to next step */
    dist <<= 1;
    iter++;
  }

  /* determine whether we are one of the ranks to do the sorting */
  int sorter = 0;
  int sort_rank = rank / block_size;
  if (sort_rank * block_size == rank) {
    sorter = 1;
  }

  /* compute the number of sorter ranks */
  int sort_ranks = ranks / block_size;
  if (sort_ranks * block_size < ranks) {
    sort_ranks++;
  }

  if (sort_ranks > 1) {
    if (sorter) {
#if 0
      /* build a group of sorters */
      groupinfo sort;
      sort.comm       = comm;
      sort.comm_rank  = comm_rank;
      sort.comm_left  = left_rank;
      sort.comm_right = right_rank;
      sort.group_rank = sort_rank;
      sort.group_size = sort_ranks;
#endif
      int* sort_group = (int*) malloc(sort_ranks * sizeof(int));
      int i;
      for (i = 0; i < sort_ranks; i++) {
        sort_group[i] = i * block_size;
      }

      /* parallel sort */
      DTCMP_Sortv_cheng(
        send_buf, recv_buf, (int)elem_count, key, keysat, cmp,
        sort_rank, sort_ranks, sort_group, comm
      );

      free(sort_group);

      /* swap our buffers so that the sorted data is in send_buf */
      void* tmp = send_buf;
      send_buf = recv_buf;
      recv_buf = tmp;
    }
  }

  /* rename variables pointing to our data to avoid confusion below */
  void* final_buf = send_buf;  /* this holds our sorted data */
  int final_count = 0; /* running total of number of elements we have already sent */

  /* scatter sorted data back to ranks */
  received = 0;
  if (sorter) {
    received = 1;
  }
  while (dist > 1) {
    iter--;
    dist >>= 1;

    /* TODO: avoid send/recv if counts are 0 */
    /* determine whether we should send or receive in this round */
    if (!received) {
      int receive = (iter == send_iteration);
      if (receive) {
        /* get global rank we'll be receiving from in this step */
        left_rank = left_list[iter];

        /* determine the number of entries we'll receive from this rank */
        int recv_count = (int)elem_count;

        /* receive data (don't bother with the actual recv unless we really have data) */
        if (recv_count > 0) {
          MPI_Status recv_status;
          MPI_Recv(
            final_buf, recv_count, keysat,
            left_rank, 0, comm, &recv_status
          );
        }

        received = 1;
      }
    } else {
      /* we've received our data, now we forward it during each step */
      int send_rank = rank + dist;
      if (send_rank < ranks) {
        /* get global rank we'll be sending to in this step */
        right_rank = right_list[iter];
 
        /* determine the number of elements to send to this rank */
        int send_count = (int)count_list[iter];

        /* determine our offset in the send buffer */
        final_count += send_count;
        int final_offset = ((int)elem_count - final_count) * true_extent;

        /* send data and update our offset for the next send */
        if (send_count > 0) {
          MPI_Send(
            (char*)final_buf + final_offset, send_count, keysat,
            right_rank, 0, comm
          );
        }
      }
    }
  }

  /* copy result into outbuf */
  DTCMP_Memcpy(outbuf, count, keysat, final_buf, count, keysat);

  /* free our scratch space */
  dtcmp_free(&recv_buf);
  dtcmp_free(&send_buf);
  dtcmp_free(&merge_buf);

  /* free our scratch space */
  dtcmp_free(&count_list);
  dtcmp_free(&right_list);
  dtcmp_free(&left_list);

  /* after the initial allreduce up and down the tree, processes that
   * contribute 0 items may skip all remaining sends/recvs so place
   * a barrier here to prevent them from escaping ahead */
  MPI_Barrier(comm);

  return 0;
}
