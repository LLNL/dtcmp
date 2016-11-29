/* Copyright (c) 2012, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by Adam Moody <moody20@llnl.gov>.
 * LLNL-CODE-557516.
 * All rights reserved.
 * This file is part of the DTCMP library.
 * For details, see https://github.com/hpc/dtcmp
 * Please also read this file: LICENSE.TXT. */

// mpicc -g -O0 -o test_segscan test_segscan.c -I./install/include -Wl,-rpath,`pwd`/install/lib -L./install/lib -ldtcmp
// srun -n4 ./test_segscan
//
// This runs a few tests with Segmented scan/exscan operations and
// verifies the results.  The program prints a message to stdout,
// and returns with an exit code of 1 if an error is detected,
// it returns 0 otherwise.

#include <stdlib.h>
#include <stdio.h>
#include "mpi.h"
#include "dtcmp.h"

int main(int argc, char* argv[])
{
  int rc = 0;

  MPI_Init(&argc, &argv);
  DTCMP_Init();

  int rank, ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);

  int count = 10;
  int* keys = (int*) malloc(count * sizeof(int));
  int* vals = (int*) malloc(count * sizeof(int));
  int* ltr  = (int*) malloc(count * sizeof(int));
  int* rtl  = (int*) malloc(count * sizeof(int));

  int segment_length = 7;
  int i;
  for (i = 0; i < count; i++) {
    keys[i] = (rank*count + i) / segment_length;
    vals[i] = 1;
    ltr[i] = -1;
    rtl[i] = -1;
  }

  DTCMP_Segmented_exscan(count, keys, MPI_INT, vals, ltr, rtl, MPI_INT, DTCMP_OP_INT_ASCEND, DTCMP_FLAG_NONE, MPI_SUM, MPI_COMM_WORLD);

#if 0
  printf("exscan ltr: %d ", rank);
  for (i = 0; i < count; i++) {
    printf("(%d,%d)=%d, ", keys[i], vals[i], ltr[i]);
  }
  printf("\n");

  printf("exscan rtl: %d ", rank);
  for (i = 0; i < count; i++) {
    printf("(%d, %d)=%d, ", keys[i], vals[i], rtl[i]);
  }
  printf("\n");
#endif

  // check ltr results
  for (i = 0; i < count; i++) {
    int val = (rank*count + i) % segment_length;
    if (val == 0 && ltr[i] != -1) {
      rc = 1;
      printf("ERROR: Segmented_exscan rank=%d ltr[%d]=%d, expected -1\n", rank, i, ltr[i]);
    } else if (val > 0 && ltr[i] != val) {
      rc = 1;
      printf("ERROR: Segmented_exscan rank=%d ltr[%d]=%d, expected -1\n", rank, i, ltr[i], val);
    }
  }

  // check rtl results
  int last_length = (ranks * count) % segment_length;
  int length = segment_length;
  for (i = 0; i < count; i++) {
    // if we're in the last segment, it may not be full length
    int offset = rank * count + i;
    int remainder = ranks * count - offset;
    if (remainder <= last_length) {
      length = last_length;
    }
    int val = offset % segment_length;
    int actual = rtl[i];
    if (val == (length - 1) && actual != -1) {
      rc = 1;
      printf("ERROR: Segmented_exscan rank=%d rtl[%d]=%d, expected -1\n", rank, i, actual);
    } else if (val < (length - 1) && actual != (length - val - 1)) {
      rc = 1;
      printf("ERROR: Segmented_exscan rank=%d rtl[%d]=%d, expected %d\n", rank, i, actual, length - val - 1);
    }
  }

  for (i = 0; i < count; i++) {
    keys[i] = (rank*count + i) / 7;
    vals[i] = 1;
    ltr[i] = -1;
    rtl[i] = -1;
  }

  DTCMP_Segmented_scan(count, keys, MPI_INT, vals, ltr, rtl, MPI_INT, DTCMP_OP_INT_ASCEND, DTCMP_FLAG_NONE, MPI_SUM, MPI_COMM_WORLD);

#if 0
  printf("scan ltr: %d ", rank);
  for (i = 0; i < count; i++) {
    printf("(%d,%d)=%d, ", keys[i], vals[i], ltr[i]);
  }
  printf("\n");

  printf("scan rtl: %d ", rank);
  for (i = 0; i < count; i++) {
    printf("(%d, %d)=%d, ", keys[i], vals[i], rtl[i]);
  }
  printf("\n");
#endif

  // check ltr results
  for (i = 0; i < count; i++) {
    int val = (rank*count + i) % segment_length;
    if (ltr[i] != val + 1) {
      rc = 1;
      printf("ERROR: Segmented_scan rank=%d ltr[%d]=%d, expected -1\n", rank, i, ltr[i], val + 1);
    }
  }

  // check rtl results
  last_length = (ranks * count) % segment_length;
  length = segment_length;
  for (i = 0; i < count; i++) {
    // if we're in the last segment, it may not be full length
    int offset = rank * count + i;
    int remainder = ranks * count - offset;
    if (remainder <= last_length) {
      length = last_length;
    }
    int val = offset % segment_length;
    int actual = rtl[i];
    int expected = length - val;
    if (actual != expected) {
      rc = 1;
      printf("ERROR: Segmented_scan rank=%d rtl[%d]=%d, expected %d\n", rank, i, actual, expected);
    }
  }

  for (i = 0; i < count; i++) {
    keys[i] = (rank*count + i) / 7;
    vals[i] = 1;
    ltr[i] = -1;
    rtl[i] = -1;
  }

  DTCMP_Segmented_exscan_ltr(count, keys, MPI_INT, vals, ltr, MPI_INT, DTCMP_OP_INT_ASCEND, DTCMP_FLAG_NONE, MPI_SUM, MPI_COMM_WORLD);

#if 0
  printf("exscan_ltr: %d ", rank);
  for (i = 0; i < count; i++) {
    printf("(%d,%d)=%d, ", keys[i], vals[i], ltr[i]);
  }
  printf("\n");
#endif

  DTCMP_Segmented_scan_ltr(count, keys, MPI_INT, vals, ltr, MPI_INT, DTCMP_OP_INT_ASCEND, DTCMP_FLAG_NONE, MPI_SUM, MPI_COMM_WORLD);

#if 0
  printf("scan_ltr: %d ", rank);
  for (i = 0; i < count; i++) {
    printf("(%d,%d)=%d, ", keys[i], vals[i], ltr[i]);
  }
  printf("\n");
#endif

  free(rtl);
  free(ltr);
  free(vals);
  free(keys);

  DTCMP_Finalize();
  MPI_Finalize();

  return rc;
}
