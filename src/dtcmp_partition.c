/* Copyright (c) 2012, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by Adam Moody <moody20@llnl.gov> and Edgar A. Leon <leon@llnl.gov>.
 * LLNL-CODE-557516.
 * All rights reserved.
 * This file is part of the DTCMP library.
 * For details, see https://github.com/hpc/dtcmp
 * Please also read this file: LICENSE.TXT. */

#include <stdlib.h>
#include <string.h>
#include "mpi.h"
#include "dtcmp_internal.h"

int dtcmp_partition_combined_memcpy(
  void* buf,
  void* scratch,
  int pivot,
  int num,
  size_t size,
  DTCMP_Op cmp)
{
  /* swap pivot element with last element */
  char* pivotbuf = (char*)buf + pivot   * size;
  char* lastbuf  = (char*)buf + (num-1) * size;
  memcpy(scratch,  pivotbuf, size);
  memcpy(pivotbuf, lastbuf,  size);
  memcpy(lastbuf,  scratch,  size);
  
  /* move small elements to left side and large elements to right side of array,
   * split equal elements evenly on either side of pivot */
  int jumpball_arrow = 0; /* basketball reference */
  int divide = 0;
  char* dividebuf = buf;  /* tracks element dividing small and large sides */
  char* itembuf   = buf;
  while (itembuf != lastbuf) {
    /* compare current item to pivot */
    int result = dtcmp_op_eval(itembuf, lastbuf, cmp);
    if (result < 0) {
      /* current element is smaller than pivot,
       * swap this element with the one at the divide and advance divide pointer */
      memcpy(scratch,   itembuf,   size);
      memcpy(itembuf,   dividebuf, size);
      memcpy(dividebuf, scratch,   size);
      dividebuf += size;
      divide++;
    } else if (result == 0) {
      /* we put every other equal element on left side */
      if (jumpball_arrow) {
        /* swap this element with the one at the divide and advance divide pointer */
        memcpy(scratch,   itembuf,   size);
        memcpy(itembuf,   dividebuf, size);
        memcpy(dividebuf, scratch,   size);
        dividebuf += size;
        divide++;
      }
      /* flip the jumpball arrow for the next tie */
      jumpball_arrow ^= 0x1;
    }

    /* advance pointer to next item */
    itembuf += size;
  }

  /* copy pivot to its proper place */
  memcpy(scratch,   lastbuf,   size);
  memcpy(lastbuf,   dividebuf, size);
  memcpy(dividebuf, scratch,   size);

  /* return position of pivot */
  return divide;
}

#if 0
static int dtcmp_sort_local_combined_randquicksort_partition_scratch(
  void* buf,
  void* scratch,
  int pivot,
  int num,
  MPI_Aint extent,
  MPI_Datatype keysat,
  DTCMP_Op cmp)
{
  /* swap pivot element with last element */
  char* pivotbuf = (char*)buf + pivot   * extent;
  char* lastbuf  = (char*)buf + (num-1) * extent;
  DTCMP_Memcpy(scratch, 1, keysat, pivotbuf, 1, keysat);
  DTCMP_Memcpy(pivotbuf, 1, keysat, lastbuf, 1, keysat);
  DTCMP_Memcpy(lastbuf, 1, keysat, pivotbuf, 1, keysat);
  
  /* TODO: spread equal items in a balanced way on either side of pivot */
  int i;
  int current = 0;
  for (i = 0; i < num-1; i++) {
    char* itembuf = (char*)buf + i * extent;
    int result = dtcmp_op_eval(itembuf, lastbuf, cmp);
    if (result < 0) {
      char* currentbuf = (char*)buf + current * extent;
      DTCMP_Memcpy(scratch, 1, keysat, itembuf, 1, keysat);
      DTCMP_Memcpy(itembuf, 1, keysat, currentbuf, 1, keysat);
      DTCMP_Memcpy(currentbuf, 1, keysat, scratch, 1, keysat);
      current++;
    }
  }

  /* return position of pivot */
  return current;
}
#endif
