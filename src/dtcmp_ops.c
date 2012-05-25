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
#include "dtcmp_ops.h"

/* initialize operation handle with explicit byte displacement */
void dtcmp_op_hinit(
  uint32_t type,
  MPI_Datatype key,
  DTCMP_Op_fn fn,
  MPI_Aint disp,
  DTCMP_Op series,
  DTCMP_Op* cmp)
{
  DTCMP_Handle_t* c = (DTCMP_Handle_t*) malloc(sizeof(DTCMP_Handle_t));
  if (c != NULL) {
    c->magic  = 1;
    c->type   = type;
    MPI_Type_dup(key, &(c->key));
    c->fn     = fn;
    c->disp   = disp;
    c->series = series;
  }
  *cmp = (DTCMP_Op) c;
}

/* initialize operation handle with extent(key) */
void dtcmp_op_init(
  uint32_t type,
  MPI_Datatype key,
  DTCMP_Op_fn fn,
  DTCMP_Op series,
  DTCMP_Op* cmp)
{
  /* get extent of key type */
  MPI_Aint lb, extent;
  MPI_Type_get_extent(key, &lb, &extent);

  /* use extent of key type as displacement to second item */
  dtcmp_op_hinit(type, key, fn, extent, series, cmp);
}

/* make a copy of src operation and save in dst */
void dtcmp_op_copy(DTCMP_Op* dst, DTCMP_Op src)
{
  DTCMP_Handle_t* s = (DTCMP_Handle_t*) src;
  if (s->series == DTCMP_OP_NULL) {
    dtcmp_op_hinit(s->type, s->key, s->fn, s->disp, s->series, dst);
  } else {
    DTCMP_Op copy;
    dtcmp_op_copy(&copy, s->series);
    dtcmp_op_hinit(s->type, s->key, s->fn, s->disp, copy, dst);
  }
}

int dtcmp_op_eval(const void* a, const void* b, DTCMP_Op cmp)
{
  /* get pointer to handle and comparison operation */
  DTCMP_Handle_t* c = (DTCMP_Handle_t*) cmp;
  DTCMP_Op_fn compare = c->fn;

  /* invoke comparison function to compare a and b */
  int rc = (*compare)(a, b);
  if (rc != 0) {
    /* comparison found the two are not equal, just return our result */
    return rc;
  } else {
    /* a and b are equal, see if there is a series comparison */
    if (c->series != DTCMP_OP_NULL) {
      /* advance pointers to point past end of exurrent key */
      const void* a_new = (char*)a + c->disp;
      const void* b_new = (char*)b + c->disp;

      /* invoke series comparison function */
      return dtcmp_op_eval(a_new, b_new, c->series);
    } else {
      /* a and b are equal and there is no series comparison, just return our result (equal) */
      return rc;
    }
  }
}

int dtcmp_op_fn_int_ascend(const void* bufa, const void* bufb)
{
  int a = *(int*)bufa;
  int b = *(int*)bufb;
  if (a < b) {
    return -1;
  } else if (b < a) {
    return  1;
  } else {
    return 0;
  }
}

int dtcmp_op_fn_int_descend(const void* bufa, const void* bufb)
{
  int a = *(int*)bufa;
  int b = *(int*)bufb;
  if (a > b) {
    return -1;
  } else if (b > a) {
    return  1;
  } else {
    return 0;
  }
}
