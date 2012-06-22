/* Copyright (c) 2012, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by Adam Moody <moody20@llnl.gov>.
 * LLNL-CODE-557516.
 * All rights reserved.
 * This file is part of the DTCMP library.
 * For details, see https://github.com/hpc/dtcmp
 * Please also read this file: LICENSE.TXT. */

#include <stdlib.h>
#include "mpi.h"
#include "dtcmp_internal.h"
#include "dtcmp_ops.h"

/* This file defines the constants exposed by the library,
 * and it also serves as the dispatch function which does
 * error checking on input parameters and then calls
 * a specific underlying implementation. */

/* reference count for number of DTCMP_Init calls,
 * allows each library to call DTCMP_Init without
 * worrying about whether it's already been called. */
static int dtcmp_init_count = 0;

/* set up our DTCMP_IN_PLACE constant
 * (just a void* pointer to an int in static memory) */
static int DTCMP_IN_PLACE_LOCATION;
const void* DTCMP_IN_PLACE = (const void*) &DTCMP_IN_PLACE_LOCATION;

/* we'll dup comm_self during init, which we need for our memcpy */
MPI_Comm dtcmp_comm_self = MPI_COMM_NULL;

/* we create a type of 3 consecutive uint64_t for computing min/max/sum reduction */
MPI_Datatype dtcmp_type_3uint64t = MPI_DATATYPE_NULL;

/* op for computing min/max/sum reduction */
MPI_Op dtcmp_reduceop_mms_3uint64t = MPI_OP_NULL;

/* we call rand_r() to acquire random numbers,
 * and this keeps track of the seed between calls */
unsigned dtcmp_rand_seed = 0;

/* set a NULL handle to be a NULL pointer */
DTCMP_Handle DTCMP_HANDLE_NULL = NULL;

/* define our NULL comparison handle */
DTCMP_Op DTCMP_OP_NULL = NULL;

/* set predefined comparison ops to NULL now,
 * we'll fill these in during init */
DTCMP_Op DTCMP_OP_SHORT_ASCEND  = NULL;
DTCMP_Op DTCMP_OP_SHORT_DESCEND = NULL;
DTCMP_Op DTCMP_OP_INT_ASCEND  = NULL;
DTCMP_Op DTCMP_OP_INT_DESCEND = NULL;
DTCMP_Op DTCMP_OP_LONG_ASCEND  = NULL;
DTCMP_Op DTCMP_OP_LONG_DESCEND = NULL;
DTCMP_Op DTCMP_OP_LONGLONG_ASCEND  = NULL;
DTCMP_Op DTCMP_OP_LONGLONG_DESCEND = NULL;
DTCMP_Op DTCMP_OP_UNSIGNEDSHORT_ASCEND  = NULL;
DTCMP_Op DTCMP_OP_UNSIGNEDSHORT_DESCEND = NULL;
DTCMP_Op DTCMP_OP_UNSIGNEDINT_ASCEND  = NULL;
DTCMP_Op DTCMP_OP_UNSIGNEDINT_DESCEND = NULL;
DTCMP_Op DTCMP_OP_UNSIGNEDLONG_ASCEND  = NULL;
DTCMP_Op DTCMP_OP_UNSIGNEDLONG_DESCEND = NULL;
DTCMP_Op DTCMP_OP_UNSIGNEDLONGLONG_ASCEND  = NULL;
DTCMP_Op DTCMP_OP_UNSIGNEDLONGLONG_DESCEND = NULL;
DTCMP_Op DTCMP_OP_INT8T_ASCEND  = NULL;
DTCMP_Op DTCMP_OP_INT8T_DESCEND = NULL;
DTCMP_Op DTCMP_OP_INT16T_ASCEND  = NULL;
DTCMP_Op DTCMP_OP_INT16T_DESCEND = NULL;
DTCMP_Op DTCMP_OP_INT32T_ASCEND  = NULL;
DTCMP_Op DTCMP_OP_INT32T_DESCEND = NULL;
DTCMP_Op DTCMP_OP_INT64T_ASCEND  = NULL;
DTCMP_Op DTCMP_OP_INT64T_DESCEND = NULL;
DTCMP_Op DTCMP_OP_UINT8T_ASCEND  = NULL;
DTCMP_Op DTCMP_OP_UINT8T_DESCEND = NULL;
DTCMP_Op DTCMP_OP_UINT16T_ASCEND  = NULL;
DTCMP_Op DTCMP_OP_UINT16T_DESCEND = NULL;
DTCMP_Op DTCMP_OP_UINT32T_ASCEND  = NULL;
DTCMP_Op DTCMP_OP_UINT32T_DESCEND = NULL;
DTCMP_Op DTCMP_OP_UINT64T_ASCEND  = NULL;
DTCMP_Op DTCMP_OP_UINT64T_DESCEND = NULL;
DTCMP_Op DTCMP_OP_FLOAT_ASCEND  = NULL;
DTCMP_Op DTCMP_OP_FLOAT_DESCEND = NULL;
DTCMP_Op DTCMP_OP_DOUBLE_ASCEND  = NULL;
DTCMP_Op DTCMP_OP_DOUBLE_DESCEND = NULL;
DTCMP_Op DTCMP_OP_LONGDOUBLE_ASCEND  = NULL;
DTCMP_Op DTCMP_OP_LONGDOUBLE_DESCEND = NULL;

/* determine whether type is contiguous, has a true lower bound of 0,
 * and extent == true_extent */
static int dtcmp_type_is_valid(MPI_Datatype type)
{
  /* get (user-defined) lower bound and extent */
  MPI_Aint lb, extent;
  MPI_Type_get_extent(type, &lb, &extent);

  /* get true lower bound and extent */
  MPI_Aint true_lb, true_extent;
  MPI_Type_get_true_extent(type, &true_lb, &true_extent);

  /* get size of type */
  int size;
  MPI_Type_size(type, &size);

  /* check that type is contiguous (size == true_extent ==> no holes) */
  if (size != true_extent) {
    return 0;
  }

  /* check that lower bounds are 0 */
  if (lb != 0 || true_lb != 0) {
    return 0;
  }

  /* check that extent == true_extent ==> no funny business if we
   * concatenate a series of these types */
  if (extent != true_extent) {
    return 0;
  }

  /* check that extent is positive */
  if (extent <= 0) {
    return 0;
  }

  return 1;
}

static int dtcmp_op_is_valid(DTCMP_Op cmp)
{
  if (cmp == DTCMP_OP_NULL) {
    return 0;
  }

  dtcmp_op_handle_t* c = (dtcmp_op_handle_t*) cmp;
  if (c->magic != 1) {
    return 0;
  }

  return 1;
}

/* user-defined reduction operation to compute min/max/sum */
static void dtcmp_reducefn_uint64t_min_max_sum(void* invec, void* inoutvec, int* len, MPI_Datatype* type)
{
   uint64_t* a = (uint64_t*) invec;
   uint64_t* b = (uint64_t*) inoutvec;

   int i;
   for (i = 0; i < *len; i++) {
     /* compute minimum across all ranks */
     if (a[MMS_MIN] < b[MMS_MIN]) {
       b[MMS_MIN] = a[MMS_MIN];
     }

     /* compute maximum across all ranks */
     if (a[MMS_MAX] > b[MMS_MAX]) {
       b[MMS_MAX] = a[MMS_MAX];
     }

     /* compute sum across all ranks */
     b[MMS_SUM] += a[MMS_SUM];

     /* advance to next element */
     a += 3;
     b += 3;
  }
}

/* initialize the sorting library */
int DTCMP_Init()
{
  /* initialize our values if init has not already been called */
  if (dtcmp_init_count == 0) {
    /* copy comm_self */
    MPI_Comm_dup(MPI_COMM_SELF, &dtcmp_comm_self);

    /* set up a datatype for min/max/sum reduction */
    MPI_Type_contiguous(3, MPI_UINT64_T, &dtcmp_type_3uint64t);
    MPI_Type_commit(&dtcmp_type_3uint64t);

    /* set up our user-defined op for min/max/sum reduction,
     * just integer min/max/addition of non-negative values,
     * so assume this is commutative */
    int commutative = 1;
    MPI_Op_create(dtcmp_reducefn_uint64t_min_max_sum, commutative, &dtcmp_reduceop_mms_3uint64t);

    /* setup predefined cmp handles */
    DTCMP_Op_create(MPI_SHORT, dtcmp_op_fn_short_ascend,  &DTCMP_OP_SHORT_ASCEND);
    DTCMP_Op_create(MPI_SHORT, dtcmp_op_fn_short_descend, &DTCMP_OP_SHORT_DESCEND);
    DTCMP_Op_create(MPI_INT, dtcmp_op_fn_int_ascend,  &DTCMP_OP_INT_ASCEND);
    DTCMP_Op_create(MPI_INT, dtcmp_op_fn_int_descend, &DTCMP_OP_INT_DESCEND);
    DTCMP_Op_create(MPI_LONG, dtcmp_op_fn_long_ascend,  &DTCMP_OP_LONG_ASCEND);
    DTCMP_Op_create(MPI_LONG, dtcmp_op_fn_long_descend, &DTCMP_OP_LONG_DESCEND);
    DTCMP_Op_create(MPI_LONG_LONG, dtcmp_op_fn_longlong_ascend,  &DTCMP_OP_LONGLONG_ASCEND);
    DTCMP_Op_create(MPI_LONG_LONG, dtcmp_op_fn_longlong_descend, &DTCMP_OP_LONGLONG_DESCEND);
    DTCMP_Op_create(MPI_UNSIGNED_SHORT, dtcmp_op_fn_unsignedshort_ascend,  &DTCMP_OP_UNSIGNEDSHORT_ASCEND);
    DTCMP_Op_create(MPI_UNSIGNED_SHORT, dtcmp_op_fn_unsignedshort_descend, &DTCMP_OP_UNSIGNEDSHORT_DESCEND);
    DTCMP_Op_create(MPI_UNSIGNED, dtcmp_op_fn_unsignedint_ascend,  &DTCMP_OP_UNSIGNEDINT_ASCEND);
    DTCMP_Op_create(MPI_UNSIGNED, dtcmp_op_fn_unsignedint_descend, &DTCMP_OP_UNSIGNEDINT_DESCEND);
    DTCMP_Op_create(MPI_UNSIGNED_LONG, dtcmp_op_fn_unsignedlong_ascend,  &DTCMP_OP_UNSIGNEDLONG_ASCEND);
    DTCMP_Op_create(MPI_UNSIGNED_LONG, dtcmp_op_fn_unsignedlong_descend, &DTCMP_OP_UNSIGNEDLONG_DESCEND);
    DTCMP_Op_create(MPI_UNSIGNED_LONG_LONG, dtcmp_op_fn_unsignedlonglong_ascend,  &DTCMP_OP_UNSIGNEDLONGLONG_ASCEND);
    DTCMP_Op_create(MPI_UNSIGNED_LONG_LONG, dtcmp_op_fn_unsignedlonglong_descend, &DTCMP_OP_UNSIGNEDLONGLONG_DESCEND);
    DTCMP_Op_create(MPI_INT8_T,  dtcmp_op_fn_int8t_ascend,   &DTCMP_OP_INT8T_ASCEND);
    DTCMP_Op_create(MPI_INT8_T,  dtcmp_op_fn_int8t_descend,  &DTCMP_OP_INT8T_DESCEND);
    DTCMP_Op_create(MPI_INT16_T, dtcmp_op_fn_int16t_ascend,  &DTCMP_OP_INT16T_ASCEND);
    DTCMP_Op_create(MPI_INT16_T, dtcmp_op_fn_int16t_descend, &DTCMP_OP_INT16T_DESCEND);
    DTCMP_Op_create(MPI_INT32_T, dtcmp_op_fn_int32t_ascend,  &DTCMP_OP_INT32T_ASCEND);
    DTCMP_Op_create(MPI_INT32_T, dtcmp_op_fn_int32t_descend, &DTCMP_OP_INT32T_DESCEND);
    DTCMP_Op_create(MPI_INT64_T, dtcmp_op_fn_int64t_ascend,  &DTCMP_OP_INT64T_ASCEND);
    DTCMP_Op_create(MPI_INT64_T, dtcmp_op_fn_int64t_descend, &DTCMP_OP_INT64T_DESCEND);
    DTCMP_Op_create(MPI_UINT8_T,  dtcmp_op_fn_uint8t_ascend,   &DTCMP_OP_UINT8T_ASCEND);
    DTCMP_Op_create(MPI_UINT8_T,  dtcmp_op_fn_uint8t_descend,  &DTCMP_OP_UINT8T_DESCEND);
    DTCMP_Op_create(MPI_UINT16_T, dtcmp_op_fn_uint16t_ascend,  &DTCMP_OP_UINT16T_ASCEND);
    DTCMP_Op_create(MPI_UINT16_T, dtcmp_op_fn_uint16t_descend, &DTCMP_OP_UINT16T_DESCEND);
    DTCMP_Op_create(MPI_UINT32_T, dtcmp_op_fn_uint32t_ascend,  &DTCMP_OP_UINT32T_ASCEND);
    DTCMP_Op_create(MPI_UINT32_T, dtcmp_op_fn_uint32t_descend, &DTCMP_OP_UINT32T_DESCEND);
    DTCMP_Op_create(MPI_UINT64_T, dtcmp_op_fn_uint64t_ascend,  &DTCMP_OP_UINT64T_ASCEND);
    DTCMP_Op_create(MPI_UINT64_T, dtcmp_op_fn_uint64t_descend, &DTCMP_OP_UINT64T_DESCEND);
    DTCMP_Op_create(MPI_FLOAT, dtcmp_op_fn_float_ascend,  &DTCMP_OP_FLOAT_ASCEND);
    DTCMP_Op_create(MPI_FLOAT, dtcmp_op_fn_float_descend, &DTCMP_OP_FLOAT_DESCEND);
    DTCMP_Op_create(MPI_DOUBLE, dtcmp_op_fn_double_ascend,  &DTCMP_OP_DOUBLE_ASCEND);
    DTCMP_Op_create(MPI_DOUBLE, dtcmp_op_fn_double_descend, &DTCMP_OP_DOUBLE_DESCEND);
    DTCMP_Op_create(MPI_LONG_DOUBLE, dtcmp_op_fn_longdouble_ascend,  &DTCMP_OP_LONGDOUBLE_ASCEND);
    DTCMP_Op_create(MPI_LONG_DOUBLE, dtcmp_op_fn_longdouble_descend, &DTCMP_OP_LONGDOUBLE_DESCEND);
  }

  /* increment the number of times init has been called */
  dtcmp_init_count++;

  return DTCMP_SUCCESS;
}

/* finalize the sorting library and set static values back
 * to their pre-init state */
int DTCMP_Finalize()
{
  /* decrement our reference count for number of init calls */
  dtcmp_init_count--;

  /* if we hit 0, free everything off */
  if (dtcmp_init_count == 0) {
    /* free off predefined cmp handles */
    DTCMP_Op_free(&DTCMP_OP_LONGDOUBLE_DESCEND);
    DTCMP_Op_free(&DTCMP_OP_LONGDOUBLE_ASCEND);
    DTCMP_Op_free(&DTCMP_OP_DOUBLE_DESCEND);
    DTCMP_Op_free(&DTCMP_OP_DOUBLE_ASCEND);
    DTCMP_Op_free(&DTCMP_OP_FLOAT_DESCEND);
    DTCMP_Op_free(&DTCMP_OP_FLOAT_ASCEND);
    DTCMP_Op_free(&DTCMP_OP_UINT64T_DESCEND);
    DTCMP_Op_free(&DTCMP_OP_UINT64T_ASCEND);
    DTCMP_Op_free(&DTCMP_OP_UINT32T_DESCEND);
    DTCMP_Op_free(&DTCMP_OP_UINT32T_ASCEND);
    DTCMP_Op_free(&DTCMP_OP_UINT16T_DESCEND);
    DTCMP_Op_free(&DTCMP_OP_UINT16T_ASCEND);
    DTCMP_Op_free(&DTCMP_OP_UINT8T_DESCEND);
    DTCMP_Op_free(&DTCMP_OP_UINT8T_ASCEND);
    DTCMP_Op_free(&DTCMP_OP_INT64T_DESCEND);
    DTCMP_Op_free(&DTCMP_OP_INT64T_ASCEND);
    DTCMP_Op_free(&DTCMP_OP_INT32T_DESCEND);
    DTCMP_Op_free(&DTCMP_OP_INT32T_ASCEND);
    DTCMP_Op_free(&DTCMP_OP_INT16T_DESCEND);
    DTCMP_Op_free(&DTCMP_OP_INT16T_ASCEND);
    DTCMP_Op_free(&DTCMP_OP_INT8T_DESCEND);
    DTCMP_Op_free(&DTCMP_OP_INT8T_ASCEND);
    DTCMP_Op_free(&DTCMP_OP_UNSIGNEDLONGLONG_DESCEND);
    DTCMP_Op_free(&DTCMP_OP_UNSIGNEDLONGLONG_ASCEND);
    DTCMP_Op_free(&DTCMP_OP_UNSIGNEDLONG_DESCEND);
    DTCMP_Op_free(&DTCMP_OP_UNSIGNEDLONG_ASCEND);
    DTCMP_Op_free(&DTCMP_OP_UNSIGNEDINT_DESCEND);
    DTCMP_Op_free(&DTCMP_OP_UNSIGNEDINT_ASCEND);
    DTCMP_Op_free(&DTCMP_OP_UNSIGNEDSHORT_DESCEND);
    DTCMP_Op_free(&DTCMP_OP_UNSIGNEDSHORT_ASCEND);
    DTCMP_Op_free(&DTCMP_OP_LONGLONG_DESCEND);
    DTCMP_Op_free(&DTCMP_OP_LONGLONG_ASCEND);
    DTCMP_Op_free(&DTCMP_OP_LONG_DESCEND);
    DTCMP_Op_free(&DTCMP_OP_LONG_ASCEND);
    DTCMP_Op_free(&DTCMP_OP_INT_DESCEND);
    DTCMP_Op_free(&DTCMP_OP_INT_ASCEND);
    DTCMP_Op_free(&DTCMP_OP_SHORT_DESCEND);
    DTCMP_Op_free(&DTCMP_OP_SHORT_ASCEND);

    if (dtcmp_reduceop_mms_3uint64t != MPI_OP_NULL) {
      MPI_Op_free(&dtcmp_reduceop_mms_3uint64t);
      dtcmp_reduceop_mms_3uint64t = MPI_OP_NULL;
    }

    if (dtcmp_type_3uint64t != MPI_DATATYPE_NULL) {
      MPI_Type_free(&dtcmp_type_3uint64t);
      dtcmp_type_3uint64t = MPI_DATATYPE_NULL;
    }

    /* free our copy of comm_self */
    if (dtcmp_comm_self != MPI_COMM_NULL) {
      MPI_Comm_free(&dtcmp_comm_self);
      dtcmp_comm_self = MPI_COMM_NULL;
    }
  }

  return DTCMP_SUCCESS;
}

/* make a full copy of a comparison operation */
int DTCMP_Op_dup(DTCMP_Op cmp, DTCMP_Op* newcmp)
{
  /* check parameters */
  if (newcmp == NULL) {
    return DTCMP_FAILURE;
  }

  /* make a copy of cmp in newcmp */
  dtcmp_op_copy(newcmp, cmp);

  return DTCMP_SUCCESS;
}

/* create a user-defined comparison operation, associate compare function
 * pointer and datatype of key */
int DTCMP_Op_create(MPI_Datatype key, DTCMP_Op_fn fn, DTCMP_Op* cmp)
{
  /* check parameters */
  if (! dtcmp_type_is_valid(key)) {
    return DTCMP_FAILURE;
  }
  if (cmp == NULL) {
    return DTCMP_FAILURE;
  }

  /* allocate a handle and fill in its type and function pointer */
  dtcmp_op_init(DTCMP_OP_TYPE_BASIC, key, fn, DTCMP_OP_NULL, cmp);

  return DTCMP_SUCCESS;
}

/* create a series comparison which executes the first comparison operation
 * and then the second if the first evaluates to equal */
int DTCMP_Op_create_series(DTCMP_Op first, DTCMP_Op second, DTCMP_Op* cmp)
{
  /* check parameters */
  if (cmp == NULL) {
    return DTCMP_FAILURE;
  }

  /* copy the first into cmp */
  dtcmp_op_copy(cmp, first);

  /* traverse to last in series of first */
  dtcmp_op_handle_t* c = (dtcmp_op_handle_t*)(*cmp);
  while (c->series != DTCMP_OP_NULL) {
    c = (dtcmp_op_handle_t*)(c->series);
  }

  /* set this item to OP_SERIES */
  c->type = DTCMP_OP_TYPE_SERIES;

  /* make a full copy of the second op and attach it */
  DTCMP_Op copy;
  dtcmp_op_copy(&copy, second);
  c->series = copy;

  return DTCMP_SUCCESS;
}

/* create a series comparison which executes the first comparison operation
 * and then the second if the first evaluates to equal */
int DTCMP_Op_create_hseries(DTCMP_Op first, MPI_Aint cmpdisp, MPI_Aint disp, DTCMP_Op second, DTCMP_Op* cmp)
{
  /* check parameters */
  if (cmp == NULL) {
    return DTCMP_FAILURE;
  }

  /* copy the first into cmp */
  dtcmp_op_copy(cmp, first);

  /* set the cmpdisp value */
  dtcmp_op_handle_t* c = (dtcmp_op_handle_t*)(*cmp);
  c->cmpdisp = cmpdisp;

  /* traverse to last in series of first, and count off our
   * displacement as we go */
  while (c->series != DTCMP_OP_NULL) {
    disp -= c->disp;
    c = (dtcmp_op_handle_t*)(c->series);
  }

  /* set this item to OP_SERIES and adjust the displacement */
  c->type = DTCMP_OP_TYPE_SERIES;
  c->disp = disp;

  /* make a full copy of the second op and attach it */
  DTCMP_Op copy;
  dtcmp_op_copy(&copy, second);
  c->series = copy;

  return DTCMP_SUCCESS;
}

/* free object referenced by comparison operation handle */
int DTCMP_Op_free(DTCMP_Op* cmp)
{
  if (cmp != NULL && *cmp != DTCMP_OP_NULL) {
    dtcmp_op_handle_t* c = (dtcmp_op_handle_t*)(*cmp);
    MPI_Type_free(&(c->key));
    if (c->series != DTCMP_OP_NULL) {
      DTCMP_Op_free(&(c->series));
    }
    free(*cmp);
    *cmp = DTCMP_OP_NULL;
    return DTCMP_SUCCESS;
  } else {
    return DTCMP_FAILURE;
  }
}

int DTCMP_Op_eval(const void* a, const void* b, DTCMP_Op cmp, int* flag)
{
  if (flag != NULL && cmp != DTCMP_OP_NULL) {
    *flag = dtcmp_op_eval(a, b, cmp);
    return DTCMP_SUCCESS;
  } else {
    return DTCMP_FAILURE;
  }
}

/* free resources associated with handle, if any */
int DTCMP_Free(DTCMP_Handle* handle)
{
  if (handle != NULL && *handle != DTCMP_HANDLE_NULL) {
    dtcmp_handle_free_fn* fn = (dtcmp_handle_free_fn*)(*handle);
    return (*fn)(handle);
  }
  return DTCMP_SUCCESS;
}

/* TODO: turn this into a macro */
/* copy memory from srcbuf to dstbuf using committed MPI datatypes */
int DTCMP_Memcpy(
  void* dstbuf,       int dstcount, MPI_Datatype dsttype,
  const void* srcbuf, int srccount, MPI_Datatype srctype)
{
  /* execute sendrecv to ourself on comm_self */
  MPI_Sendrecv(
    (void*)srcbuf, srccount, srctype, 0, 999,
    dstbuf,        dstcount, dsttype, 0, 999,
    dtcmp_comm_self, MPI_STATUS_IGNORE
  );

  return DTCMP_SUCCESS;
}

int DTCMP_Search_low_local(
  const void* target,
  const void* list,
  int low,
  int high,
  MPI_Datatype key,
  MPI_Datatype keysat,
  DTCMP_Op cmp,
  DTCMP_Flags hints,
  int* flag,
  int* index)
{
  /* check parameters */
  if (target == NULL || flag == NULL || list == NULL) {
    return DTCMP_FAILURE;
  }
  if (! dtcmp_type_is_valid(key)) {
    return DTCMP_FAILURE;
  }
  if (! dtcmp_type_is_valid(keysat)) {
    return DTCMP_FAILURE;
  }

  return DTCMP_Search_low_local_binary(target, list, low, high, key, keysat, cmp, hints, flag, index);
}

int DTCMP_Search_high_local(
  const void* target,
  const void* list,
  int low,
  int high,
  MPI_Datatype key,
  MPI_Datatype keysat,
  DTCMP_Op cmp,
  DTCMP_Flags hints,
  int* flag,
  int* index)
{
  /* check parameters */
  if (target == NULL || flag == NULL || list == NULL) {
    return DTCMP_FAILURE;
  }
  if (! dtcmp_type_is_valid(key)) {
    return DTCMP_FAILURE;
  }
  if (! dtcmp_type_is_valid(keysat)) {
    return DTCMP_FAILURE;
  }

  return DTCMP_Search_high_local_binary(target, list, low, high, key, keysat, cmp, hints, flag, index);
}

int DTCMP_Search_low_list_local(
  int num,
  const void* targets,
  const void* list,
  int low,
  int high,
  MPI_Datatype key,
  MPI_Datatype keysat,
  DTCMP_Op cmp,
  DTCMP_Flags hints,
  int flags[],
  int indicies[])
{
  /* check parameters */
  if (num < 0) {
    return DTCMP_FAILURE;
  }
  if (num > 0 && (targets == NULL || flags == NULL || indicies == NULL)) {
    return DTCMP_FAILURE;
  }
  if (! dtcmp_type_is_valid(key)) {
    return DTCMP_FAILURE;
  }
  if (! dtcmp_type_is_valid(keysat)) {
    return DTCMP_FAILURE;
  }

  return DTCMP_Search_low_list_local_binary(num, targets, list, low, high, key, keysat, cmp, hints, flags, indicies);
}

int DTCMP_Partition_local(
  void* buf,
  int count,
  int inpivot,
  int* outpivot,
  MPI_Datatype key,
  MPI_Datatype keysat,
  DTCMP_Op cmp,
  DTCMP_Flags hints)
{
  /* check parameters */
  if (count < 0) {
    return DTCMP_FAILURE;
  }
  if (count > 0 && buf == NULL) {
    return DTCMP_FAILURE;
  }
  if (! dtcmp_type_is_valid(key)) {
    return DTCMP_FAILURE;
  }
  if (! dtcmp_type_is_valid(keysat)) {
    return DTCMP_FAILURE;
  }

  /* TODO: if we determine key and keysat don't have holes,
   * we could call the memcpy routine instead */

  return DTCMP_Partition_local_dtcpy(buf, count, inpivot, outpivot, key, keysat, cmp, hints);
}

int DTCMP_Merge_local(
  int num,
  const void* inbufs[],
  int counts[],
  void* outbuf,
  MPI_Datatype key,
  MPI_Datatype keysat,
  DTCMP_Op cmp,
  DTCMP_Flags hints)
{
  /* check parameters */
  if (num < 2) {
    return DTCMP_FAILURE;
  }
  if (num > 0 && (inbufs == NULL || counts == NULL || outbuf == NULL)) {
    return DTCMP_FAILURE;
  }
  if (! dtcmp_type_is_valid(key)) {
    return DTCMP_FAILURE;
  }
  if (! dtcmp_type_is_valid(keysat)) {
    return DTCMP_FAILURE;
  }

  if (num == 2) {
    /* O(N) time */
    return DTCMP_Merge_local_2way(num, inbufs, counts, outbuf, key, keysat, cmp, hints);
  } else {
    /* O(log(num) * N) time */
    return DTCMP_Merge_local_kway_heap(num, inbufs, counts, outbuf, key, keysat, cmp, hints);
  }
}

int DTCMP_Select_local(
  const void* buf,
  int num,
  int k,
  void* item,
  MPI_Datatype key,
  MPI_Datatype keysat,
  DTCMP_Op cmp,
  DTCMP_Flags hints)
{
  /* check parameters */
  if (num <= 0) {
    return DTCMP_FAILURE;
  }
  if (buf == NULL || item == NULL) {
    return DTCMP_FAILURE;
  }
  if (k < 0 || k >= num) {
    return DTCMP_FAILURE;
  }
  if (! dtcmp_type_is_valid(key)) {
    return DTCMP_FAILURE;
  }
  if (! dtcmp_type_is_valid(keysat)) {
    return DTCMP_FAILURE;
  }

  /* if k == 0 or k == num-1, special case this by finding
   * the minimum or maximum, which we can do in a single sweep
   * with minimal memory copies */
  if (k == 0 || k == num-1) {
    return DTCMP_Select_local_ends(buf, num, k, item, key, keysat, cmp, hints);
  }

  /* otherwise use random pivot partitioning to narrow down on target rank */
  return DTCMP_Select_local_randpartition(buf, num, k, item, key, keysat, cmp, hints);
}

int DTCMP_Selectv(
  const void* buf,
  int num,
  int k,
  void* item,
  MPI_Datatype key,
  MPI_Datatype keysat,
  DTCMP_Op cmp,
  DTCMP_Flags hints,
  MPI_Comm comm)
{
  /* check parameters */
  if (num <= 0) {
    return DTCMP_FAILURE;
  }
  if (buf == NULL || item == NULL) {
    return DTCMP_FAILURE;
  }
  if (! dtcmp_type_is_valid(key)) {
    return DTCMP_FAILURE;
  }
  if (! dtcmp_type_is_valid(keysat)) {
    return DTCMP_FAILURE;
  }

#if 0
  /* execute allreduce to compute min/max counts per process,
   * and sum of all elements */ 
  uint64_t min, max, sum;
  dtcmp_get_uint64t_min_max_sum(count, &min, &max, &sum, comm);

  /* check that k is in range */
  if (k < 0 || k >= sum) {
    return DTCMP_FAILURE;
  }

  /* if k == 0 or k == num-1, special case this by finding
   * the minimum or maximum, which we can do in a single sweep
   * with minimal memory copies */
  if (k == 0 || k == sum-1) {
    return DTCMP_Select_local_ends(buf, num, k, item, key, keysat, cmp, hints);
  }

#endif

  /* TODO: add Christian Siebert's median selection, which can be cleanly done
   * with allreduce and split operations (to exclude procs with 0 counts) */

  /* otherwise use random pivot partitioning to narrow down on target rank */
  return DTCMP_Selectv_medianofmedians(buf, num, k, item, key, keysat, cmp, hints, comm);
}

/* execute a purely local sort */
int DTCMP_Sort_local(
  const void* inbuf, 
  void* outbuf,
  int count,
  MPI_Datatype key,
  MPI_Datatype keysat,
  DTCMP_Op cmp,
  DTCMP_Flags hints)
{
  /* check parameters */
  if (count < 0) {
    return DTCMP_FAILURE;
  }
  if (count > 0 && outbuf == NULL) {
    return DTCMP_FAILURE;
  }
  if (! dtcmp_type_is_valid(key)) {
    return DTCMP_FAILURE;
  }
  if (! dtcmp_type_is_valid(keysat)) {
    return DTCMP_FAILURE;
  }

  /* TODO: select algorithm based on number of elements */

  return DTCMP_Sort_local_randquicksort(inbuf, outbuf, count, key, keysat, cmp, hints);
  return DTCMP_Sort_local_insertionsort(inbuf, outbuf, count, key, keysat, cmp, hints);
  return DTCMP_Sort_local_mergesort(inbuf, outbuf, count, key, keysat, cmp, hints);

  /* if keysat is valid type and if function is basic, we can just call qsort */
  dtcmp_op_handle_t* c = (dtcmp_op_handle_t*) cmp;
  if (c->type == DTCMP_OP_TYPE_BASIC) {
    return DTCMP_Sort_local_qsort(inbuf, outbuf, count, key, keysat, cmp, hints);
  }
}

int DTCMP_Sort(
  const void* inbuf,
  void* outbuf,
  int count,
  MPI_Datatype key,
  MPI_Datatype keysat,
  DTCMP_Op cmp,
  DTCMP_Flags hints,
  MPI_Comm comm)
{
  /* check parameters */
  if (count < 0) {
    return DTCMP_FAILURE;
  }
  if (count > 0 && outbuf == NULL) {
    return DTCMP_FAILURE;
  }
  if (! dtcmp_type_is_valid(key)) {
    return DTCMP_FAILURE;
  }
  if (! dtcmp_type_is_valid(keysat)) {
    return DTCMP_FAILURE;
  }

  /* if comm is a single rank, call sort_local */
  int ranks;
  MPI_Comm_size(comm, &ranks);
  if (ranks < 2) {
    return DTCMP_Sort_local(inbuf, outbuf, count, key, keysat, cmp, hints);
  }

  /* execute allreduce to compute min/max counts per process,
   * and sum of all elements */ 
  uint64_t min, max, sum;
  dtcmp_get_uint64t_min_max_sum(count, &min, &max, &sum, comm);

  /* nothing to do if the total element count is 0 */
  if (sum == 0) {
    return DTCMP_SUCCESS;
  }

  /* TODO: pick algorithm based on number of items */

  return DTCMP_Sort_allgather(inbuf, outbuf, count, key, keysat, cmp, hints, comm);
}

int DTCMP_Sortv(
  const void* inbuf,
  void* outbuf,
  int count,
  MPI_Datatype key,
  MPI_Datatype keysat,
  DTCMP_Op cmp,
  DTCMP_Flags hints,
  MPI_Comm comm)
{
  /* check parameters */
  if (count < 0) {
    return DTCMP_FAILURE;
  }
  if (count > 0 && outbuf == NULL) {
    return DTCMP_FAILURE;
  }
  if (! dtcmp_type_is_valid(key)) {
    return DTCMP_FAILURE;
  }
  if (! dtcmp_type_is_valid(keysat)) {
    return DTCMP_FAILURE;
  }

  /* if comm is a single rank, call sort_local */
  int ranks;
  MPI_Comm_size(comm, &ranks);
  if (ranks < 2) {
    return DTCMP_Sort_local(inbuf, outbuf, count, key, keysat, cmp, hints);
  }

  /* execute allreduce to compute min/max counts per process,
   * and sum of all elements */ 
  uint64_t min, max, sum;
  dtcmp_get_uint64t_min_max_sum(count, &min, &max, &sum, comm);

  /* nothing to do if the total element count is 0 */
  if (sum == 0) {
    return DTCMP_SUCCESS;
  }

  /* if min==max, then just invoke Sort() routine */
  if (min == max) {
    return DTCMP_Sort(inbuf, outbuf, count, key, keysat, cmp, hints, comm);
  }

  /* if sum(counts) is small gather to one task */

  /* if we have any elements, invoke the sortv routines */
  return DTCMP_Sortv_sortgather_scatter(inbuf, outbuf, count, key, keysat, cmp, hints, comm);
  return DTCMP_Sortv_allgather(inbuf, outbuf, count, key, keysat, cmp, hints, comm);
}

int DTCMP_Sortz(
  const void* inbuf,
  int count,
  void** outbuf,
  int* outcount,
  MPI_Datatype key,
  MPI_Datatype keysat,
  DTCMP_Op cmp,
  DTCMP_Flags hints,
  MPI_Comm comm, 
  DTCMP_Handle* handle)
{
  /* check parameters */
  if (count < 0) {
    return DTCMP_FAILURE;
  }
  if (outbuf == NULL || outcount == NULL) {
    return DTCMP_FAILURE;
  }
  if (! dtcmp_type_is_valid(key)) {
    return DTCMP_FAILURE;
  }
  if (! dtcmp_type_is_valid(keysat)) {
    return DTCMP_FAILURE;
  }

  /* execute allreduce to compute min/max counts per process,
   * and sum of all elements */ 
  uint64_t min, max, sum;
  dtcmp_get_uint64t_min_max_sum(count, &min, &max, &sum, comm);

  /* nothing to do if the total element count is 0 */
  if (sum == 0) {
    dtcmp_handle_alloc_single(0, outbuf, handle);
    *outcount = 0;
    return DTCMP_SUCCESS;
  }

  /* for now, we can only use sample sort if min==max */
  if (min == max) {
    /* TODO: if number of elements per process is small, call bitonic sort */
    return DTCMP_Sortz_samplesort(inbuf, count, outbuf, outcount, key, keysat, cmp, hints, comm, handle);
  }

  /* otherwise, force things into a Sortv but where we allocate and
   * return memory with a handle */
  MPI_Aint keysat_true_lb, keysat_true_extent;
  MPI_Type_get_true_extent(keysat, &keysat_true_lb, &keysat_true_extent);
  size_t outbuf_size = count * keysat_true_extent;
  dtcmp_handle_alloc_single(outbuf_size, outbuf, handle);
  DTCMP_Sortv(inbuf, *outbuf, count, key, keysat, cmp, hints, comm);
  *outcount = count;

  return DTCMP_FAILURE;
}

int DTCMP_Rankv(
  int count,
  const void* buf,
  int* groups,
  int  group_id[],
  int  group_ranks[],
  int  group_rank[],
  MPI_Datatype key,
  MPI_Datatype keysat,
  DTCMP_Op cmp,
  DTCMP_Flags hints,
  MPI_Comm comm)
{
  /* check parameters */
  if (count < 0) {
    return DTCMP_FAILURE;
  }
  if (count > 0 && buf == NULL) {
    return DTCMP_FAILURE;
  }
  if (! dtcmp_type_is_valid(key)) {
    return DTCMP_FAILURE;
  }
  if (! dtcmp_type_is_valid(keysat)) {
    return DTCMP_FAILURE;
  }

  return DTCMP_Rankv_sort(
    count, buf,
    groups, group_id, group_ranks, group_rank,
    key, keysat, cmp, hints, comm
  );
}

int DTCMP_Rankv_strings(
  int count,
  const char* strings[],
  int* groups,
  int  group_id[],
  int  group_ranks[],
  int  group_rank[],
  DTCMP_Flags hints,
  MPI_Comm comm)
{
  /* check parameters */
  if (count < 0) {
    return DTCMP_FAILURE;
  }
  if (count > 0 && strings == NULL) {
    return DTCMP_FAILURE;
  }

  return DTCMP_Rankv_strings_sort(count, strings, groups, group_id, group_ranks, group_rank, hints, comm);
}
