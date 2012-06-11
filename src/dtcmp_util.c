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
#include <string.h>

#include "mpi.h"
#include "dtcmp_internal.h"

/* malloc with some checks on size and the returned pointer, along with future support for alignment,
 * if size <= 0, malloc is not called and NULL pointer is returned
 * error is printed with file name and line number if size > 0 and malloc returns NULL pointer */
void* dtcmp_malloc(size_t size, size_t align, const char* file, int line)
{
  void* ptr = NULL;
  if (size > 0) {
/*
    if (align == 0) {
      ptr = malloc(size);
    } else {
      posix_memalign(&ptr, size, align);
    }
*/
    ptr = malloc(size);
    if (ptr == NULL) {
      printf("ERROR: Failed to allocate memory %lu bytes @ %s:%d\n", size, file, line);
      exit(1);
    }
  }
  return ptr;
}

/* careful to not call free if pointer is already NULL,
 * sets pointer value to NULL */
void dtcmp_free(void* p)
{
  /* we really receive a pointer to a pointer (void**),
   * but it's typed as void* so caller doesn't need to add casts all over the place */
  void** ptr = (void**) p;

  /* free associated memory and set pointer to NULL */
  if (ptr == NULL) {
    printf("ERROR: Expected address of pointer value, but got NULL instead @ %s:%d\n",
      __FILE__, __LINE__
    );
  } else if (*ptr != NULL) {
    free(*ptr);
    *ptr = NULL;
  }
}

int dtcmp_handle_alloc_single(size_t size, void** buf, DTCMP_Handle* handle)
{
  /* setup handle and buffer to merge data to return to user */
  void* ret_buf = NULL;
  size_t ret_buf_size = sizeof(dtcmp_handle_free_fn) + size;
  if (ret_buf_size > 0) {
    ret_buf = (void*) dtcmp_malloc(ret_buf_size, 0, __FILE__, __LINE__);
    if (ret_buf == NULL) {
      /* TODO: error */
    }
  }

  /* compute and allocate space to merge received data */
  char* ret_buf_tmp = (char*)ret_buf;
  if (ret_buf != NULL) {
    /* allocate and initialize function pointer as first item in handle struct */
    dtcmp_handle_free_fn* fn = (dtcmp_handle_free_fn*) ret_buf;
    *fn = dtcmp_handle_free_single;
    ret_buf_tmp += sizeof(dtcmp_handle_free_fn);

    /* record the start of the buffer */
    if (size > 0) {
      *buf = (void*) ret_buf_tmp;
    } else {
      *buf = NULL;
    }
  }

  /* set the handle value */
  *handle = ret_buf;

  return DTCMP_SUCCESS;
}

/* assumes that handle just points to one big block of memory that must be freed */
int dtcmp_handle_free_single(DTCMP_Handle* handle)
{
  if (handle != NULL && *handle != DTCMP_HANDLE_NULL) {
    free(*handle);
    *handle = DTCMP_HANDLE_NULL;
  }
  return DTCMP_SUCCESS;
}

/* execute reduction to compute min/max/sum over comm */
int dtcmp_get_uint64t_min_max_sum(int count, uint64_t* min, uint64_t* max, uint64_t* sum, MPI_Comm comm)
{
  /* initialize our input with our count value */
  uint64_t input[3];
  input[MMS_MIN] = (uint64_t) count;
  input[MMS_MAX] = (uint64_t) count;
  input[MMS_SUM] = (uint64_t) count;

  /* execute the allreduce */
  uint64_t output[3];
  MPI_Allreduce(input, output, 1, dtcmp_type_3uint64t, dtcmp_reduceop_mms_3uint64t, comm);

  /* copy result to output parameters */
  *min = output[MMS_MIN];
  *max = output[MMS_MAX];
  *sum = output[MMS_SUM];

  return DTCMP_SUCCESS;
}

/* builds and commits a new datatype that is the concatenation of the
 * list of old types, each oldtype should have no holes */
int dtcmp_type_concat(int num, const MPI_Datatype oldtypes[], MPI_Datatype* newtype)
{
  /* if there is nothing in the list, just return a NULL handle */
  if (num <= 0) {
    *newtype = MPI_DATATYPE_NULL;
    return DTCMP_SUCCESS;
  }

  /* TODO: could use a small array to handle most calls,
   * rather than allocate an array each time */

  /* allocate memory for blocklens, displs, and types arrays for
   * MPI_Type_create_struct */
  int* blocklens      = (int*) dtcmp_malloc(num * sizeof(int), 0, __FILE__, __LINE__);
  MPI_Aint* displs    = (MPI_Aint*) dtcmp_malloc(num * sizeof(MPI_Aint), 0, __FILE__, __LINE__);
  MPI_Datatype* types = (MPI_Datatype*) dtcmp_malloc(num * sizeof(MPI_Datatype), 0, __FILE__, __LINE__);

  /* build new key type */
  int i;
  MPI_Aint disp = 0;
  for (i = 0; i < num; i++) {
    blocklens[i] = 1;
    displs[i] = disp;
    types[i] = oldtypes[i];

    /* get true extent of current type */
    MPI_Aint true_lb, true_extent;
    MPI_Type_get_true_extent(oldtypes[i], &true_lb, &true_extent);
    disp += true_extent;
  }

  /* create and commit the new type */
  MPI_Type_create_struct(num, blocklens, displs, types, newtype);
  MPI_Type_commit(newtype);

  /* free memory */
  dtcmp_free(&types);
  dtcmp_free(&displs);
  dtcmp_free(&blocklens);

  return DTCMP_SUCCESS;
}

int dtcmp_type_concat2(MPI_Datatype type1, MPI_Datatype type2, MPI_Datatype* newtype)
{
  MPI_Datatype types[2];
  types[0] = type1;
  types[1] = type2;
  return dtcmp_type_concat(2, types, newtype);
}
