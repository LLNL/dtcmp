/* Copyright (c) 2012, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by Adam Moody <moody20@llnl.gov> and Edgar A. Leon <leon@llnl.gov>.
 * LLNL-CODE-557516.
 * All rights reserved.
 * This file is part of the DTCMP library.
 * For details, see https://github.com/hpc/dtcmp
 * Please also read this file: LICENSE.TXT. */

#include <stdlib.h>
#include <stdio.h>
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
      printf("ERROR: Failed to allocate memory @ %s:%d\n", file, line);
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
