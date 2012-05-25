/* Copyright (c) 2012, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by Adam Moody <moody20@llnl.gov> and Edgar A. Leon <leon@llnl.gov>.
 * LLNL-CODE-557516.
 * All rights reserved.
 * This file is part of the DTCMP library.
 * For details, see https://github.com/hpc/dtcmp
 * Please also read this file: LICENSE.TXT. */

#include "dtcmp_internal.h"

#define DTCMP_OP_TYPE_BASIC  (1)
#define DTCMP_OP_TYPE_SERIES (2)

void dtcmp_op_hinit(
  uint32_t type,
  MPI_Datatype key,
  DTCMP_Op_fn fn,
  MPI_Aint disp,
  DTCMP_Op series,
  DTCMP_Op* cmp
);

void dtcmp_op_init(
  uint32_t type,
  MPI_Datatype key,
  DTCMP_Op_fn fn,
  DTCMP_Op series,
  DTCMP_Op* cmp
);

void dtcmp_op_copy(DTCMP_Op* dst, DTCMP_Op src);

int dtcmp_op_eval(const void* a, const void* b, DTCMP_Op cmp);


int dtcmp_op_fn_int_ascend(const void*, const void*);
int dtcmp_op_fn_int_descend(const void*, const void*);
