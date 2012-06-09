/* Copyright (c) 2012, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by Adam Moody <moody20@llnl.gov>.
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
int dtcmp_op_fn_long_ascend(const void*, const void*);
int dtcmp_op_fn_long_descend(const void*, const void*);
int dtcmp_op_fn_longlong_ascend(const void*, const void*);
int dtcmp_op_fn_longlong_descend(const void*, const void*);
int dtcmp_op_fn_unsignedlong_ascend(const void*, const void*);
int dtcmp_op_fn_unsignedlong_descend(const void*, const void*);
int dtcmp_op_fn_unsignedlonglong_ascend(const void*, const void*);
int dtcmp_op_fn_unsignedlonglong_descend(const void*, const void*);
int dtcmp_op_fn_int8t_ascend(const void*, const void*);
int dtcmp_op_fn_int8t_descend(const void*, const void*);
int dtcmp_op_fn_int16t_ascend(const void*, const void*);
int dtcmp_op_fn_int16t_descend(const void*, const void*);
int dtcmp_op_fn_int32t_ascend(const void*, const void*);
int dtcmp_op_fn_int32t_descend(const void*, const void*);
int dtcmp_op_fn_int64t_ascend(const void*, const void*);
int dtcmp_op_fn_int64t_descend(const void*, const void*);
int dtcmp_op_fn_uint8t_ascend(const void*, const void*);
int dtcmp_op_fn_uint8t_descend(const void*, const void*);
int dtcmp_op_fn_uint16t_ascend(const void*, const void*);
int dtcmp_op_fn_uint16t_descend(const void*, const void*);
int dtcmp_op_fn_uint32t_ascend(const void*, const void*);
int dtcmp_op_fn_uint32t_descend(const void*, const void*);
int dtcmp_op_fn_uint64t_ascend(const void*, const void*);
int dtcmp_op_fn_uint64t_descend(const void*, const void*);
int dtcmp_op_fn_float_ascend(const void*, const void*);
int dtcmp_op_fn_float_descend(const void*, const void*);
int dtcmp_op_fn_double_ascend(const void*, const void*);
int dtcmp_op_fn_double_descend(const void*, const void*);
int dtcmp_op_fn_longdouble_ascend(const void*, const void*);
int dtcmp_op_fn_longdouble_descend(const void*, const void*);
