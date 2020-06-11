# Overview
The Datatype Comparison ([DTCMP](src/dtcmp.h)) Library provides pre-defined and
user-defined comparison operations to compare the values of two items
which can be arbitrary MPI datatypes.  Using these comparison
operations, the library provides various routines for manipulating
data, which may be distributed over the processes of an MPI
communicator including:

 - search - search for a target value in an ordered list of values
 - merge  - combine multiple ordered lists into a single ordered list
 - partition - divide a list of items into lower and higher values around a specified pivot value
 - select - identify the kth largest value
 - sort - sort data items into an ordered list
 - rank - assign group ids and ranks to a list of items
 - scan - execute a segmented scan on an ordered list of values

The DTCMP library is designed to provide a high-level interface to the
above functionality.  These high-level routines will invoke various
algorithm implementations to achieve the desired output.  The goal of
DTCMP is to be efficient given the input and the data distribution
among processes.  It is also intended to be portable to different
platforms and to allow for easy addition of new algorithms over time.

While performance is important, the goal is not to provide the fastest
routines.  The generality provided by the DTCMP API that makes
portability possible also tends to reduce performance in some respects,
e.g., forcing memory copies, abstracting some details about datatype,
etc.  Most likely a hand-tuned algorithm for the precise problem at
hand will always be faster than DTCMP.  However, DTCMP should be fast,
efficient, and portable, so it will generally be a good option except
for those cases where the application bottleneck demands absolute
performance.

Currently, the following pre-defined comparison operations are
provided. More will be added with time.  All pre-defined operations
have the following naming convention:

    DTCMP_OP_<TYPE>_<DIRECTION>

where TYPE may be one of:

    SHORT            - C short
    INT              - C int
    LONG             - C long
    LONGLONG         - C long long
    UNSIGNEDSHORT    - C unsigned short
    UNSIGNEDINT      - C unsigned int
    UNSIGNEDLONG     - C unsigned long
    UNSIGNEDLONGLONG - C unsigned long long
    INT8T            - C int8_t
    INT16T           - C int16_t
    INT32T           - C int32_t
    INT64T           - C int64_t
    UINT8T           - C uint8_t
    UINT16T          - C uint16_t
    UINT32T          - C uint32_t
    UINT64T          - C uint64_t
    FLOAT            - C float
    DOUBLE           - C double
    LONGDOUBLE       - C long double

and DIRECTION may be one of:

    ASCEND  - order values from smallest to largest
    DESCEND - order values from largest to smallest

Often when sorting data, each item contains a "key" that determines
its position within the global order and a "value", called "satellite
data", which travels with the key value but has no affect on its order.
DTCMP assumes that satellite data is relatively small and is attached
to the key in the same input buffer.  In many DTCMP routines, one must
specify the datatype for the key and another datatype for the key with
its satellite data.  The first is often named "key" and the second
"keysat".  The key datatype is used to infer the type and size of the
key when comparing key values.  This can be exploited to select
optimized comparison routines, e.g., radix sort on integers, and it
enables the library to siphon off and only process the key component if
needed.  The keysat type is needed to copy full items in memory or
transfer items between processes.

# Example
As an example use case for DTCMP, consider a problem in which each
process in MPI_COMM_WORLD has 10 items, each consisting of an integer
key and a integer satellite value.  One could use DTCMP to globally
sort and redistribute these items across the communicator like so:

    int inbuf[20] = {... 10 key/satellite pairs ...};
    int outbuf[20];
  
    MPI_Datatype keysat;
    MPI_Type_contiguous(2, MPI_INT, &keysat);
    MPI_Type_commit(&keysat);
  
    DTCMP_Sort(
      inbuf, outbuf, 10, MPI_INT, keysat,
      DTCMP_OP_INT_ASCEND, DTCMP_FLAG_NONE, MPI_COMM_WORLD
    );
  
    MPI_Type_free(&keysat);

Each process creates a datatype to describe the key with its satellite,
which is the keysat type that consists of 2 consecutive integers.  Then
each process calls DTCMP_Sort and provides a pointer to its input items
(inbuf), a buffer to hold the output items (outbuf), the number of
items in the input buffer (10), the datatype of the key (MPI_INT),
the datatype of the key together with its satellite data (keysat),
the comparison operation (DTCMP_OP_INT_ASCEND -- a pre-defined op to
sort integer keys in increasing order), an optional set of bit flags
(DTCMP_FLAG_NONE -- in this case, nothing), and the communicator on
which to execute the sort (MPI_COMM_WORLD).  The DTCMP_Sort routine is
collective over the processes in the communicator.  Upon return from
this routine, the items will be globally sorted, meaning the items
in the output buffer on each process will be in order and if i < j,
every item on rank i will come before every item on rank j.

The DTCMP_Sort routine assumes that each process provides the same
number of items for input.  If the number of input items varies across
processes, one can use DTCMP_Sortv instead.

In addition to the pre-defined comparison operations, DTCMP enables
users to create user-defined operations.  DTCMP_Op_create_series allows
one to create compound keys whose components are compared
lexicographically in series.  For example, if each process has a
compound key consisting of two ints which should be ordered with the
first int in increasing order and then the second int in decreasing
order, one can combine two pre-defined comparison operations like so:

    DTCMP_Op op;
    DTCMP_Op_create_series(DTCMP_OP_INT_ASCEND, DTCMP_OP_INT_DESCEND, &op);
    ... use op ...
    DTCMP_Op_free(&op);

This function will compare key values by the first comparison operation,
and then by the second if the first is equal.  One can chain together
an arbitrary number of comparision operations in this way, and the type
of each component may be different.  This enables one to create keys
consisting of arbitrary tuples.

The comparison operation encodes an "extent" which is used to advance
the pointer from one component to the next when the first comparison is
equal.  By default, this extent is the extent of the component type,
but one can override this default with the DTCMP_Op_create_hseries
routine, which takes a third parameter to specify the number of bytes
and direction (plus or minus) to locate the next key component after
the current one.  Encoding a negative extent in the comparison op
enables one to jump backwards in case less-significant components come
before more significant components.

One can also create a new operation with DTCMP_Op_create, which takes a
datatype to specify the key and a function pointer to be called to
compare two key values.  This function pointer has the same prototype
as a qsort() comparison operation:

    int my_compare_op(const void* a, const void* b);

Such a function should return:

    negative int if *a < *b
               0 if *a = *b
    positive int if *a > *b

Given such a function, one can create a new op like so:

    MPI_Datatype key; // datatype that describes key value
    DTCMP_Op op;
    DTCMP_Op_create(key, my_compare_op, &op);
    ... use op ...
    DTCMP_Op_free(&op);

The operation encodes the key datatype, its extent, and a pointer to
the comparison operation function.  Once a new operation has been
created with DTCMP_Op_create, it can be used in any DTCMP routine and
it can also be used as a component within a larger series.  DTCMP
currently only supports fixed-length keys.

Currently, key and keysat datatypes must adhere to a certain set of
constraints.  Namely, they cannot have holes, lb=0, extent > 0,
extent=true extent, and extent=size.

A common need is to sort strings, which may be of variable length.
Since the keys are variable length, there are not predefined operations
to handle strings.  However, one may still sort strings using an
algorithm like the following:

    1) define a string comparison function:

    int my_strcmp(const void* a, const void& b) {
      return strcmp((const char*)a, (const char*)b);
    }

    2) determine maximum string length across all procs:

    int my_size = strlen(my_str) + 1;
    MPI_Allreduce(&my_size, &max_size, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    3) allocate buffer of maximum length and copy string:

    char* my_new_str = malloc(max_size);
    strcpy(my_new_str, my_str);

    4) create a type of the max length using MPI_Type_contigious:

    MPI_Datatype my_type;
    MPI_Type_contiguous(max_size, MPI_CHAR, &my_type);
    MPI_Type_commit(&my_type);

    5) create a new DTCMP op with DTCMP_Op_create:

    DTCMP_Op my_op;
    DTCMP_Op_create(my_type, my_strcmp, &my_op);

    6) use copy of string, new type, and new op in any DTCMP calls

    7) free op and type

Since this use case is common, DTCMP includes two functions that package
steps 1, 4, and 5 above into a single routine:

    DTCMP_Str_create_ascend
    DTCMP_Str_create_descend

Given the number of characters in a fixed-length string, each function
returns a committed MPI_Datatype and a newly created DTCMP_Op bound to
strcmp.

TODO
====
Add mechanism to provide assertions in API:

 - sorted (locally & globally) - done
 - unique (locally & globally) - done
 - stable - caller is requesting that sort is stable
 - in_place - caller wants DTCMP to use in-place algorithms
 - deterministic - caller wants deterministic time algorithm

Routine to return communicator optimized for sorting
Find way to support variable length keys (e.g., strings)
Enable apps/libs to create DTCMP_Handles freeable via DTCMP_FREE:

    DTCMP_Handle_create(fn* my_delete, void* my_arg, DTCMP_Handle* out)

Build
=====
First build and install the LWGRP library, available at [http://github.com/llnl/lwgrp](http://github.com/llnl/lwgrp).

Then to build:

    ./configure --prefix <installdir> --with-lwgrp=<lwgrp_installdir>
    make
    make install
