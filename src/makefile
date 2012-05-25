CC    = gcc
MPICC = mpicc

#MPI = -mpi=mvapich-gnu-debug-1.2
OPT = -g -O0 -Wall -fPIC

#MPI = 
#OPT = -g -O3 -Wall -fPIC

OBJS= \
  dtcmp.o \
  dtcmp_util.o \
  dtcmp_ops.o \
  dtcmp_search_binary.o \
  dtcmp_partition.o \
  dtcmp_merge_2way.o \
  dtcmp_merge_kway_heap.o \
  dtcmp_is_sorted.o \
  dtcmp_sort_local_insertionsort.o \
  dtcmp_sort_local_randquicksort.o \
  dtcmp_sort_local_mergesort.o \
  dtcmp_sort_local_qsort.o \
  dtcmp_sort_allgather.o \
  dtcmp_sort_bitonic.o \
  dtcmp_sortv_allgather.o \
  dtcmp_sortv_sortgather_scatter.o \
  dtcmp_sortv_cheng.o \
  dtcmp_rankv_sort.o \
  ranklist_allreduce_recursive.o \
  ranklist_allgather_brucks.o \
  ranklist_alltoall_brucks.o \
  ranklist_alltoallv_linear.o

all: clean libdtcmp.a main

libdtcmp.a: $(OBJS)
	ar r $@ $(OBJS)
	ranlib $@

%.o: %.c *.h
	$(MPICC) $(OPT) $(CFLAGS) -c $< -o $@ -I.

main: main.c *h libdtcmp.a
	$(MPICC) $(OPT) $(CFLAGS) $< libdtcmp.a -o $@ $(LDFLAGS)

clean:
	rm -rf include/* lib/* libdtcmp.a *.o main
