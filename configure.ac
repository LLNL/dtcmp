##
# Prologue.
##
AC_INIT([Datatype Comparison Library],[1.1.4],[],[dtcmp],[https://github.com/hpc/dtcmp])
AC_CONFIG_AUX_DIR(config)
AC_CONFIG_MACRO_DIR([m4])
AC_CONFIG_HEADERS([config/config.h])
AC_CONFIG_SRCDIR([src/dtcmp.c])

#AC_CANONICAL_SYSTEM

##
# Automake support.
##
AM_INIT_AUTOMAKE([foreign])

#AM_MAINTAINER_MODE

##
# Checks for programs.
##
AC_PROG_CC
AM_PROG_CC_C_O
AC_PROG_CXX
LX_FIND_MPI
X_AC_LWGRP

##
# enable libtool
##
AC_PROG_LIBTOOL

##
# Extra packages
##

##
# Checks for header files.
##
AC_HEADER_STDC

##
# Checks for typedefs, structures, and compiler characteristics.
##
AC_C_BIGENDIAN

##
# Checks for library functions.
##

##
# Epilogue.
##
AC_CONFIG_FILES([ \
  Makefile \
  src/Makefile \
  libdtcmp.pc \
])
AC_OUTPUT
