##
# Prologue.
##
AC_INIT
AC_CONFIG_SRCDIR([src/dtcmp.c])
AC_CONFIG_AUX_DIR(config)
AC_CONFIG_MACRO_DIR([m4])

AM_CONFIG_HEADER([config/config.h])

#AC_CANONICAL_SYSTEM

X_AC_META
X_AC_EXPAND_INSTALL_DIRS

##
# Automake support.
##
AM_INIT_AUTOMAKE([$META_NAME], [$META_VERSION]-[$META_RELEASE])

#AM_MAINTAINER_MODE

##
# Checks for programs.
##
AC_PROG_CC
AM_PROG_CC_C_O
AC_PROG_CXX
#X_AC_PROG_MPICC
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

