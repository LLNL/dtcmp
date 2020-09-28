##*****************************************************************************
#  AUTHOR:
#    Adam Moody <moody20@llnl.gov>
#
#  SYNOPSIS:
#    X_AC_LWGRP()
#
#  DESCRIPTION:
#    Check the usual suspects for a LWGRP library installation,
#    setting LWGRP_CFLAGS, LWGRP_LDFLAGS, and LWGRP_LIBS as necessary.
#
#  WARNINGS:
#    This macro must be placed after AC_PROG_CC and before AC_PROG_LIBTOOL.
##*****************************************************************************

AC_DEFUN([X_AC_LWGRP], [

  _x_ac_lwgrp_dirs="/usr /opt/freeware"
  _x_ac_lwgrp_libs="lib64 lib"

  AC_ARG_WITH(
    [lwgrp],
    AS_HELP_STRING(--with-lwgrp=PATH,Specify liblwgrp path),
    [_x_ac_lwgrp_dirs="$withval"
     with_lwgrp=yes],
    [_x_ac_lwgrp_dirs=no
     with_lwgrp=no]
  )

  if test "x$_x_ac_lwgrp_dirs" = xno ; then
    # user explicitly wants to disable liblwgrp support
    LWGRP_CFLAGS=""
    LWGRP_LDFLAGS=""
    LWGRP_LIBS=""
    # TODO: would be nice to print some message here to record in the config log
  else
    # user wants liblwgrp enabled, so let's define it in the source code
    AC_DEFINE([HAVE_LIBLWGRP], [1], [Define if liblwgrp is available])

    # now let's locate the install location
    found=no

    # check for liblwgrp in a system default location if:
    #   --with-lwgrp or --without-lwgrp is not specified
    #   --with-lwgrp=yes is specified
    #   --with-lwgrp is specified
    if test "$with_lwgrp" = check || \
       test "x$_x_ac_lwgrp_dirs" = xyes || \
       test "x$_x_ac_lwgrp_dirs" = "x" ; then
       AC_CHECK_LIB([lwgrp], [lwgrp_comm_free])

      # if we found it, set the build flags
      if test "$ac_cv_lib_lwgrp_lwgrp_comm_free" = yes; then
        found=yes
        LWGRP_CFLAGS=""
        LWGRP_LDFLAGS=""
        LWGRP_LIBS="-llwgrp"
      fi
    fi

    # if we have not already found it, check the lwgrp_dirs
    if test "$found" = no; then
      AC_CACHE_CHECK(
        [for liblwgrp installation],
        [x_ac_cv_lwgrp_dir],
        [
          for d in $_x_ac_lwgrp_dirs; do
            test -d "$d" || continue
            test -d "$d/include" || continue
            test -f "$d/include/lwgrp.h" || continue
            for bit in $_x_ac_lwgrp_libs; do
              test -d "$d/$bit" || continue
        
# TODO: replace line below with link test
              x_ac_cv_lwgrp_dir=$d
#              _x_ac_lwgrp_libs_save="$LIBS"
#              LIBS="-L$d/$bit -llwgrp $LIBS $MPI_CLDFLAGS"
#              AC_LINK_IFELSE(
#                AC_LANG_CALL([], [LWGRP_Comm_split]),
#                AS_VAR_SET([x_ac_cv_lwgrp_dir], [$d]))
#              LIBS="$_x_ac_lwgrp_libs_save"
#              test -n "$x_ac_cv_lwgrp_dir" && break
            done
            test -n "$x_ac_cv_lwgrp_dir" && break
          done
      ])

      # if we found it, set the build flags
      if test -n "$x_ac_cv_lwgrp_dir"; then
        found=yes
        LWGRP_CFLAGS="-I$x_ac_cv_lwgrp_dir/include"
        LWGRP_LDFLAGS="-L$x_ac_cv_lwgrp_dir/$bit"
        LWGRP_LIBS="-llwgrp"
      fi
    fi

    # if we failed to find liblwgrp, throw an error
    if test "$found" = no ; then
      AC_MSG_ERROR([unable to locate liblwgrp installation])
    fi
  fi

  # lwgrp is required
  if test "x$LWGRP_LIBS" = "x" ; then
    AC_MSG_ERROR([unable to locate liblwgrp installation])
  fi

  # propogate the build flags to our makefiles
  AC_SUBST(LWGRP_CFLAGS)
  AC_SUBST(LWGRP_LDFLAGS)
  AC_SUBST(LWGRP_LIBS)
])
