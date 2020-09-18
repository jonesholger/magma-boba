#!/usr/bin/env python3
""" package-routine.py - package up a single routine into a distributable package

Example:

$ ./tools/package-routine.py dgesv

$ ./tools/package-routine.py dgesv sgeqrf

$ ./tools/package-routine.py -h



"""
## imports (all std)
import argparse
import os
import glob
import re
import shutil
from time import time
import errno

# for 'blas' endings
import magmasubs

# construct & parse given arguments
parser = argparse.ArgumentParser(description='Package a single MAGMA routine into a folder')

parser.add_argument('routines', nargs='+', help='Routines to package up (i.e. sgemm, dgemm, etc)')
parser.add_argument('-o', '--output', default=None, help='Destination folder (leave empty for a default)')
parser.add_argument('--interface', default='cuda', choices=['hip', 'cuda'], help='Which interface/backend to use?')

args = parser.parse_args()

# generate output directory
if not args.output:
    args.output = "magma_pkg_" + args.interface + "_" + "_".join(args.routines)

if args.interface not in ('cuda', 'hip',):
    raise Exception(f"Unknown interface requested: {args.interface}")


print (f"""Packaging routines: {args.routines} and storing in: {args.output}/""")



# escape sequence, so it can be in an fstring
_n = '\n'

# -*- Regex Definitions -*-

# regex for calling a MAGMA function
re_call = re.compile(r" *(magma(?:blas)?_\w+)\(")

# regex for a function definition, i.e. not just a declaration (must be multiline due to how many functions are declared)
re_funcdef = re.compile(r"(?:extern \"C\"|static inline)? ?\n?(?:[\w\* ]+ *?)\n?(magma(?:blas)?_\w+)( \n)*\([\w\[\]\* ,\n]*\)\n? *\n?\{", re.MULTILINE)

# regex for a macro definition
re_macdef = re.compile(r"#define  *(magma(?:blas)?_\w+)\(")

# regex for an include statement
re_include = re.compile(r"\#include (?:\"|\<)(magma[\w\.]+)(?:\"|\<)")


# -*- Sets/Collections


# all files possible
allfiles = {
    *glob.glob("src/*.cpp"),
    *glob.glob("control/*.cpp"),
    *glob.glob("include/*.h"),
    *glob.glob(f"interface_{args.interface}/*.cpp"),
}

if args.interface == 'cuda':
    allfiles |= {
        *glob.glob(f"magmablas/*.cpp"),
        *glob.glob(f"magmablas/*.cu"),
        *glob.glob(f"magmablas/*.cuh"),
        *glob.glob(f"magmablas/*.h"),
    }
else:
    allfiles |= {
        *glob.glob(f"magmablas_hip/*.cpp"),
        *glob.glob(f"magmablas_hip/*.hpp"),
        *glob.glob(f"magmablas_hip/*.h"),
    }


# set of all files included/used
set_c = { 'control/pthread_barrier.h', 'control/affinity.h', 'control/trace.h', 'control/batched_kernel_param.h', 'include/magma_v2.h', f"interface_{args.interface}/error.h",  }

# functions needed to distribute
funcs_needed = {*args.routines, 'magma_init', 'magma_finalize', 'magma_getmatrix_internal'}

# functions currently defined
funcs_defined = { 'magma_warn_leaks' }

# set of BLAS routines requested
blas_requested = set()

# functions that will emit a warning, due to some special case
funcs_warn = set()

# -*- Util Funcs -*-

# filter through files, and only return those that exist and have not yet been included
def newfiles(fls):
    for fl in fls:
        if fl not in set_c and os.path.exists(fl):
            yield fl

# Read entire file
def readall(fl):
    # read entire file
    src_file = open(fl, 'r')
    src = src_file.read()
    src_file.close()
    return src

# return a set of matches pf 'regex' in 'src'
# NOTE: most are default `group==1`, so that is defaulted
def matches(regex, src, group=1):
    ret = set()
    for match in re.finditer(regex, src):
        ret.add(match.group(group))
    return ret


allfiles -= set_c

# find out which functions to add
for func in args.routines:

    ct = 0

    # check a list of files
    for fl in newfiles([f"src/{func}.cpp"]):
        src = readall(fl)

        ct += 1

        # add functions which were defined
        funcs_defined.update(matches(re_funcdef, src))

        # add references to subroutines & other functions called
        funcs_needed.update(matches(re_call, src))

        # assume it contained it
        funcs_needed.remove(func)

        set_c.add(fl)

    if ct < 1:
        raise Exception(f"Unknown routine '{func}'")

st = time()

# now, while there are functions that are needed, search for them
funcs_needed -= funcs_defined
while funcs_needed:
    func = next(iter(funcs_needed))
    print (f"[func] searching for '{func}'... ({len(funcs_needed)} left)")

    # ensure it is a magma function
    if 'magma' not in func:
        raise Exception(f"Need function '{func}', which is not part of MAGMA!")

    magma_name = func.replace('magma_', '')

    # iterate through files the routine probably needs
    for fl in newfiles([f"src/{magma_name}.cpp", f"src/{''.join([i for i in magma_name if not i.isdigit()])}.cpp"]):
        src = readall(fl)

        # get matches and see if this file works
        defs = matches(re_funcdef, src) | matches(re_macdef, src)

        if func in defs:
            # found it, so we need to include this file
            set_c.add(fl)
            funcs_defined.update(defs)

            # we need to see what else is requested
            funcs_needed.update(matches(re_call, src))

            # we found the requested function, so stop looking for it
            break


    if func not in funcs_defined:
        # we haven't found anything valid yet
        isFound = False

        print ("  bruteforce searching...")
    
        if not isFound:
            if 'opencl' in func and 'opencl' not in args.interface:
                # we don't care about OpenCL functions
                funcs_needed.remove(func)
                isFound = True

        if not isFound:
            # check if it is a BLAS routine (in which case, it should
            #   be provided by someone else)
            for rout in magmasubs.blas:
                for prout in rout:
                    if prout in func:
                        funcs_defined.add(func)
                        blas_requested.add(magma_name)
                        isFound = True
                        break
        if not isFound:
            #print ("not yet found:", func)


            st1 = time()
            ct = 0
            # not a BLAS routine, so now just search everywhere for it
            for fl in newfiles(allfiles):
                src = readall(fl)
                ct += 1

                # get matches and see if this file works
                defs = matches(re_funcdef, src) | matches(re_macdef, src)
                
                if func in defs:
                    # found it, so we need to include this file
                    funcs_defined.update(defs)

                    # we need to see what else is requested (if not a header)
                    if fl[fl.index('.'):] not in ('.h', '.hh', '.hpp', '.cuh',):
                        funcs_needed.update(matches(re_call, src))

                    set_c.add(fl)
                    isFound = True
                    break
            print ("    bruteforce took: %.2f (%i files)" % (time() - st1, ct))
        
            if not isFound:
                #funcs_err.add(func)
                raise Exception(f"Could not find '{func}'")

    funcs_needed -= funcs_defined

print ("finding funcs took %.3fs" % (time() - st,))


st = time()

# new includse
keepGoing = True
while keepGoing:

    new_includes = set()

    for fl in set_c:
        src = readall(fl)
        for incfl in matches(re_include, src):
            possible = [
                f"include/{incfl}",
                f"control/{incfl}",
                f"magmablas/{incfl}" if args.interface == "cuda" else f"magmablas_{args.interface}/{incfl}",
            ]

            isFound = False
            for pos in possible:
                if pos in set_c:
                    isFound = True
                    break
            
            if isFound:
                continue

            # we need to find
            for pos in possible:
                if os.path.exists(pos):
                    new_includes.add(pos)
                    isFound = True
                    break
            if isFound:
                continue

            # not found
            raise Exception(f"Could not find included file '{incfl}'")

    set_c.update(new_includes)
    keepGoing = bool(new_includes)

print ("finding includes took %.3fs" % (time() - st,))

# -*- Generated Output -*-


# string for the README file 
README = f"""# MAGMA Packaged Routines

This folder includes the following routines:
{''.join("  * " + rout + _n for rout in args.routines)}

Interface requested: {args.interface}

"""

# update readme
README += f"""

# BLAS NEEDED

{''.join("  * " + fl + _n for fl in blas_requested)}

(for all of these, define `magma_<name>` to the correct one for your platform, for example: `#define magma_dgemm cublasZgemm`, or for no-op, you can use `#define magma_dgemm(...) (fprintf(stderr, "magma_dgemm(...)\n", 0))`

# INCLUDED FILES

{''.join("  * " + fl + _n for fl in set_c)}

# DEFINED FUNCTIONS

{''.join("  * " + fl + _n for fl in funcs_defined)}

# WARNINGS (these may take special attention)

{''.join("  * " + fl + _n for fl in funcs_warn)}

"""


TEST_C = f"""/* test.c - GENERATED test file to ensure magma compiles & can execute
 *
 * Generated by `package-routine.py`
 *
 * @author: Cade Brown <cade@utk.edu>
 */
 
#include <magma_v2.h>
#include <stdio.h>

int main(int argc, char** argv) {{
    // initialize
    int st;
    if ((st = magma_init()) != MAGMA_SUCCESS) {{
        fprintf(stderr, "magma_init() failed! (code: %i)\\n", st);
        return -1;
    }}
    
    if ((st = magma_finalize()) != MAGMA_SUCCESS) {{
        fprintf(stderr, "magma_finalize() failed! (code: %i)\\n", st);
        
        return -1;
    }}
    // success
    return 0;
}} 


"""

# string for `Makefile`
MAKEFILE = f"""# -*- Makefile - generated by `package-routine.py`

# variables
NVCC       ?= nvcc

# source files
MAGMA_C    := {" ".join(filter(lambda x: x.endswith('.cpp'), set_c))}

# object files
MAGMA_O    := $(patsubst %.cpp,%.o,$(MAGMA_C))

MAGMA_CFLAGS := -std=c++11 -DADD_ -DMIN_CUDA_ARCH=600 { {'hip': '-DHAVE_HIP', 'cuda': '-DHAVE_CUDA -DHAVE_CUBLAS'}[args.interface] }


default: libmagma_pkg.so test

# single file
%.o: %.cpp
\t$(NVCC) -I./include -I./control $(MAGMA_CFLAGS) $< -Xcompiler "-fPIC" -c -o $@

# compile magma embedded
# (i.e. `magmapkg`)
libmagma_pkg.so: $(MAGMA_O)
\t$(CC) $^ -lcublas -lcusparse -lcudart -lcudadevrt -shared -o $@

test: test.c libmagma_pkg.so
\t$(CC) -I./include -I./control $(MAGMA_CFLAGS) $^ -L./ -lmagma_pkg -o $@

clean: FORCE
\trm -f $(wildcard libmagma_pkg.so control/*.o src/*.o interface_{args.interface}/*.o)

FORCE:

.PHONY: default clean FORCE

"""


# Package for a given interface
if args.interface == 'cuda':

    README += f"""
## Interface (CUDA)

To build with CUDA, source files that end in `.cu` should be compiled with `nvcc`, i.e. the NVIDIA CUDA compiler. Given as makefile rules, you should have (approximately):

(keep in mind, throughout these examples, that some variables are just illustrative; you will have to define or supplement them with the relevant files/definitions in your build system)

```makefile

# rule to compile single object file
%.o: %.cu $(magma_H)
\t$(NVCC) -std=c++11 $< -Xcompiler "-fPIC" -o $@

```

And, to compile MAGMA into your own library (say `libmine.so`), you would modify your existing rule:

```makefile

# rule to compile your library (including MAGMA objects from this folder)
libmine.so: $(MAGMA_CU_O) $(MINE_C_O)
\t$(CC) $^ -lcublas -lcusparse -lcudart -lcudadevrt -shared -o $@

```

Assuming `MAGMA_CU_O` are the object files from MAGMA, and `MINE_C_O` are the object files from your library, this should link them together and create your shared library


"""
elif args.interface == 'hip':

    README += f"""
## Interface (HIP)

To build with HIP, source files that end in `.hip.cpp` should be compiled with `hipcc`, i.e. the HIP device compiler.

(keep in mind, throughout these examples, that some variables are just illustrative; you will have to define or supplement them with the relevant files/definitions in your build system)

```makefile

# rule to compile single object file
%.o: %.cu $(magma_H)
\t$(HIPCC) -DHAVE_HIP -std=c++11 -fno-gpu-rdc $< -fPIC -o $@

```

And, to compile MAGMA into your own library (say `libmine.so`), you would modify your existing rule:

```makefile

# rule to compile your library (including MAGMA objects from this folder)
libmine.so: $(MAGMA_HIP_O) $(MINE_C_O)
\t$(CC) $^ -lhipsparse -lhipblas -shared -o $@

```

Assuming `MAGMA_HIP_O` are the object files from MAGMA, and `MINE_C_O` are the object files from your library, this should link them together and create your shared library

"""


# -*- Output -*-

# make the output
try:
    os.makedirs(args.output)
except:
    pass

# write readme
with open(f"{args.output}/README.md", 'w') as fp:
    fp.write(README)

# write makefile
with open(f"{args.output}/Makefile", 'w') as fp:
    fp.write(MAKEFILE)

# write makefile
with open(f"{args.output}/test.c", 'w') as fp:
    fp.write(TEST_C)


# copy a file, creating destination folder
def copy(src, dst):
    try:
        os.makedirs(os.path.dirname(dst))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise
    shutil.copy(src, dst)

# copy in everything

for fl in set_c:
    copy(fl, f"{args.output}/{fl}")





