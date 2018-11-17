/*
 * CLI for bin-packing optimizer
 * Takes an input file in the format specified in README.md
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <iostream>
#include <unistd.h>

#include "bin.h"


void usage() {
  fprintf(stderr, "-f /path/to/input [-o /path/to/output]\n");
}

int main(int argc, char *argv[]) {
  char *infile = NULL;
  char *outfile = NULL;
  bool bfd_flag = false;
  int opt;
  opterr = 0;

  while ((opt = getopt(argc, argv, "f:o:b")) != -1) {
    switch (opt) {
      case 'f':
        infile = optarg;
        break;
      case 'o':
        outfile = optarg;
        break;
      case 'b':
        bfd_flag = true;
        break;
      case '?':
        if (optopt == 'f') {
          fprintf(stderr, "Option -%c requires an argument.\n", optopt);
        } else if (optopt == 'o') {
          fprintf(stderr, "Option -%c requires an argument.\n", optopt);
        } else if (isprint (optopt)) {
          fprintf(stderr, "Unknown option `-%c'.\n", optopt);
        } else {
          fprintf(stderr, "Unknown option character `\\x%x'.\n", optopt);
          usage();
          return 1;
        }
      default:
        fprintf(stderr, "Malformed input arguments. Aborting.\n");
        usage();
        return 1;
    }
  }
  if (infile==NULL) {
    fprintf(stderr, "Input file unspecified\n");
    usage();
    return 1;
  }
  if (!parse(infile)) {
   fprintf(stderr, "Failed to parse input\n");
  }

  if(!bfd_flag){
    run();
  } else {
    runBFD();
  }

  if (!dump(outfile)) {
      fprintf(stderr, "Failed to write output\n");
  }

  return 0;
}
