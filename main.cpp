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
  FILE *outfile_io = stdout;
  FILE *infile_io = stdin;
  int opt;
  opterr = 0;

  while ((opt = getopt (argc, argv, "f:o:")) != -1)
  switch (opt)
  {
    case 'f':
      infile = optarg;
      break;
    case 'o':
      outfile = optarg;
      break;
    case '?':
      if (optopt == 'f') {
        fprintf (stderr, "Option -%c requires an argument.\n", optopt);
      } else if (optopt == 'o') {
        fprintf (stderr, "Option -%c requires an argument.\n", optopt);
      } else if (isprint (optopt)) {
        fprintf (stderr, "Unknown option `-%c'.\n", optopt);
      } else {
        fprintf (stderr, "Unknown option character `\\x%x'.\n", optopt);
        usage();
        return 1;
      }
    default:
      fprintf(stderr, "Malformed input arguments. Aborting.\n");
      usage();
      return 1;
  }
  //TODO: check infile exists, outfile does not have stuff already
  if (infile==NULL) {
    fprintf(stderr, "Input file unspecified\n");
    usage();
    return 1;
  }
  if (outfile!=NULL) {
      outfile_io = fopen(outfile, "rw");
      fseek(outfile_io, 0, SEEK_END); // goto end of file
      if (ftell(outfile_io) != 0) {
        outfile_io = stdout;
      }
      fseek(outfile_io, 0, SEEK_SET);
  }
  infile_io = fopen(infile, "r");
  parse(infile_io);

  run();

  dump(outfile_io);

  return 0;
}
