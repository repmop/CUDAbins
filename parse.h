#ifndef _PARSE_H
#define _PARSE_H
bool parse(char *infile);

bool dump(char *outfile);

/* Outputs to Dump */
extern uint32_t host_num_bins;
extern bin *bins_out;

#endif