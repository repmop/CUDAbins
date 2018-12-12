#ifndef _BIN_H_
#define _BIN_H_

#include <stdint.h>
#include <vector>

// #define TAGGING

#ifdef TAGGING

typedef struct obj {
    uint32_t size;
    uint32_t tag;
} obj_t;

#else

typedef struct obj {
    uint32_t size;
} obj_t;

#endif

typedef struct alias {
    int i1;
    int i2;
    float divider;
} alias_t;

typedef struct bin {
  uint32_t occupancy;
  std::vector<obj_t> obj_list;
  alias_t alias;
} bin_t;

bool parse(char *infile);

void run();
void runBFD();
void runNF();

bool dump(char *outfile);

/* Parsed Inputs */
extern obj *host_objs;
extern uint32_t host_num_objs;
extern uint32_t host_bin_size;
extern uint32_t host_total_obj_size; //pseudo-constant

#endif /* _BIN_H_ */
