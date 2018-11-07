#include <stdint.h>
#include <vector>

#ifdef TAGGING

// #ifndef _BIN_H_
// #define _BIN_H_

typedef struct obj {
    uint32_t size;
    uint32_t tag;
} obj_t;

#else

typedef struct obj {
    uint32_t size;
} obj_t;

#endif

typedef struct bin {
  uint32_t occupancy;
  uint32_t capacity;
  std::vector<obj_t> *obj_list;
} bin_t;

bool parse(char *infile);

void run();

bool dump(char *outfile);
