#include <stdint.h>
#include <vector>

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

typedef struct bin {
  uint32_t occupancy;
  uint32_t capacity;
  std::vector<obj_t> *obj_list;
} bin_t;

void parse(FILE *infile);

void run();

void dump(FILE *outfile);
