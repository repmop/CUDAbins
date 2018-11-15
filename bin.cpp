#include "bin.h"
#include <json/json.h>
#include <jsoncpp.cpp>
#include <iostream>
#include <fstream>

using namespace std;

int num_objs, bin_size, num_bins;
obj_t *objs;
bin_t *bins;
bool parse(char *infile) {
    ifstream f(infile, std::ifstream::binary);
    if (f.fail()) {
        return false;
    }
    Json::Value obj_data;
    f >> obj_data;
    cout << obj_data;
    bin_size = obj_data["bin_size"].asUInt();
    num_objs = obj_data["num_objs"].asUInt();
    objs = new obj_t[num_objs];
    for (int i = 0; i < num_objs; i++) {
        objs[i].size = obj_data["objs"][i].asUInt();
        objs[i].tag = i;
    }
    cout << endl;
    return true;
}


void run() {
    num_bins = num_objs;
    bins = new bin_t[num_objs]; // worst case # bins == # objs
    return;
}

bool dump(char *outfile) {
    Json::Value obj_data;
    obj_data["bin_size"] = bin_size;
    obj_data["num_objs"] = num_objs;
    obj_data["num_bins"] = num_bins;
    if (outfile==NULL) { //print results to stdout
    } else { //print to file

    }
    return true;
}
