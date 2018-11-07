#include "bin.h"
#include <json/json.h>
#include <jsoncpp.cpp>
#include <iostream>
#include <fstream>

using namespace std;

bool parse(char *infile) {
    ifstream f(infile, std::ifstream::binary);
    if (f.fail()) {
        return false;
    }
    Json::Value obj_data;
    f >> obj_data;
    cout << "bin_size: " << obj_data["bin_size"].asUInt() << endl;
    cout << "num_objs: " << obj_data["num_objs"].asUInt() << endl;
    return true;
}


void run() {
    return;
}

bool dump(char *outfile) {

    return true;
}
