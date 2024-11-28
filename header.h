#ifndef header_h
#define header_h

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <fstream>
#include <algorithm>
#include <execution>

using std::min;
using std::copy;
using std::sort;
using std::ofstream;
using std::unordered_map;
using std::vector;
using std::string;
using std::cout;
using std::endl;

typedef long long ll;


#define BLK_NUMS 56
#define BLK_DIM 1024

typedef struct G_pointers {
    int* in_adj;
    int* in_deg;
    int* in_offset;

    int* out_adj;
    int* out_deg;
    int* out_offset;

    int* t_in_deg;
    int* t_out_deg;

    bool* flag;

    int num_vtx;
} G_pointers;//graph related

#endif
