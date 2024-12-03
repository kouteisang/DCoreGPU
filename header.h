#ifndef header_h
#define header_h

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <fstream>
#include <algorithm>
#include <execution>
#include <cuda_runtime.h>



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
// #define BLK_DIM 256
#define BUFFER_SIZE 1000000
// #define BUFFER_SIZE 100
#define WARP_SIZE 32

typedef struct G_pointers {
    int* in_adj;
    int* in_deg;
    int* in_offset;

    int* out_adj;
    int* out_deg;
    int* out_offset;

    int* t_in_deg; // We use it for each iteration
    int* t_out_deg; // Weuse it for each iteration

    bool* flag;

    int num_vtx;
} G_pointers;//graph related

/**
 * Error check
 */
inline void chkerr(cudaError_t code){
    if (code != cudaSuccess){
        std::cout << cudaGetErrorString(code) << std::endl;
        exit(-1);
    }
}


#endif
