#ifndef UTILS_H
#define UTILS_H

#include <cuda_runtime.h>
#include "Graph/Graph.h"

void copyData2GPU(Graph& g, int* in_adj, int* in_deg, int* in_offset, int* out_adj, int* out_deg, int* out_offset){
    
    int num_vtx = g.get_num_vtx();
    long long num_edge = g.get_num_edge();

    int* h_in_adj = g.get_h_in_adj();
    int* h_in_deg = g.get_h_in_deg();
    int* h_in_offset = g.get_h_in_offset();

    int* h_out_adj = g.get_h_out_adj();
    int* h_out_deg = g.get_h_out_deg();
    int* h_out_offset = g.get_h_out_offset();

    cudaMalloc(&in_adj, num_edge * sizeof(int));
    cudaMemcpy(in_adj, h_in_adj, num_edge * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&out_adj, num_edge * sizeof(int));
    cudaMemcpy(out_adj, h_out_adj, num_edge * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&in_deg, num_vtx * sizeof(int));
    cudaMemcpy(in_deg, h_in_deg, num_vtx * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&out_deg, num_vtx * sizeof(int));
    cudaMemcpy(out_deg, h_out_deg, num_vtx * sizeof(int), cudaMemcpyHostToDevice);


    cudaMalloc(&in_offset, (num_vtx + 1) * sizeof(int));
    cudaMemcpy(in_offset, h_in_offset, (num_vtx + 1) * sizeof(int), cudaMemcpyHostToDevice);
 
    cudaMalloc(&out_offset, (num_vtx + 1) * sizeof(int));     
    cudaMemcpy(out_offset, h_out_offset, (num_vtx + 1) * sizeof(int), cudaMemcpyHostToDevice);
 

}

#endif