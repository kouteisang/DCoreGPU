#ifndef UTILS_H
#define UTILS_H


#include "header.h"
#include "Graph/Graph.h"

/**
 * Init the data
 */
void malloc_graph_gpu_memory(Graph& g, G_pointers &p){
    
    int num_vtx = g.get_num_vtx();
    long long num_edge = g.get_num_edge();

    int* h_in_adj = g.get_h_in_adj();
    int* h_in_deg = g.get_h_in_deg();
    int* h_in_offset = g.get_h_in_offset();

    int* h_out_adj = g.get_h_out_adj();
    int* h_out_deg = g.get_h_out_deg();
    int* h_out_offset = g.get_h_out_offset();

    chkerr(cudaMalloc(&(p.in_adj), num_edge * sizeof(int)));
    chkerr(cudaMemcpy(p.in_adj, h_in_adj, num_edge * sizeof(int), cudaMemcpyHostToDevice));

    chkerr(cudaMalloc(&(p.out_adj), num_edge * sizeof(int)));
    chkerr(cudaMemcpy(p.out_adj, h_out_adj, num_edge * sizeof(int), cudaMemcpyHostToDevice));

    chkerr(cudaMalloc(&(p.in_deg), num_vtx * sizeof(int)));
    chkerr(cudaMemcpy(p.in_deg, h_in_deg, num_vtx * sizeof(int), cudaMemcpyHostToDevice));

    chkerr(cudaMalloc(&(p.out_deg), num_vtx * sizeof(int)));
    chkerr(cudaMemcpy(p.out_deg, h_out_deg, num_vtx * sizeof(int), cudaMemcpyHostToDevice));


    chkerr(cudaMalloc(&(p.in_offset), (num_vtx + 1) * sizeof(int)));
    chkerr(cudaMemcpy(p.in_offset, h_in_offset, (num_vtx + 1) * sizeof(int), cudaMemcpyHostToDevice));
 
    chkerr(cudaMalloc(&(p.out_offset), (num_vtx + 1) * sizeof(int)));     
    chkerr(cudaMemcpy(p.out_offset, h_out_offset, (num_vtx + 1) * sizeof(int), cudaMemcpyHostToDevice));

    chkerr(cudaMalloc(&(p.t_in_deg), num_vtx * sizeof(int)));
    chkerr(cudaMemcpy(p.t_in_deg, h_in_deg, num_vtx * sizeof(int), cudaMemcpyHostToDevice));

    chkerr(cudaMalloc((&p.t_out_deg), num_vtx * sizeof(int)));
    chkerr(cudaMemcpy(p.t_out_deg, h_out_deg, num_vtx * sizeof(int), cudaMemcpyHostToDevice));

    p.num_vtx = g.get_num_vtx();
}


/**
 * Copy in_deg -> t_in_deg
 * Copy out_deg -> t_out_deg
 * Memset flag
 */
void gpu_data_init(G_pointers &p){
    cudaMemcpy(p.t_in_deg, p.in_deg, p.num_vtx * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(p.t_out_deg, p.out_deg, p.num_vtx * sizeof(int), cudaMemcpyDeviceToDevice);

    cudaMemset(p.visit, false, p.num_vtx * sizeof(bool));
}

#endif