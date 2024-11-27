#ifndef GRAPH_H
#define GRAPH_H

#include "../header.h"

class Graph
{
private:
    int num_vtx{0}; // number of verticesl
    long long num_edge{0}; // number of edge;

    int *h_in_deg{nullptr}; // vertice in degree
    int *h_in_adj{nullptr}; // vertice in adj
    int *h_in_offset{nullptr}; // vertice in offset

    int *h_out_deg{nullptr}; // vertice out degree
    int *h_out_adj{nullptr}; // vertice out adj
    int *h_out_offset{nullptr}; // vertice out offset;

   

    unordered_map<int, int> vtx2id;


public:
    Graph(/* args */);
    Graph(string file, string dataset);
    ~Graph();

    void GetVtxMapping(string file_path, string map_file, std::basic_ofstream<char> &map_file_out);

    int get_num_vtx(){
        return num_vtx;
    }

    long long get_num_edge(){
        return num_edge;
    }

    int* get_h_in_deg(){
        return h_in_deg;
    }
    
    int* get_h_out_deg(){
        return h_out_deg;
    }
    
    int* get_h_in_adj(){
        return h_in_adj;
    }
    
    int* get_h_out_adj(){
        return h_out_adj;
    }
    
    int* get_h_in_offset(){
        return h_in_offset;
    }
    
    int* get_h_out_offset(){
        return h_out_offset;
    }

};


#endif