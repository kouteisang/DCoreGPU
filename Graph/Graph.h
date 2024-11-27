#ifndef GRAPH_H
#define GRAPH_H

#include "../header.h"

class Graph
{
private:
    int n{0}; // number of verticesl
    int m{0}; // number of edge;

    int *in_deg{nullptr}; // vertice in degree
    int *in_adj{nullptr}; // vertice in adj
    int *in_offset{nullptr}; // vertice in offset

    int *out_deg{nullptr}; // vertice out degree
    int *out_adj{nullptr}; // vertice out adj
    int *out_offset{nullptr}; // vertice out offset;

    vector<int> v_in_deg;
    vector<vector<int>> v_in_adj;

    vector<int> v_out_deg;
    vector<vector<int>> v_out_adj;


public:
    Graph(/* args */);
    Graph(string file_path);
    ~Graph();
};


#endif