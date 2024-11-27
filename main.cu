#include "utils.h"

int main(int argc, char* argv[]){

    string dataset = "example";
    string file_path = "/home/cheng/DCoreGPU/dataset/"+ dataset + "/";

    Graph g = Graph(file_path, dataset);

    

    int* in_adj;
    int* in_deg;
    int* in_offset;

    int* out_adj;
    int* out_deg;
    int* out_offset;
    copyData2GPU(g, in_adj, in_deg, in_offset, out_adj, out_deg, out_offset);

    // cout << "num_vtx = " << num_vtx << endl;
    // cout << "num_edge = " << num_edge << endl;

    // for(uint i = 0; i < num_vtx; i ++){
    //     cout << "i = " << i << " in deg = " << h_in_deg[i] << endl;
    // }
    // for(uint i = 0; i < num_vtx; i ++){
    //     cout << "i = " << i << " out deg = " << h_out_deg[i] << endl;
    // }

    // for(long long e = 0; e < num_edge; e ++){
    //     cout << h_out_adj[e] << " ";
    // }
    // cout << endl;

    // for(long long e = 0; e < num_edge; e ++){
    //     cout << h_in_adj[e] << " ";
    // }
    // cout << endl;

    // for(uint v = 0; v <= num_vtx; v ++){
    //     cout << h_in_offset[v] << " ";
    // }
    // cout << endl;

    // for(uint v = 0; v <= num_vtx; v ++){
    //     cout << h_out_offset[v] << " ";
    // }
    // cout << endl;
     



}