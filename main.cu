#include "utils.h"
#include "./src/klist.cuh"


int main(int argc, char* argv[]){

    string dataset = "em";

    for(int i = 1; i < argc; i ++){
        string arg = argv[i];
        if (arg == "-d" && i + 1 < argc) {
            dataset = argv[++i];
        }
    }

    cout << "dataset = " << dataset << endl;

    

    cudaEvent_t start, stop; // Calculate time
    cudaEventCreate(&start); // Calculate time
    cudaEventCreate(&stop);  // Calculate time


    string file_path = "/home/cheng/DCoreGPU/dataset/"+ dataset + "/";

    Graph g = Graph(file_path, dataset);

    int num_vtx = g.get_num_vtx();
    ll num_edge = g.get_num_edge();

    G_pointers data_pointers;
    
    malloc_graph_gpu_memory(g, data_pointers);

    gpu_data_init(data_pointers);

    cudaEventRecord(start, 0);
    
    klist_de(data_pointers);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start, stop);
    std::cout << "GPU time = " << gpu_time/1000 << " s" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}