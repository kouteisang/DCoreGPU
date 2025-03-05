#include "utils.h"
#include "./src/klist.cuh"
#include "./src/klistprune.cuh"
#include "./src/klistanchorbinary.cuh"
#include "./src/klistanchorbinaryprune.cuh"

enum Algorithm{
    klist = 1,
    klistprune = 2,
    klistanchorbinary = 3,
    klistanchorbinaryprune = 4,
};

int main(int argc, char* argv[]){

    cudaSetDevice(1);

    string dataset = "em";
    int alg = 1; // klist, klist-prune

    for(int i = 1; i < argc; i ++){
        string arg = argv[i];
        if (arg == "-d" && i + 1 < argc) {
            dataset = argv[++i];
        }else if(arg == "-a" && i+1 < argc){
            alg = std::stoi(argv[++i]);
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

    switch (alg){
        case Algorithm::klist:
            cout << "Algorithm = klist" << endl;
            klist_de(data_pointers);
            break;
        case Algorithm::klistprune:
            cout << "Algorithm = klistprune" << endl;
            klistprune_de(data_pointers);
            break;
        case Algorithm::klistanchorbinary:
            cout << "Algorithm = klistanchorbinary" << endl;
            klistanchorbinary_de(data_pointers);
            break;
        case Algorithm::klistanchorbinaryprune:
            cout << "Algorithm = klistanchorbinaryprune" << endl;
            klistanchorbinaryprune_de(data_pointers);
            break;
        default:
            cout << "Algorithm = klist" << endl;
            klist_de(data_pointers);
            break;
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start, stop);
    std::cout << "GPU time = " << gpu_time << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}