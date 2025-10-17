#include "utils.h"
#include "./src/klist.cuh"
#include "./src/klistprune.cuh"
#include "./src/klistanchorbinary.cuh"
#include "./src/klistanchorbinaryprune.cuh"
#include "./src/klistanchorsequenceprune.cuh"
#include "./src/gpubaseline.cuh"
#include "./src/klist_balance.cuh"
#include "./src/klist_balance_buffer.cuh"
#include "./src/klist_balance_buffer_one_stream.cuh"
#include "./src/klist_thread.cuh"
#include "./src/klist_block.cuh"
#include "./src/klist_balance_buffer_atomic.cuh"
#include "./src/observation1.cuh"
#include "./src/observation2.cuh"
#include "./src/observation3.cuh"
#include "./src/ghthread.cuh"
#include "./src/ghblock.cuh"
#include "./src/ghmultiblockfrontier.cuh"
#include "./src/hindex_gpu_baseline.cuh"

enum Algorithm{
    klist = 1,
    klistprune = 2,
    klistanchorbinary = 3,
    klistanchorbinaryprune = 4,
    klistanchorsequenceprune = 5,
    klist_balance = 6,
    klist_balance_buffer = 7,
    klist_balance_buffer_one_stream = 8,
    gpubaseline = 9, 
    klistthread = 10,
    klistblock = 11,
    klist_balance_buffer_with_atomic = 12,
    ablation_observe_1 = 13,
    ablation_observe_2 = 14,
    ablation_observe_3 = 15,
    gh_thread = 16,
    gh_block = 17,
    gh_multiblockfrontier = 18,
    hindex_gpu_baseline = 19,
};

bool file_exists(const std::string& name) {
    std::ifstream f(name);
    return f.good();
}

int main(int argc, char* argv[]){

    cudaSetDevice(1);

    string dataset = "em";
    int t = 0;
    int alg = 1; // klist, klist-prune
    int order = 0; // 0: randoem, 1: sort by out-degree, 2: sort by in-degree

    for(int i = 1; i < argc; i ++){
        string arg = argv[i];
        if (arg == "-d" && i + 1 < argc) {
            dataset = argv[++i];
        }else if(arg == "-a" && i+1 < argc){
            alg = std::stoi(argv[++i]);
        }else if(arg == "-o" && i+1 < argc){
            order = std::stoi(argv[++i]);
        }else if(arg == "-t" && i+1 < argc){
            t = std::stoi(argv[++i]);
        }
    }

    cout << "dataset = " << dataset << endl;

    cudaEvent_t start, stop; // Calculate time
    cudaEventCreate(&start); // Calculate time
    cudaEventCreate(&stop);  // Calculate time


    string file_path = "DCoreGPU/dataset/"+ dataset + "/";

    std::string bin_file = "DCoreGPU/dataset/"+ dataset + "/" + dataset+  "-" +std::to_string(order) + ".bin";
    // std::string bin_file = "DCoreGPU/dataset/"+ dataset + "/" + dataset + ".bin";


    Graph g = file_exists(bin_file) ?
          Graph(bin_file) :
          Graph(file_path, dataset, order);

    // Graph g = Graph(file_path, dataset);
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
        case Algorithm::klistanchorsequenceprune:
            cout << "Algorithm = klistanchorsequenceprune" << endl;
            klistanchorsequenceprune_de(data_pointers);
            break;
        case Algorithm::klist_balance:
            cout << "Algorithm = klist balanced" << endl;
            klist_balance_de(data_pointers);
            break;
        case Algorithm::klist_balance_buffer:
            cout << "Algorithm = klist balanced buffer" << endl;
            klist_balance_buffer_de(data_pointers, t);
            break;
        case Algorithm::klist_balance_buffer_one_stream:
            cout << "Algorithm = klist balanced buffer one stream" << endl;
            klist_balance_buffer_de_one_stream(data_pointers);
            break;
        case Algorithm::gpubaseline:
            cout << "Algorithm = gpubaseline" << endl;
            gpu_baseline_de(data_pointers);
            break;
        case Algorithm::klistthread:
            cout << "Algorithm = klistthread" << endl;
            // klistprune_de(data_pointers);
            klist_thread(data_pointers);
            break;
        case Algorithm::klistblock:
            cout << "Algorithm = klistblock" << endl;
            klist_block(data_pointers);
            break;
        case Algorithm::klist_balance_buffer_with_atomic:
            cout << "Algorithm = klist balance buffer with atomic" << endl;
            klist_balance_buffer_atomic_de(data_pointers);
            break;
        case Algorithm::ablation_observe_1:
            cout << "Algorithm = ablation_observe_1" << endl;
            klist_observation1(data_pointers);
            break;
        case Algorithm::ablation_observe_2:
            cout << "Algorithm = ablation_observe_2" << endl;
            klist_observation2(data_pointers);
            break;
        case Algorithm::ablation_observe_3:
            cout << "Algorithm = ablation_observe_3" << endl;
            klist_observation3(data_pointers);
            break;
        case Algorithm::gh_thread:
            cout << "Algorithm = gh_thread" << endl;
            ghthread_decomposition(data_pointers);
            break;
        case Algorithm::gh_block:
            cout << "Algorithm = gh_block" << endl;
            ghblock_decomposition(data_pointers);
            break;
        case Algorithm::gh_multiblockfrontier:
            cout << "Algorithm = gh_multiblockfrontier" << endl;
            ghmultiblockfrontier_decomposition(data_pointers, t);
            break;
        case Algorithm::hindex_gpu_baseline:
            cout << "Algorithm = hindex GPU baseline" << endl;
            hindex_baseline(data_pointers, g);
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
    std::cout << "GPU time = " << gpu_time*1.0/1000 << " second" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}