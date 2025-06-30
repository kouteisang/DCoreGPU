
#include "Graph.h"

struct node{
    int id;
    int indegree;
    int outdegree;
};

Graph::Graph(/* args */){
}

Graph::~Graph(){

    delete[] h_in_deg;
    delete[] h_in_adj;
    delete[] h_in_offset;
    delete[] h_out_deg;
    delete[] h_out_adj;
    delete[] h_out_offset;

}

Graph::Graph(const std::string& bin_file) {
    std::cout << bin_file << std::endl;    
    std::cout << "Read graph binary file" << std::endl;
    FILE* f = fopen(bin_file.c_str(), "rb");
    assert(f);

    fread(&num_vtx, sizeof(int), 1, f);
    fread(&num_edge, sizeof(long long), 1, f);

    h_in_deg = new int[num_vtx];
    h_out_deg = new int[num_vtx];

    h_in_offset = new int[num_vtx + 1];
    h_out_offset = new int[num_vtx + 1];

    h_in_adj = new int[num_edge];
    h_out_adj = new int[num_edge];

    fread(h_in_deg, sizeof(int), num_vtx, f);
    fread(h_out_deg, sizeof(int), num_vtx, f);

    fread(h_in_offset, sizeof(int), num_vtx + 1, f);
    fread(h_out_offset, sizeof(int), num_vtx + 1, f);

    fread(h_in_adj, sizeof(int), num_edge, f);
    fread(h_out_adj, sizeof(int), num_edge, f);

    fclose(f);
}


void Graph::SaveToBinary(const std::string& bin_file) {
    FILE* f = fopen(bin_file.c_str(), "wb");
    assert(f);

    fwrite(&num_vtx, sizeof(int), 1, f);
    fwrite(&num_edge, sizeof(long long), 1, f);

    fwrite(h_in_deg, sizeof(int), num_vtx, f);
    fwrite(h_out_deg, sizeof(int), num_vtx, f);

    fwrite(h_in_offset, sizeof(int), num_vtx + 1, f);
    fwrite(h_out_offset, sizeof(int), num_vtx + 1, f);

    fwrite(h_in_adj, sizeof(int), num_edge, f);
    fwrite(h_out_adj, sizeof(int), num_edge, f);

    fclose(f);
}


void Graph::GetVtxMapping(string file_path, string map_file, std::basic_ofstream<char> &map_file_out){

    int nn;
    int uid, vid;

    FILE* Dfile = fopen(file_path.c_str(), "r");
      
    map_file_out = ofstream(map_file);

    fscanf(Dfile, "%d%lld", &nn, &num_edge);

    for(ll i = 0; i < num_edge; i ++){
        int u, v;
        fscanf(Dfile, "%d%d", &u, &v);

        auto iter1 = vtx2id.find(u);
        if (iter1 != vtx2id.end()) {
            uid = iter1->second;
        } else {
            uid = num_vtx++;
            vtx2id.emplace(u, uid);
            map_file_out << u << " " << uid << endl;
        }

        auto iter2 = vtx2id.find(v);
        if (iter2 != vtx2id.end()) {
            vid = iter2->second;
        } else {
            vid = num_vtx++;
            vtx2id.emplace(v, vid);
            map_file_out << v << " " << vid << endl;
        }
    }


}

Graph::Graph(string file, string dataset){

    std::cout << "Construct graph" << std::endl;
    int nn;
    int uid, vid;

    string file_path = file + dataset + ".txt";
    string map_file = file + "vtx2id.txt";

    cout << "file_path = " << file_path << endl;
    cout << "map_file = " << map_file << endl;

    std::basic_ofstream<char> map_file_out;

    GetVtxMapping(file_path, map_file, map_file_out);

    num_vtx = vtx2id.size();

    vector<int> vec_in_deg(num_vtx);
    vector<vector<int>> vec_in_adj(num_vtx);

    vector<int> vec_out_deg(num_vtx);
    vector<vector<int>> vec_out_adj(num_vtx);

    // cout << "num_vtx = " << num_vtx << endl; 

    FILE* Dfile = fopen(file_path.c_str(), "r");

    fscanf(Dfile, "%d%lld", &nn, &num_edge);

    // cout << "nn = " << nn << " num_edge = " << num_edge << endl;

    for(ll i = 0; i < num_edge; i ++){
        int u, v;
        fscanf(Dfile, "%d%d", &u, &v);

        auto iter1 = vtx2id.find(u);
        if (iter1 != vtx2id.end()) {
            uid = iter1->second;
        }

        auto iter2 = vtx2id.find(v);
        if (iter2 != vtx2id.end()) {
            vid = iter2->second;
        }

        vec_out_deg[uid] ++;
        vec_out_adj[uid].push_back(vid);

        vec_in_deg[vid] ++;
        vec_in_adj[vid].push_back(uid);
    }

    h_out_deg = new int[num_vtx];
    h_in_deg = new int[num_vtx];


    copy(vec_out_deg.begin(), vec_out_deg.end(), h_out_deg);
    copy(vec_in_deg.begin(), vec_in_deg.end(), h_in_deg);

    h_in_offset = new int[num_vtx+1];
    h_out_offset = new int[num_vtx+1];

    h_in_offset[0] = 0;
    h_out_offset[0] = 0;
 
    for(uint v = 0; v < num_vtx; v ++){
        if(!vec_out_adj[v].empty()) sort(vec_out_adj[v].begin(), vec_out_adj[v].end());
        if(!vec_in_adj[v].empty()) sort(vec_in_adj[v].begin(), vec_in_adj[v].end());
        
        h_in_offset[v+1] = h_in_offset[v] + h_in_deg[v];
        h_out_offset[v+1] = h_out_offset[v] + h_out_deg[v];
    }
    
    h_in_adj = new int[num_edge];
    h_out_adj = new int[num_edge];

    long long e = 0;

    for(uint v = 0; v < num_vtx; v ++){
        for(uint i = 0; i < h_in_deg[v]; i ++){
            auto u = vec_in_adj[v][i];
            h_in_adj[e ++] = u;
        }
    }

    e = 0;
    for(uint v = 0; v < num_vtx; v ++){
        for(uint i = 0; i < h_out_deg[v]; i ++){
            auto u = vec_out_adj[v][i];
            h_out_adj[e ++] = u;
        }
    }

    SaveToBinary(file + dataset + "-0.bin");

//    std::ofstream wr("/home/cheng/DCoreGPU/dataset/enwiki-2024/degree.txt");

//     for(uint v = 0; v < num_vtx; v ++){
//         wr << h_in_deg[v] << " " << h_out_deg[v] << endl;
//     }

}


Graph::Graph(string file, string dataset, int order){
    
    std::cout << "Construct graph" << std::endl;
    int nn;
    int uid, vid;

    string file_path = file + dataset + ".txt";
    string map_file = file + "vtx2id.txt";

    cout << "file_path = " << file_path << endl;
    cout << "map_file = " << map_file << endl;

    std::basic_ofstream<char> map_file_out;

    GetVtxMapping(file_path, map_file, map_file_out);

    num_vtx = vtx2id.size();

    vector<int> vec_in_deg(num_vtx);
    vector<vector<int>> vec_in_adj(num_vtx);

    vector<int> vec_out_deg(num_vtx);
    vector<vector<int>> vec_out_adj(num_vtx);


    FILE* Dfile = fopen(file_path.c_str(), "r");

    fscanf(Dfile, "%d%lld", &nn, &num_edge);


    for(ll i = 0; i < num_edge; i ++){
        int u, v;
        fscanf(Dfile, "%d%d", &u, &v);

        auto iter1 = vtx2id.find(u);
        if (iter1 != vtx2id.end()) {
            uid = iter1->second;
        }

        auto iter2 = vtx2id.find(v);
        if (iter2 != vtx2id.end()) {
            vid = iter2->second;
        }

        vec_out_deg[uid] ++;
        vec_out_adj[uid].push_back(vid);

        vec_in_deg[vid] ++;
        vec_in_adj[vid].push_back(uid);
    }

    node* nodes = new node[num_vtx];

    for(int i = 0; i < num_vtx; i ++){
        nodes[i].id = i;
        nodes[i].indegree = vec_in_deg[i];
        nodes[i].outdegree = vec_out_deg[i];
    }

    if(order == 1){
        std::sort(nodes, nodes + num_vtx, [](const node& a, const node& b) {
            if (a.outdegree != b.outdegree) {
                return a.outdegree > b.outdegree; // primary: outdegree descending
            } else {
                return a.indegree > b.indegree;   // secondary: indegree descending
            }
        });
    }else if(order == 2){
        std::sort(nodes, nodes + num_vtx, [](const node& a, const node& b) {
            if( a.indegree != b.indegree){
                return a.indegree > b.indegree;
            }else{
                return a.outdegree > b.outdegree;
            }
        });
    }

    int *order_vertex_map = new int[num_vtx];
    for(int vid = 0; vid < num_vtx; vid ++){
        int v = nodes[vid].id;
        order_vertex_map[v] = vid;
    }

    for(int v = 0; v < num_vtx; v ++){
        for(int i = 0; i < vec_out_adj[v].size(); i ++){
            int u = vec_out_adj[v][i];
            vec_out_adj[v][i] = order_vertex_map[u];
        }
    }
    
    for(int v = 0; v < num_vtx; v ++){
        for(int i = 0; i < vec_in_adj[v].size(); i ++){
            int u = vec_in_adj[v][i];
            vec_in_adj[v][i] = order_vertex_map[u];
        }
    }

    h_out_deg = new int[num_vtx];
    h_in_deg = new int[num_vtx];

    for(int vid = 0; vid < num_vtx; vid ++){
        h_out_deg[vid] = nodes[vid].outdegree;
        h_in_deg[vid] = nodes[vid].indegree;
    }


    h_in_offset = new int[num_vtx+1];
    h_out_offset = new int[num_vtx+1];
    h_in_offset[0] = 0;
    h_out_offset[0] = 0;

    for(uint vid = 0; vid < num_vtx; vid ++){
        int v = nodes[vid].id;

        if(!vec_out_adj[v].empty()) sort(vec_out_adj[v].begin(), vec_out_adj[v].end());
        if(!vec_in_adj[v].empty()) sort(vec_in_adj[v].begin(), vec_in_adj[v].end());

        h_in_offset[vid+1] = h_in_offset[vid] + h_in_deg[vid];
        h_out_offset[vid+1] = h_out_offset[vid] + h_out_deg[vid];
    }


    h_in_adj = new int[num_edge];
    h_out_adj = new int[num_edge];
    long long e = 0;

    for(uint vid = 0; vid < num_vtx; vid ++){
        int v = nodes[vid].id; 
        for(uint i = 0; i < vec_in_adj[v].size(); i ++){
            auto u = vec_in_adj[v][i];
            h_in_adj[e ++] = u;
        }
    }


    e = 0;
    for(uint vid = 0; vid < num_vtx; vid ++){
        int v = nodes[vid].id; 
        for(uint i = 0; i < vec_out_adj[v].size(); i ++){
            auto u = vec_out_adj[v][i];
            h_out_adj[e ++] = u;
        }
    }

    SaveToBinary(file + dataset + "-" + std::to_string(order) + ".bin");


    


    

}