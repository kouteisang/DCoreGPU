#include "Graph.h"

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

    cout << "nn = " << nn << " num_edge = " << num_edge << endl;

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

    // for(uint v = 0; v < nn; v ++){
    //     cout << "v =" << vtx2id[v] << " in = " << h_in_deg[v] << " out = " << h_out_deg[v] << endl;
    // }

}
