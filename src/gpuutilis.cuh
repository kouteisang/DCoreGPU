#pragma once 
#include "../header.h"


__device__ int warpReduceMin_balance_buffer_utils(int val);

__device__ int warp_reduce_sum_balance_buffer_utils(int val);

__global__ void reduceMinkernel_balance_buffer_utils(int* in_count_num, int* d_min, int num_vtx);


__global__ void hin_count_thread_buffer_utils(int* in_count_num, int* in_buffer_s, int* count_in_s, int* core, int* core0, int* in_adj, int* in_offset, int k);

__global__ void hin_count_warp_buffer_utils(int* in_count_num, int* in_buffer_m, int* count_in_m, int* core, int* core0, int* in_adj, int* in_offset, int k);

__global__ void hin_count_block_buffer_utils(int* in_count_num, int* in_buffer_l, int* count_in_l, int* core, int* core0, int* in_adj, int* in_offset, int k);

__global__ void vertex_to_buffer_by_core0_buffer_utils(int k, int* core0, int* in_degree, 
    int* in_buffer_s, int* in_buffer_m, int* in_buffer_l, 
    int* count_in_s, int* count_in_m, int* count_in_l, int* global_buffer, int* buf_count);

__global__ void update_upper_by_out_buffer_s_utils(int* out_buffer_s, int* count_out_s, int* hindex_in, int* hindex_out, int* core);

__global__ void update_upper_by_out_buffer_m_utils(int* out_buffer_m, int* count_out_m, int* hindex_in, int* hindex_out, int* core);

__global__ void update_upper_by_out_buffer_l_utils(int* out_buffer_l, int* count_out_l, int* hindex_in, int* hindex_out, int* core);

__global__ void update_change_status_out_thread_utils(int* out_buffer_s, int* count_out_s, int* hindex_in, int* hindex_out, int* core, int* out_adj, int* out_offset, int* core0, int k, int * global_done, int* visit);

__global__ void update_change_status_in_thread_utils(int* in_buffer_s, int* count_in_s, int* hindex_in, int* hindex_out, int* core, int* in_adj, int* in_offset, int* core0, int k, int * global_done, int* visit);

__global__ void update_change_status_out_warp_utils(int* out_buffer_m, int* count_out_m, int* hindex_in, int* hindex_out, int* core, int* out_adj, int* out_offset, int* core0, int k, int * global_done, int* visit);

__global__ void update_change_status_in_warp_utils(int* in_buffer_m, int* count_in_m, int* hindex_in, int* hindex_out, int* core, int* in_adj, int* in_offset, int* core0, int k, int * global_done, int* visit);

__global__ void update_change_status_out_block_utils(int* out_buffer_l, int* count_out_l, int* hindex_in, int* hindex_out, int* core, int* out_adj, int* out_offset, int* core0, int k, int * global_done, int* visit);

__global__ void update_change_status_in_block_utils(int* in_buffer_l, int* count_in_l, int* hindex_in, int* hindex_out, int* core, int* in_adj, int* in_offset, int* core0, int k, int * global_done, int* visit);

__global__ void vertex_to_buffer_by_out_degree_buffer_utils(int* visit, int num_vtx, int k, int* core, int* out_degree, 
    int* out_buffer_s, int* out_buffer_m, int* out_buffer_l, 
    int* count_out_s, int* count_out_m, int* count_out_l, int* global_buffer, int* buf_count);

__global__ void vertex_to_buffer_by_in_degree_buffer_utils(int* visit, int num_vtx, int k, int* core, int* in_degree, 
    int* in_buffer_s, int* in_buffer_m, int* in_buffer_l, 
    int* count_in_s, int* count_in_m, int* count_in_l, int* global_buffer, int* buf_count);


__device__ int warpReduceMin_utils(int val);

__global__ void klistprune_scan_utils(int* t_in_deg, int num_vtx, int* global_buffer, int* buf_count, int level);

__global__ void klistprune_update_utils(int* global_buffer, int* buf_count, int* global_count, int* t_in_deg, int* t_out_deg, int* out_offset, int *out_adj, int level);

__global__ void kstatus_update_utils(int* t_in_deg, bool* kstatus, int num_vtx);

__global__ void scan_block_utils(int* t_in_deg, int *t_out_deg, int* visit, int num_vtx, int* global_buffer, int* buf_count, int k, int l, int* core);


__global__ void check_innb_count_utils(int* in_count_num, int* core, int* core0, int num_vtx, int* in_offset, int* in_adj, int k);

__global__ void reduceMinkernel_utils(int* in_count_num, int* d_min, int num_vtx);

__global__ void scan_phase_balance_buffer_utils(int* t_in_deg, int *t_out_deg, int* visit, int num_vtx, int* global_buffer, int* buf_count, int k, int l, int* core);

__global__ void update_phase_balance_buffer_utils(int* global_buffer, int* buf_count, int* global_count, int* t_in_deg, int* in_adj, int* in_offset, int* t_out_deg, int* out_adj, int* out_offset, int* visit, int k, int l, int* core);

__global__ void update_visit_by_core0_balance_buffer_utils(int* core0, int* visit, int num_vtx, int k, int* core, int* in_count_number);

__global__ void vertex_to_buffer_buffer_utils(int num_vtx, int* global_buffer, int* buf_count, int* visit);