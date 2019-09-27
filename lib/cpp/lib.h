#include <vector>
#include <cstdio>
#include <queue>
#include <iostream>
#include <assert.h>
#include <omp.h>
#include <unordered_set>


void set_seed(int random_state);

void batch_dijkstra(
    int num_vertices_plus1, int* slices,
    int total_edges, int* sliced_edges,
    int _total_edges, float* sliced_adjacency_logits,
    int _total_edges2, float* sliced_weight_logits,
    int batch_size, int* initial_vertices,
    int _batch_size, int num_targets, int* target_vertices,
    int _batch_size2, int _num_targets, int max_length_target, int* target_paths,
    int _batch_size3, int k_nearest, int max_length_nearest, int* nearest_paths,
    bool *deterministic, bool *soft, int *max_steps, int *n_threads
);

void dijkstra(
    int num_vertices_plus1, int* slices,
    int total_edges, int* sliced_edges,
    int _total_edges, float* sliced_adjacency_logits,
    int _total_edges2, float* sliced_weight_logits,
    int initial_vertex, int num_targets, int* target_vertices,
    int _num_targets, int max_length_target, int* target_paths,
    int k_nearest, int max_length_nearest, int* nearest_paths,
    bool deterministic, bool soft, int max_steps
);
