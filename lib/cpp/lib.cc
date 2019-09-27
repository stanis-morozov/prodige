#include <iostream>
#include <queue>
#include <vector>
#include <climits>
#include <bits/stdc++.h>
#include <tuple>
using namespace std;
#include "lib.h"

random_device                     rand_dev;
mt19937                           generator(rand_dev());
uniform_real_distribution<float>  distr(0.0, 1.0);
float SIGMOID_LOGIT_CUTOFF_VALUE = 10.0;
// ^-- if abs(logit) is this or higher, considers sigmoid(logit) to be exactly 0 or 1, used to save time

inline float softplus(float x) {
    if (x > 0)
        return x + log(1 + exp(-x));
    else
        return log(1 + exp(x));
}

inline bool sample_sigmoid_with_logit(float logit, bool deterministic) {
    if(deterministic) return logit > 0;
    if(logit > SIGMOID_LOGIT_CUTOFF_VALUE) return true;
    if(logit < -SIGMOID_LOGIT_CUTOFF_VALUE) return false;
    float tau = 1 / (1 + exp(-logit));
    float z = distr(generator);  // Uniform(0, 1)
    return z < tau;
}

void set_seed(int random_state) {
    srand(random_state);
}

// see documentation in __init__.py / batch_dijkstra

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
    )
{
    #pragma omp parallel for num_threads(*n_threads)
    for(int batch_i=0; batch_i < batch_size; batch_i++){
        dijkstra(
            num_vertices_plus1, slices,
            total_edges, sliced_edges,
            _total_edges, sliced_adjacency_logits,
            _total_edges2, sliced_weight_logits,
            initial_vertices[batch_i], num_targets, target_vertices + batch_i * num_targets,
            num_targets, max_length_target, target_paths + (batch_i * num_targets * max_length_target),
            k_nearest, max_length_nearest, nearest_paths + (batch_i * k_nearest * max_length_nearest),
            *deterministic, *soft, *max_steps
        );
    }
}

using HeapElement = tuple<int, float, int>;

class PrioritizeDijkstraHeap
{// compares elements in heap used by dijkstra (see below), tuples are (vertex, distance_to_v0, num_illegal_edges)
    public:
        bool operator()(HeapElement &a, HeapElement &b){
            if(get<2>(a) != get<2>(b))
                return get<2>(a) > get<2>(b);
            else
                return get<1>(a) > get<1>(b);
        }
};

void dijkstra(
    int num_vertices_plus1, int* slices,
    int total_edges, int* sliced_edges,
    int _total_edges, float* sliced_adjacency_logits,
    int _total_edges2, float* sliced_weight_logits,
    int initial_vertex, int num_targets, int* target_vertices,
    int _num_targets, int max_length_target, int* target_paths,
    int k_nearest, int max_length_nearest, int* nearest_paths,
    bool deterministic, bool soft, int max_steps
)
{
    int num_vertices = num_vertices_plus1 - 1;
    int steps = 0;

    // indicators whether target was found
    unordered_set<int> unfound_targets;       // targets that don't have shortest path
    unordered_set<int> undiscovered_targets;  // targets that don't have any path
    unordered_set<int> nearest_vertices;      // up to k_nearest nearest vertices
    for(int i = 0; i < num_targets; i++){
        undiscovered_targets.insert(target_vertices[i]);
        unfound_targets.insert(target_vertices[i]);
    }

    // distances to each vertex, predecessors of each vertex
    float distances[num_vertices];              // distance of "best" path to each vertex from initial_vertex
    int illegal_edge_counts[num_vertices];      // number of illegal edges along "best "path from initial_vertex
    int predecessors[num_vertices];             // previous vertex (index) along "best" path from initial vertex
    int predecessor_edge_indices[num_vertices]; // index of edge (in sliced_edges) to vertex from predecessors[vertex]

    for(int i = 0; i < num_vertices; i++)
    {
        distances[i] = numeric_limits<float>::infinity();
        illegal_edge_counts[i] = total_edges;
        predecessors[i] = -1;
        predecessor_edge_indices[i] = 0;
    }

    //  Priority queue to store (vertex, weight, num_illegal_edges) tuples
    priority_queue<HeapElement, vector<HeapElement>, PrioritizeDijkstraHeap> unscanned_heap;
    unscanned_heap.push(
        HeapElement(initial_vertex,
                    distances[initial_vertex]=0,
                    illegal_edge_counts[initial_vertex]=0)
    );

    while (
        (!unscanned_heap.empty()) && (                     // terminate if queue is empty
            (unfound_targets.size() != 0) ||               // or found all targets
            (nearest_vertices.size() < (size_t) k_nearest) // and got at least k neighbors
        )) {
        HeapElement current = unscanned_heap.top(); //Current vertex. The shortest distance for this has been found
        unscanned_heap.pop();
        int current_ix = get<0>(current);
        float current_distance = get<1>(current);
        int current_num_illegal = get<2>(current);

        if(current_distance > distances[current_ix])
            continue; // if we've already found a shorter path to this vertex before, there's no need to consider it
        if((current_num_illegal > 0) && (undiscovered_targets.size() == 0))
            break; // no more reachable vertices, but we've already found some path to target vertex
        if((nearest_vertices.size() < (size_t) k_nearest) && (current_ix != initial_vertex))
            nearest_vertices.insert(current_ix); // return this vertex as one of k nearest

        unfound_targets.erase(current_ix);

        for(int edge_i = slices[current_ix]; edge_i < slices[current_ix + 1]; edge_i++)
        {
            int adjacent_vertex = sliced_edges[edge_i];
            float edge_weight = softplus(sliced_weight_logits[edge_i]);
            bool edge_exists = sample_sigmoid_with_logit(sliced_adjacency_logits[edge_i], deterministic);

            // (in hard mode) discard if edge is not sampled
            if((!soft) && (!edge_exists)) continue;

            // discard if existing path is shorter (or same length)
            float new_distance = current_distance + edge_weight;
            if(new_distance >= distances[adjacent_vertex]) continue;

            // discard if existing path had strictly less illegal edges
            int new_num_illegal = current_num_illegal + (int)(!edge_exists);
            if(new_num_illegal > illegal_edge_counts[adjacent_vertex]) continue;

            // otherwise save new best path
            distances[adjacent_vertex] = new_distance;
            illegal_edge_counts[adjacent_vertex] = new_num_illegal;
            predecessors[adjacent_vertex] = current_ix;
            predecessor_edge_indices[adjacent_vertex] = edge_i;
            unscanned_heap.push(HeapElement(adjacent_vertex, new_distance, new_num_illegal));
            undiscovered_targets.erase(adjacent_vertex);
        }

        if((max_steps != -1) && (++steps > max_steps)) break;
   }

   // compute path to target
   for(int target_i = 0; target_i < num_targets; target_i++){
       if(predecessors[target_vertices[target_i]] == -1) continue;
       int vertex = target_vertices[target_i];
       for(int t = 0; t < max_length_target; t++){
            if(vertex == initial_vertex) break;
            target_paths[target_i * max_length_target + t] = predecessor_edge_indices[vertex];
            vertex = predecessors[vertex];
       }
   }

   // compute paths to k nearest vertices
   int neighbor_i = 0;
   for(auto vertex: nearest_vertices)
   {
       for(int t = 0; t < max_length_nearest; t++){
            if(vertex == initial_vertex) break;
            nearest_paths[neighbor_i * max_length_nearest + t] = predecessor_edge_indices[vertex];
            vertex = predecessors[vertex];
       }
       ++neighbor_i;
   }
}
