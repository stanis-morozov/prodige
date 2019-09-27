%module bindings

%{
    #define SWIG_FILE_WITH_INIT
    #include "lib.h"
%}

%include "numpy.i"
%include "typemaps.i"

%init %{
    import_array();
%}


%apply (int DIM1, int* IN_ARRAY1) {
    (int num_vertices_plus1, int* slices),
    (int total_edges, int* sliced_edges)
}

%apply (int DIM1, float* IN_ARRAY1) {
    (int _total_edges, float* sliced_adjacency_logits),
    (int _total_edges2, float* sliced_weight_logits)
}

%apply (int DIM1, int* IN_ARRAY1) {(int batch_size, int* initial_vertices)}
%apply (int DIM1, int DIM2, int* IN_ARRAY2) {(int _batch_size, int num_targets, int* target_vertices)}

%apply (int DIM1, int DIM2, int DIM3, int* IN_ARRAY3) {
    (int _batch_size2, int _num_targets, int max_length_target, int* target_paths),
    (int _batch_size3, int k_nearest, int max_length_nearest, int* nearest_paths)
}

%apply int *INPUT {bool* deterministic, bool *soft}
%apply int *INPUT {int* random_state, int *n_threads, int *max_steps}

%include "lib.h"
