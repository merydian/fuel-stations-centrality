#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <limits>
#include <omp.h>   // OpenMP for parallelism

namespace py = pybind11;

// -----------------------------------------------------------------------------
// Graph-level straightness centrality (single double result)
// -----------------------------------------------------------------------------
double compute_graph_straightness_core(
    const std::vector<double>& coords_x,
    const std::vector<double>& coords_y,
    const std::vector<double>& shortest_paths, // flattened n×n row-major
    int n)
{
    double global_num = 0.0;
    long long global_den = 0;

    #pragma omp parallel
    {
        double local_num = 0.0;
        long long local_den = 0;

        #pragma omp for schedule(dynamic)
        for (int i = 0; i < n; i++) {
            double xi = coords_x[i];
            double yi = coords_y[i];

            for (int j = 0; j < n; j++) {
                if (i == j) continue;

                double d_g = shortest_paths[i * n + j];
                if (d_g == std::numeric_limits<double>::infinity())
                    continue;

                double dx = xi - coords_x[j];
                double dy = yi - coords_y[j];
                double d_e = std::sqrt(dx * dx + dy * dy);

                if (d_e > 0.0) {
                    local_num += d_e / d_g;
                    local_den += 1;
                }
            }
        }

        #pragma omp atomic
        global_num += local_num;
        #pragma omp atomic
        global_den += local_den;
    }

    return (global_den > 0) ? (global_num / global_den) : 0.0;
}

// -----------------------------------------------------------------------------
// pybind11 module
// -----------------------------------------------------------------------------
PYBIND11_MODULE(centrality_core, m) {
    m.doc() = "Graph-level straightness centrality computation in C++";
    m.def("graph_centrality", &compute_graph_straightness_core,
          "Compute graph-level straightness centrality",
          py::arg("coords_x"),
          py::arg("coords_y"),
          py::arg("shortest_paths"),
          py::arg("n"));
}
