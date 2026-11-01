// Simple CLI: evaluate one source/tile and print the 1536 tile values.
// Build: g++ -std=c++20 -O3 -march=native -fopenmp tile_bench.cpp -o tile_bench
// Run:   ./tile_bench sx sy sz ox oy oz dx dy dz [tol]
#include "neural_reference.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#ifdef _OPENMP
#    include <omp.h>
#endif

int main(int argc, char** argv)
{
    if(argc < 10)
    {
        std::fprintf(stderr, "usage: %s sx sy sz ox oy oz dx dy dz [tol]\n", argv[0]);
        return 2;
    }
    double const v[9]
        = {std::atof(argv[1]),
           std::atof(argv[2]),
           std::atof(argv[3]),
           std::atof(argv[4]),
           std::atof(argv[5]),
           std::atof(argv[6]),
           std::atof(argv[7]),
           std::atof(argv[8]),
           std::atof(argv[9])};
    nbref::Settings s;
    if(argc > 10)
        s.tol = std::atof(argv[10]);
    nbref::Source src;
    src.x = {v[0], v[1], v[2]};
    src.u = {0.3, 0.1, 0.0};
    src.w = 1.0;
    nbref::Vec3 const x_tile{v[3], v[4], v[5]};
    double const dx = v[6], dy = v[7], dz = v[8];

    nbref::TileReport report;
    auto const t0 = std::chrono::steady_clock::now();
    auto const tile = nbref::tile_field(x_tile, {src}, dx, dy, dz, s, &report);
    auto const t1 = std::chrono::steady_clock::now();
    double const dt = std::chrono::duration<double>(t1 - t0).count();

    for(std::size_t i = 0; i < tile.size(); ++i)
        std::printf("tile %zu %.17g\n", i, tile[i]);
    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    std::printf("time_seconds %.6f\n", dt);
    std::printf("threads %d\n", threads);
    std::printf("outputs %zu\n", report.total);
    std::printf("near_outputs %zu\n", report.near_count);
    std::printf("not_converged %zu\n", report.not_converged);
    std::printf("max_error %.6e\n", report.max_error);
    std::printf("converged %d\n", report.all_converged ? 1 : 0);
    return report.all_converged ? 0 : 1;
}
