// Reproducible C++ benchmark mirroring ../benchmark_tile.py.
// Build: g++ -std=c++20 -O2 benchmark.cpp -o benchmark
// Run:   ./benchmark

#include "neural_reference.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

using nbref::Source;
using nbref::Vec3;

namespace
{
    std::vector<Source> make_particles(int n, double distance, Vec3 const& x_tile)
    {
        std::vector<Source> out;
        out.reserve(static_cast<std::size_t>(n));
        for(int i = 0; i < n; ++i)
        {
            double const a = 2.0 * 3.14159265358979323846 * i / std::max(1, n);
            double const r = 3.0 * i / std::max(1, n);
            Source s;
            s.x = {x_tile.x - distance, x_tile.y + r * std::cos(a), x_tile.z + r * std::sin(a)};
            s.u = {0.3 * std::cos(a), 0.2 * std::sin(a), 0.1};
            s.w = 1.0 + 0.01 * i;
            out.push_back(s);
        }
        return out;
    }

    double rel_l2(std::vector<double> const& a, std::vector<double> const& b)
    {
        double num = 0.0, den = 0.0;
        for(std::size_t i = 0; i < b.size(); ++i)
        {
            num += (a[i] - b[i]) * (a[i] - b[i]);
            den += b[i] * b[i];
        }
        return std::sqrt(num / den);
    }

    double rel_max(std::vector<double> const& a, std::vector<double> const& b)
    {
        double num = 0.0, den = 0.0;
        for(std::size_t i = 0; i < b.size(); ++i)
        {
            num = std::max(num, std::abs(a[i] - b[i]));
            den = std::max(den, std::abs(b[i]));
        }
        return num / den;
    }
} // namespace

int main()
{
    Vec3 const x_tile{0.0, 0.0, 0.0};
    int const counts[] = {1, 16, 256};
    double const distances[] = {4.0, 8.0, 32.0};

    std::printf("distance  N    seconds  s/particle\n");
    for(double d : distances)
    {
        for(int n : counts)
        {
            auto particles = make_particles(n, d, x_tile);
            auto const t0 = std::chrono::steady_clock::now();
            auto tile = nbref::tile_field(x_tile, particles, 1.0, 1.0, 1.0);
            auto const t1 = std::chrono::steady_clock::now();
            double const dt = std::chrono::duration<double>(t1 - t0).count();
            std::printf("%8.1f %4d %9.3f %10.4f\n", d, n, dt, dt / n);
            std::fflush(stdout);
            if(tile.size() != 1536)
                return 3;
        }
    }

    std::printf("\naspect 10 (cells 1,10,10) tile cost\n");
    std::printf("distance  N    seconds  s/particle\n");
    for(double d : {4.0, 8.0, 32.0, 128.0})
    {
        auto particles = make_particles(256, d, x_tile);
        auto const t0 = std::chrono::steady_clock::now();
        auto tile = nbref::tile_field(x_tile, particles, 1.0, 10.0, 10.0);
        auto const t1 = std::chrono::steady_clock::now();
        double const dt = std::chrono::duration<double>(t1 - t0).count();
        std::printf("%8.1f %4d %9.3f %10.4f\n", d, 256, dt, dt / 256);
        std::fflush(stdout);
        if(tile.size() != 1536)
            return 3;
    }

    std::printf("\naccuracy order4 vs order8 (N=1)\n");
    std::printf("distance  relL2     relMax\n");
    for(double d : distances)
    {
        auto particles = make_particles(1, d, x_tile);
        nbref::Settings s4;
        nbref::Settings s8;
        s8.order = 8;
        auto t4 = nbref::tile_field(x_tile, particles, 1.0, 1.0, 1.0, s4);
        auto t8 = nbref::tile_field(x_tile, particles, 1.0, 1.0, 1.0, s8);
        std::printf("%8.1f %9.2e %9.2e\n", d, rel_l2(t4, t8), rel_max(t4, t8));
        std::fflush(stdout);
    }

    std::printf("\nadaptive near-point cost (single observation)\n");
    nbref::Settings s;
    for(double d : {2.0, 1.0, 0.5})
    {
        Vec3 E, B;
        nbref::ShapeResult sr;
        auto const t0 = std::chrono::steady_clock::now();
        nbref::shape_field(Vec3{d, 0.0, 0.0}, Vec3{}, Vec3{}, 1.0, 1.0, 1.0, s, E, B, &sr);
        auto const t1 = std::chrono::steady_clock::now();
        double const dt = std::chrono::duration<double>(t1 - t0).count();
        std::printf(
            "  R=%4.1f cells: %8.4f s  near=%d level=%d est_err=%.2e\n",
            d,
            dt,
            sr.near ? 1 : 0,
            sr.level,
            sr.error);
    }
    return 0;
}
