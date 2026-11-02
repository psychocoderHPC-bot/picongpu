// Small driver: compute the repo C++ reference tile for the 5 T13c comparison
// cases and write each as a flat text file (1536 doubles, canonical order
// c + 6*(i + 8*(j + 8*k))).  Independent of the Python reference; this is the
// repo's own C++ reference (neural_reference.hpp).
#include "neural_reference.hpp"
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace nbref;

struct Case { std::string name; Vec3 xtile; Vec3 xs; Vec3 u; double dx,dy,dz; };

int main(int argc, char** argv)
{
    std::string outdir = (argc > 1) ? argv[1] : ".";
    std::vector<Case> cases;
    // u normalized to the target gamma below.
    auto normu = [](Vec3 u, double u_norm)
    {
        double n = std::sqrt(u.x*u.x + u.y*u.y + u.z*u.z);
        if(n > 0.0)
            u = u / n * u_norm;
        return u;
    };
    double u10 = std::sqrt(100.0 - 1.0);   // gamma=10 -> |u|
    double u100 = std::sqrt(10000.0 - 1.0); // gamma=100 -> |u|

    cases.push_back({"stationary",    {0,0,0}, {0,0,0}, {0,0,0}, 1,1,1});
    cases.push_back({"gamma10_oblique", {0,0,0}, {0,0,0}, normu({9.9498743710662,1.0,0.3}, u10), 1,1,1});
    cases.push_back({"aspect10",       {0,0,0}, {0,0,0}, {0,0,0}, 1,10,10});
    cases.push_back({"target_at_source",{0,0,0}, {0.5,0,0}, {0,0,0}, 1,1,1});
    cases.push_back({"gamma100_aspect0p1", {0,0,0}, {0,0,0}, normu({99.995,0,0}, u100), 1,0.1,1});

    for(auto& c : cases)
    {
        std::vector<Source> srcs;
        srcs.push_back({c.xs, c.u, 1.0});
        std::vector<double> tile = tile_field(c.xtile, srcs, c.dx, c.dy, c.dz);
        std::string path = outdir + "/" + c.name + ".tile.txt";
        std::ofstream f(path);
        f.precision(17);
        for(double v : tile)
            f << v << "\n";
        f.close();
        printf("wrote %s (%zu values)\n", path.c_str(), tile.size());
    }
    return 0;
}
