// Header-only FP64 reference for the neural bunch-field initializer.
//
// No alpaka, PIConGPU, Python, BLAS or FFT dependency.  Mirrors
// ../convkernel.py: present-time point kernel, lab-frame TSC shape integral
// (Gauss-Legendre with adaptive per-axis grading near support) and the
// staggered Yee tile in the serialized order c + 6*(i + 8*(j + 8*k)).
//
// Conventions
//   u          = p/(m_e c), gamma = sqrt(1+|u|^2), beta = u/gamma
//   E_n        = 4*pi*eps0*E/q
//   B_n        = 4*pi*eps0*c*B/q = beta x E_n
//   tile slot  = (q/e) * dx^2 * (E_n[c] or B_n[c-3]), q/e = -w
//
// There is no epsilon regularization: near-singular shape integrands are
// integrated by refining panels, never by clipping a denominator.

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace picongpu::particles::neuralBunch::reference
{
    struct Vec3
    {
        double x = 0.0, y = 0.0, z = 0.0;

        double& operator[](int i)
        {
            return (&x)[i];
        }

        double const& operator[](int i) const
        {
            return (&x)[i];
        }
    };

    inline Vec3 operator+(Vec3 a, Vec3 const& b)
    {
        return {a.x + b.x, a.y + b.y, a.z + b.z};
    }

    inline Vec3 operator-(Vec3 a, Vec3 const& b)
    {
        return {a.x - b.x, a.y - b.y, a.z - b.z};
    }

    inline Vec3 operator*(Vec3 a, double s)
    {
        return {a.x * s, a.y * s, a.z * s};
    }

    inline Vec3 operator*(double s, Vec3 a)
    {
        return a * s;
    }

    inline Vec3 operator/(Vec3 a, double s)
    {
        return {a.x / s, a.y / s, a.z / s};
    }

    inline double dot(Vec3 const& a, Vec3 const& b)
    {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    inline Vec3 cross(Vec3 const& a, Vec3 const& b)
    {
        return {
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
    }

    struct Settings
    {
        int order = 4; //!< starting low Gauss order
        double tol = 1e-7; //!< order-pair convergence tolerance
        int max_level = 5; //!< maximum per-axis grading level
        int max_order = 16; //!< maximum Gauss order
    };

    struct ShapeResult
    {
        double error = 0.0; //!< order-pair estimate
        bool near = false;
        int level = 0;
        bool converged = true;
    };

    struct TileReport
    {
        double max_error = 0.0;
        bool all_converged = true;
        bool any_near = false;
    };

    struct Source
    {
        Vec3 x{};
        Vec3 u{};
        double w = 1.0; //!< q/e = -w
    };

    //! Yee offsets in cell units, component order Ex,Ey,Ez,Bx,By,Bz.
    inline std::array<Vec3, 6> const& offsets()
    {
        static std::array<Vec3, 6> const value{{
            {0.5, 0.0, 0.0},
            {0.0, 0.5, 0.0},
            {0.0, 0.0, 0.5},
            {0.0, 0.5, 0.5},
            {0.5, 0.0, 0.5},
            {0.5, 0.5, 0.0},
        }};
        return value;
    }

    inline double tsc_shape(double x)
    {
        double const ax = std::abs(x);
        if(ax < 0.5)
            return 0.75 - ax * ax;
        if(ax < 1.5)
            return 0.5 * (1.5 - ax) * (1.5 - ax);
        return 0.0;
    }

    //! Stable present-time point kernel (per-unit-charge fields).
    inline void point_field(Vec3 const& R, Vec3 const& u, Vec3& E, Vec3& B)
    {
        double const uu2 = dot(u, u);
        double const inv_g2 = 1.0 / (1.0 + uu2);

        if(uu2 == 0.0)
        {
            double const r2 = dot(R, R);
            double const denom = r2 * std::sqrt(r2);
            E = R / denom;
            B = Vec3{0.0, 0.0, 0.0}; // explicit static branch, exact zero
            return;
        }

        double const speed = std::sqrt(uu2);
        Vec3 const bhat = u / speed;
        double const r_par = dot(R, bhat);
        Vec3 const r_perp = R - r_par * bhat;
        double const D = r_par * r_par + dot(r_perp, r_perp) * inv_g2;
        double const denom = D * std::sqrt(D);

        E = R * (inv_g2 / denom);
        Vec3 const beta = u / std::sqrt(1.0 + uu2);
        B = cross(beta, E);
    }

    namespace detail
    {
        struct Panel
        {
            double a, b;
        };

        inline std::array<std::vector<Panel>, 3> base_panels()
        {
            std::array<std::vector<Panel>, 3> out;
            for(auto& axis : out)
                axis = {{-1.5, -0.5}, {-0.5, 0.5}, {0.5, 1.5}};
            return out;
        }

        inline std::vector<Panel> graded_panels(double target, int level)
        {
            std::vector<Panel> const base{{-1.5, -0.5}, {-0.5, 0.5}, {0.5, 1.5}};
            if(level <= 0)
                return base;

            std::vector<Panel> out;
            for(Panel const& p : base)
            {
                double const h = p.b - p.a;
                if(target < p.a - h || target > p.b + h)
                {
                    out.push_back(p);
                    continue;
                }

                std::vector<double> points{p.a, p.b};
                if(p.a <= target && target <= p.b)
                {
                    for(int k = 0; k <= level; ++k)
                    {
                        double const off = h * 0.5 * std::ldexp(1.0, -k);
                        points.push_back(std::min(p.b, std::max(p.a, target - off)));
                        points.push_back(std::min(p.b, std::max(p.a, target + off)));
                    }
                }
                else if(target < p.a)
                {
                    for(int k = 0; k <= level; ++k)
                        points.push_back(std::min(p.b, p.a + h * std::ldexp(1.0, -k)));
                }
                else
                {
                    for(int k = 0; k <= level; ++k)
                        points.push_back(std::max(p.a, p.b - h * std::ldexp(1.0, -k)));
                }

                std::sort(points.begin(), points.end());
                points.erase(std::unique(points.begin(), points.end()), points.end());
                for(std::size_t i = 0; i + 1 < points.size(); ++i)
                    if(points[i + 1] > points[i])
                        out.push_back({points[i], points[i + 1]});
            }
            return out;
        }

        inline void gauss_legendre(int n, std::vector<double>& nodes, std::vector<double>& weights)
        {
            nodes.assign(n, 0.0);
            weights.assign(n, 0.0);
            int const m = (n + 1) / 2;
            for(int i = 0; i < m; ++i)
            {
                double z = std::cos(3.141592653589793238462643383279502884 * (i + 0.75) / (n + 0.5));
                double pp = 0.0;
                for(int iteration = 0; iteration < 100; ++iteration)
                {
                    double p1 = 1.0;
                    double p2 = 0.0;
                    for(int j = 1; j <= n; ++j)
                    {
                        double const p3 = p2;
                        p2 = p1;
                        p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / j;
                    }
                    pp = n * (z * p1 - p2) / (z * z - 1.0);
                    double const z1 = z;
                    z = z1 - p1 / pp;
                    if(std::abs(z - z1) <= 1e-15)
                        break;
                }
                nodes[i] = -z;
                nodes[n - 1 - i] = z;
                double const w = 2.0 / ((1.0 - z * z) * pp * pp);
                weights[i] = w;
                weights[n - 1 - i] = w;
            }
        }

        struct AxisRule
        {
            std::vector<double> nodes;
            std::vector<double> weights;
        };

        inline AxisRule build_rule(std::vector<Panel> const& panels, int order)
        {
            AxisRule rule;
            std::vector<double> xn, wn;
            gauss_legendre(order, xn, wn);
            for(Panel const& p : panels)
            {
                double const half = 0.5 * (p.b - p.a);
                double const mid = 0.5 * (p.a + p.b);
                for(int i = 0; i < order; ++i)
                {
                    double const node = half * xn[i] + mid;
                    rule.nodes.push_back(node);
                    rule.weights.push_back(half * wn[i] * tsc_shape(node));
                }
            }
            return rule;
        }

        inline std::array<double, 6> integral_on_panels(
            std::array<std::vector<Panel>, 3> const& panels,
            int order,
            Vec3 const& R0,
            Vec3 const& cells,
            Vec3 const& u)
        {
            std::array<AxisRule, 3> rules{
                build_rule(panels[0], order),
                build_rule(panels[1], order),
                build_rule(panels[2], order)};

            std::array<double, 6> acc{};
            for(std::size_t ix = 0; ix < rules[0].nodes.size(); ++ix)
            {
                double const sx = rules[0].nodes[ix];
                double const wx = rules[0].weights[ix];
                for(std::size_t iy = 0; iy < rules[1].nodes.size(); ++iy)
                {
                    double const sy = rules[1].nodes[iy];
                    double const wxy = wx * rules[1].weights[iy];
                    for(std::size_t iz = 0; iz < rules[2].nodes.size(); ++iz)
                    {
                        double const sz = rules[2].nodes[iz];
                        double const w = wxy * rules[2].weights[iz];
                        Vec3 const R{
                            R0.x - sx * cells.x,
                            R0.y - sy * cells.y,
                            R0.z - sz * cells.z};
                        Vec3 E, B;
                        point_field(R, u, E, B);
                        acc[0] += w * E.x;
                        acc[1] += w * E.y;
                        acc[2] += w * E.z;
                        acc[3] += w * B.x;
                        acc[4] += w * B.y;
                        acc[5] += w * B.z;
                    }
                }
            }
            return acc;
        }

        //! Physical distance to the TSC support box.  Diagnostic only: the
        //! quadrature is error-controlled for every point regardless.
        inline double support_distance(Vec3 const& R0, Vec3 const& cells)
        {
            Vec3 const excess{
                std::max(std::abs(R0.x) - 1.5 * cells.x, 0.0),
                std::max(std::abs(R0.y) - 1.5 * cells.y, 0.0),
                std::max(std::abs(R0.z) - 1.5 * cells.z, 0.0)};
            return std::sqrt(dot(excess, excess));
        }

        inline bool near_support(Vec3 const& R0, Vec3 const& cells)
        {
            double const max_cell = std::max({cells.x, cells.y, cells.z});
            return support_distance(R0, cells) < 2.0 * max_cell;
        }

        inline double richardson_error(std::array<double, 6> const& a, std::array<double, 6> const& b)
        {
            double max_diff = 0.0;
            double max_val = 0.0;
            for(int i = 0; i < 6; ++i)
            {
                max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
                max_val = std::max(max_val, std::abs(a[i]));
                max_val = std::max(max_val, std::abs(b[i]));
            }
            return max_diff / (1.0 + max_val);
        }

        inline std::array<std::vector<Panel>, 3> panels_for(Vec3 const& target, int level)
        {
            return {
                graded_panels(target.x, level),
                graded_panels(target.y, level),
                graded_panels(target.z, level)};
        }
    } // namespace detail

    //! Shape-integrated per-unit-charge field of one source.
    inline void shape_field(
        Vec3 const& x_obs,
        Vec3 const& x_src,
        Vec3 const& u,
        double dx,
        double dy,
        double dz,
        Settings const& settings,
        Vec3& E,
        Vec3& B,
        ShapeResult* result = nullptr)
    {
        Vec3 const cells{dx, dy, dz};
        if(!(dx > 0.0) || !(dy > 0.0) || !(dz > 0.0))
            throw std::invalid_argument("cell sizes must be positive");

        Vec3 const R0 = x_obs - x_src;
        bool const is_near = detail::near_support(R0, cells);
        double err = 0.0;
        int level = 0;
        bool converged = false;
        Vec3 const target{R0.x / cells.x, R0.y / cells.y, R0.z / cells.z};

        int order_lo = settings.order;
        auto panels = detail::panels_for(target, level);
        auto a = detail::integral_on_panels(panels, order_lo, R0, cells, u);
        auto b = detail::integral_on_panels(panels, order_lo + 2, R0, cells, u);
        for(;;)
        {
            err = detail::richardson_error(a, b);
            if(err <= settings.tol)
            {
                converged = true;
                break;
            }
            if(level < settings.max_level)
            {
                ++level;
                order_lo += 2;
                panels = detail::panels_for(target, level);
                a = detail::integral_on_panels(panels, order_lo, R0, cells, u);
            }
            else
            {
                if(order_lo + 2 >= settings.max_order)
                    break;
                order_lo += 2;
                a = b;
            }
            b = detail::integral_on_panels(panels, order_lo + 2, R0, cells, u);
        }
        E = {b[0], b[1], b[2]};
        B = {b[3], b[4], b[5]};

        if(result)
        {
            result->error = err;
            result->near = is_near;
            result->level = level;
            result->converged = converged;
        }
    }

    //! All 1536 staggered outputs in c + 6*(i + 8*(j + 8*k)) order.
    inline std::vector<double> tile_field(
        Vec3 const& x_tile,
        std::vector<Source> const& sources,
        double dx,
        double dy,
        double dz,
        Settings const& settings = {},
        TileReport* report = nullptr)
    {
        std::vector<double> tile(8 * 8 * 4 * 6, 0.0);
        TileReport local_report;
        Vec3 const cells{dx, dy, dz};

        for(Source const& src : sources)
        {
            if(src.w == 0.0)
                continue;
            for(int k = 0; k < 4; ++k)
            {
                for(int j = 0; j < 8; ++j)
                {
                    for(int i = 0; i < 8; ++i)
                    {
                        for(int c = 0; c < 6; ++c)
                        {
                            Vec3 const pos = x_tile
                                + Vec3{
                                    (i + offsets()[c].x) * cells.x,
                                    (j + offsets()[c].y) * cells.y,
                                    (k + offsets()[c].z) * cells.z};
                            Vec3 E, B;
                            ShapeResult sr;
                            shape_field(pos, src.x, src.u, dx, dy, dz, settings, E, B, &sr);
                            double const component = c < 3 ? E[c] : B[c - 3];
                            std::size_t const index =
                                static_cast<std::size_t>(c + 6 * (i + 8 * (j + 8 * k)));
                            tile[index] += -src.w * dx * dx * component;

                            local_report.max_error = std::max(local_report.max_error, sr.error);
                            local_report.all_converged = local_report.all_converged && sr.converged;
                            local_report.any_near = local_report.any_near || sr.near;
                        }
                    }
                }
            }
        }

        if(report)
            *report = local_report;
        return tile;
    }

} // namespace picongpu::particles::neuralBunch::reference
