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
#include <mutex>
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
        return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
    }

    struct Settings
    {
        int order = 4; //!< starting low Gauss order
        double tol = 1e-7; //!< order-pair convergence tolerance
        int max_level = 5; //!< maximum per-axis grading level
        int max_order = 16; //!< maximum Gauss order
        int sphere_n = 10; //!< starting angular order for in-support boxes
        int sphere_max_n = 32; //!< maximum angular order
        int sphere_radial_order = 4; //!< radial Gauss order per corner box
        int in_support_box_order = 256; //!< Gauss order per face of the analytic box field
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
        std::size_t total = 0;
        std::size_t not_converged = 0;
        std::size_t near_count = 0;
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
                        Vec3 const R{R0.x - sx * cells.x, R0.y - sy * cells.y, R0.z - sz * cells.z};
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

        //! Coulomb field at an interior point of the physical TSC support box:
        //!     J(x_obs) = ∫_Box (x_obs - p)/|x_obs - p|^3 d^3p
        //! where Box is the box [x_src±1.5·cells] (the TSC support in lab space).
        //! By the divergence theorem (x_obs - p)/|x_obs - p|^3 = ∇_p(1/|x_obs - p|),
        //! so J is a smooth surface integral over the six faces.  The integrand
        //! 1/|x_obs - p| is nonsingular on every face because x_obs is interior,
        //! so plain Gauss converges spectrally and J is exact to roundoff.  This
        //! is the analytic singular part of the in-support shape integral.
        inline Vec3 box_field(Vec3 const& x_obs, Vec3 const& x_src, Vec3 const& cells, int n)
        {
            double const ext[3] = {1.5 * cells.x, 1.5 * cells.y, 1.5 * cells.z};
            double const lo[3] = {x_src.x - ext[0], x_src.y - ext[1], x_src.z - ext[2]};
            double const hi[3] = {x_src.x + ext[0], x_src.y + ext[1], x_src.z + ext[2]};
            std::vector<double> xn, wn;
            gauss_legendre(n, xn, wn);
            Vec3 out{};
            for(int axis = 0; axis < 3; ++axis)
                for(int sgn : {-1, 1})
                {
                    double const fixed = (sgn < 0) ? lo[axis] : hi[axis];
                    double const df = x_obs[axis] - fixed;
                    int const o0 = (axis + 1) % 3;
                    int const o1 = (axis + 2) % 3;
                    double const h0 = 0.5 * (hi[o0] - lo[o0]);
                    double const m0 = 0.5 * (hi[o0] + lo[o0]);
                    double const h1 = 0.5 * (hi[o1] - lo[o1]);
                    double const m1 = 0.5 * (hi[o1] + lo[o1]);
                    double P = 0.0;
                    for(int i = 0; i < n; ++i)
                    {
                        double const v0 = m0 + h0 * xn[i];
                        double const d0 = v0 - x_obs[o0];
                        for(int j = 0; j < n; ++j)
                        {
                            double const v1 = m1 + h1 * xn[j];
                            double const d1 = v1 - x_obs[o1];
                            P += (h0 * h1) * wn[i] * wn[j] / std::sqrt(d0 * d0 + d1 * d1 + df * df);
                        }
                    }
                    out[axis] += sgn * P;
                }
            return out;
        }

        //! Error-controlled box/pyramid quadrature for t inside the support.
        //! The support is split into the 27 TSC sub-boxes.  The box containing
        //! t is split further at t into corner boxes, each covered by three
        //! pyramids with t as the apex; r^2 cancels the 1/r^2 kernel and W is a
        //! single polynomial on each sub-box.
        inline void box_support_integral(
            Vec3 const& R0,
            Vec3 const& u,
            Vec3 const& cells,
            int nang,
            int nr,
            int ng,
            Vec3& Eout,
            Vec3& Bout)
        {
            Vec3 const t{R0.x / cells.x, R0.y / cells.y, R0.z / cells.z};
            double const tb[3] = {t.x, t.y, t.z};
            double const box_lo[3] = {-1.5, -1.5, -1.5};

            std::vector<double> xa, wa, xr, wr, xg, wg;
            gauss_legendre(nang, xa, wa);
            gauss_legendre(nr, xr, wr);
            gauss_legendre(ng, xg, wg);
            std::array<double, 6> acc{};

            auto accumulate = [&](Vec3 const& s, double w)
            {
                double const W = tsc_shape(s.x) * tsc_shape(s.y) * tsc_shape(s.z);
                if(W == 0.0)
                    return;
                Vec3 const R = R0 - Vec3{s.x * cells.x, s.y * cells.y, s.z * cells.z};
                Vec3 E, B;
                point_field(R, u, E, B);
                double const ww = w * W;
                acc[0] += ww * E.x;
                acc[1] += ww * E.y;
                acc[2] += ww * E.z;
                acc[3] += ww * B.x;
                acc[4] += ww * B.y;
                acc[5] += ww * B.z;
            };

            for(int ix = 0; ix < 3; ++ix)
                for(int iy = 0; iy < 3; ++iy)
                    for(int iz = 0; iz < 3; ++iz)
                    {
                        double const lo[3] = {box_lo[0] + ix, box_lo[1] + iy, box_lo[2] + iz};
                        double const hi[3] = {lo[0] + 1.0, lo[1] + 1.0, lo[2] + 1.0};
                        bool contains = true;
                        for(int a = 0; a < 3; ++a)
                            contains = contains && lo[a] <= tb[a] && tb[a] <= hi[a];

                        if(!contains)
                        {
                            for(int a0 = 0; a0 < ng; ++a0)
                            {
                                double const s0 = 0.5 * (lo[0] + hi[0]) + 0.5 * (hi[0] - lo[0]) * xg[a0];
                                double const w0 = 0.5 * (hi[0] - lo[0]) * wg[a0];
                                for(int a1 = 0; a1 < ng; ++a1)
                                {
                                    double const s1 = 0.5 * (lo[1] + hi[1]) + 0.5 * (hi[1] - lo[1]) * xg[a1];
                                    double const w1 = 0.5 * (hi[1] - lo[1]) * wg[a1];
                                    for(int a2 = 0; a2 < ng; ++a2)
                                    {
                                        double const s2 = 0.5 * (lo[2] + hi[2]) + 0.5 * (hi[2] - lo[2]) * xg[a2];
                                        double const w2 = 0.5 * (hi[2] - lo[2]) * wg[a2];
                                        accumulate({s0, s1, s2}, w0 * w1 * w2);
                                    }
                                }
                            }
                            continue;
                        }

                        // Split the containing box at t into up to 8 corner boxes.
                        std::array<std::vector<std::array<double, 3>>, 3> choices;
                        for(int a = 0; a < 3; ++a)
                        {
                            if(lo[a] < tb[a])
                                choices[a].push_back({lo[a], tb[a], -1.0});
                            if(tb[a] < hi[a])
                                choices[a].push_back({tb[a], hi[a], 1.0});
                        }
                        for(auto const& cx : choices[0])
                            for(auto const& cy : choices[1])
                                for(auto const& cz : choices[2])
                                {
                                    std::array<std::array<double, 3>, 3> seg{cx, cy, cz};
                                    double L[3], sgn[3];
                                    for(int a = 0; a < 3; ++a)
                                    {
                                        L[a] = seg[a][1] - seg[a][0];
                                        sgn[a] = seg[a][2];
                                    }
                                    if(L[0] <= 0.0 || L[1] <= 0.0 || L[2] <= 0.0)
                                        continue;
                                    double const V = L[0] * L[1] * L[2];
                                    for(int face = 0; face < 3; ++face)
                                    {
                                        int const o0 = (face + 1) % 3;
                                        int const o1 = (face + 2) % 3;
                                        for(int iu = 0; iu < nang; ++iu)
                                        {
                                            double const uu = 0.5 * (xa[iu] + 1.0);
                                            double const wu = 0.5 * wa[iu];
                                            for(int iv = 0; iv < nang; ++iv)
                                            {
                                                double const vv = 0.5 * (xa[iv] + 1.0);
                                                double const wv = 0.5 * wa[iv];
                                                double q[3] = {0.0, 0.0, 0.0};
                                                q[face] = sgn[face] * L[face];
                                                q[o0] = sgn[o0] * uu * L[o0];
                                                q[o1] = sgn[o1] * vv * L[o1];
                                                for(int ir = 0; ir < nr; ++ir)
                                                {
                                                    double const r = 0.5 * (xr[ir] + 1.0);
                                                    double const w = 0.5 * wr[ir] * wu * wv * V * r * r;
                                                    Vec3 const s{t.x + r * q[0], t.y + r * q[1], t.z + r * q[2]};
                                                    Vec3 const R{
                                                        -r * q[0] * cells.x,
                                                        -r * q[1] * cells.y,
                                                        -r * q[2] * cells.z};
                                                    double const W = tsc_shape(s.x) * tsc_shape(s.y) * tsc_shape(s.z);
                                                    if(W == 0.0)
                                                        continue;
                                                    Vec3 E, B;
                                                    point_field(R, u, E, B);
                                                    double const ww = w * W;
                                                    acc[0] += ww * E.x;
                                                    acc[1] += ww * E.y;
                                                    acc[2] += ww * E.z;
                                                    acc[3] += ww * B.x;
                                                    acc[4] += ww * B.y;
                                                    acc[5] += ww * B.z;
                                                }
                                            }
                                        }
                                    }
                                }
                    }
            Eout = {acc[0], acc[1], acc[2]};
            Bout = {acc[3], acc[4], acc[5]};
        }

        //! Smooth-remainder quadrature for the analytic-singularity subtraction:
        //!     ∫_S (W(s) - w_offset) K(s) d^3 s ,  K(s) = (R0 - s⊙c)/|R0 - s⊙c|^3
        //! The support box [-1.5,1.5]^3 is split at t (the singularity) into up to
        //! eight octants, so the integrable 1/r singularity (one power milder than
        //! the bare 1/r^2 kernel, because W(s)-w_offset vanishes at s=t) sits at an
        //! octant corner.  Plain Gauss on each octant converges algebraically and
        //! monotonically, so the n vs 2n order-pair is a genuine (non-saturating)
        //! error estimate.  Used for the stationary (u = 0) in-support case, where
        //! w_offset = W(t).
        inline void remainder_support_integral(
            Vec3 const& R0,
            Vec3 const& u,
            Vec3 const& cells,
            int n,
            double w_offset,
            Vec3& Eout,
            Vec3& Bout)
        {
            double const tb[3] = {R0.x / cells.x, R0.y / cells.y, R0.z / cells.z};
            // Octant segments per axis: split [-1.5, 1.5] at t (drop zero width).
            std::array<std::vector<std::array<double, 2>>, 3> seg;
            for(int a = 0; a < 3; ++a)
            {
                if(tb[a] > -1.5)
                    seg[a].push_back({-1.5, tb[a]});
                if(tb[a] < 1.5)
                    seg[a].push_back({tb[a], 1.5});
            }
            std::vector<double> xn, wn;
            gauss_legendre(n, xn, wn);
            std::array<double, 6> acc{};
            for(auto const& px : seg[0])
                for(auto const& py : seg[1])
                    for(auto const& pz : seg[2])
                    {
                        double const hx = 0.5 * (px[1] - px[0]);
                        double const mx = 0.5 * (px[1] + px[0]);
                        double const hy = 0.5 * (py[1] - py[0]);
                        double const my = 0.5 * (py[1] + py[0]);
                        double const hz = 0.5 * (pz[1] - pz[0]);
                        double const mz = 0.5 * (pz[1] + pz[0]);
                        for(int i = 0; i < n; ++i)
                        {
                            double const sx = mx + hx * xn[i];
                            double const wx = hx * wn[i];
                            for(int j = 0; j < n; ++j)
                            {
                                double const sy = my + hy * xn[j];
                                double const wxy = wx * hy * wn[j];
                                for(int k = 0; k < n; ++k)
                                {
                                    double const sz = mz + hz * xn[k];
                                    double const W = tsc_shape(sx) * tsc_shape(sy) * tsc_shape(sz);
                                    double const Ww = W - w_offset;
                                    if(Ww == 0.0)
                                        continue;
                                    Vec3 const R{R0.x - sx * cells.x, R0.y - sy * cells.y, R0.z - sz * cells.z};
                                    Vec3 E, B;
                                    point_field(R, u, E, B);
                                    double const ww = wxy * hz * wn[k] * Ww;
                                    acc[0] += ww * E.x;
                                    acc[1] += ww * E.y;
                                    acc[2] += ww * E.z;
                                    acc[3] += ww * B.x;
                                    acc[4] += ww * B.y;
                                    acc[5] += ww * B.z;
                                }
                            }
                        }
                    }
            Eout = {acc[0], acc[1], acc[2]};
            Bout = {acc[3], acc[4], acc[5]};
        }

        inline std::array<std::vector<Panel>, 3> panels_for(Vec3 const& target, int level)
        {
            return {graded_panels(target.x, level), graded_panels(target.y, level), graded_panels(target.z, level)};
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
        double const support_dist = detail::support_distance(R0, cells);
        bool const is_near = detail::near_support(R0, cells);
        double err = 0.0;
        int level = 0;
        bool converged = false;

        if(support_dist == 0.0)
        {
            double const uu2 = dot(u, u);
            if(uu2 == 0.0)
            {
                // Stationary (Coulomb) source inside the support.  The kernel
                // (R0 - s)/(R0 - s)^3 is an isotropic 1/r^3 singularity at s = t,
                // so we split the shape integral analytically:
                //     E = W(t) * ∫_S K            +   ∫_S (W - W(t)) K
                //         = S_part  (exact: the field of the uniform support    +  R_part (smooth:
                //            box at x_obs, via the divergence theorem)            (W - W(t)) kills one
                //                                                                  power of the 1/r kernel)
                // S_part is exact to roundoff and R_part is smooth, so the
                // order-pair estimate on R_part is a genuine (non-saturating)
                // error estimate instead of the bare 1/r^2 pyramid estimate.
                Vec3 const t{R0.x / cells.x, R0.y / cells.y, R0.z / cells.z};
                double const Wt = tsc_shape(t.x) * tsc_shape(t.y) * tsc_shape(t.z);
                Vec3 const S
                    = detail::box_field(x_obs, x_src, cells, settings.in_support_box_order)
                      * (Wt / (cells.x * cells.y * cells.z));

                // R_part via plain Gauss on the octants (split at t), order-doubling
                // until the order-pair estimate drops below tol.  The 1/r remainder
                // converges algebraically and monotonically, so the n vs 2n pair is
                // a genuine (non-saturating) error estimate.
                int n = 16;
                int const n_max = 512;
                Vec3 E1, B1, E2, B2;
                detail::remainder_support_integral(R0, u, cells, n, Wt, E1, B1);
                for(;;)
                {
                    int const n2 = std::min(2 * n, n_max);
                    detail::remainder_support_integral(R0, u, cells, n2, Wt, E2, B2);
                    std::array<double, 6> a{E1.x, E1.y, E1.z, B1.x, B1.y, B1.z};
                    std::array<double, 6> b{E2.x, E2.y, E2.z, B2.x, B2.y, B2.z};
                    err = detail::richardson_error(a, b);
                    if(err <= settings.tol || n2 >= n_max)
                    {
                        converged = err <= settings.tol;
                        E = S + E2;
                        B = B2; // B = 0 for a stationary source
                        break;
                    }
                    E1 = E2;
                    B1 = B2;
                    n = n2;
                }
            }
            else
            {
                // Singular point inside the TSC support (moving source): split
                // into TSC sub-boxes and use pyramid coordinates in the box
                // containing t.  (The isotropic analytic subtraction above is
                // exact only for u = 0; the relativistic kernel is anisotropic.)
                int nang = settings.sphere_n;
                int ng = 8;
                Vec3 E1, B1, E2, B2;
                detail::box_support_integral(R0, u, cells, nang, settings.sphere_radial_order, ng, E1, B1);
                for(;;)
                {
                    int const nang2 = std::min(nang + 4, settings.sphere_max_n);
                    int const ng2 = std::min(ng + 2, 16);
                    detail::box_support_integral(R0, u, cells, nang2, settings.sphere_radial_order, ng2, E2, B2);
                    std::array<double, 6> a{E1.x, E1.y, E1.z, B1.x, B1.y, B1.z};
                    std::array<double, 6> b{E2.x, E2.y, E2.z, B2.x, B2.y, B2.z};
                    err = detail::richardson_error(a, b);
                    if(err <= settings.tol || (nang2 >= settings.sphere_max_n && ng2 >= 16))
                    {
                        converged = err <= settings.tol;
                        E = E2;
                        B = B2;
                        break;
                    }
                    E1 = E2;
                    B1 = B2;
                    nang = nang2;
                    ng = ng2;
                }
            }
        }
        else
        {
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
        }

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
        constexpr std::size_t nslots = 8 * 8 * 4 * 6;
        std::vector<double> tile(nslots, 0.0);
        Vec3 const cells{dx, dy, dz};
        std::size_t const nsrc = sources.size();
        std::size_t const total = nsrc * nslots;

        // Each (source, slot) is independent.  Writing into a per-source
        // contribution array and reducing in source order makes the result
        // bitwise independent of the OpenMP thread count.
        std::vector<double> contrib(total, 0.0);
        std::vector<double> errs(total, 0.0);
        std::vector<unsigned char> conv(total, 1);
        std::vector<unsigned char> isnear(total, 0);

#pragma omp parallel for schedule(dynamic, 16)
        for(std::size_t idx = 0; idx < total; ++idx)
        {
            std::size_t const si = idx / nslots;
            std::size_t const slot = idx % nslots;
            Source const& src = sources[si];
            if(src.w == 0.0)
                continue;
            int const c = static_cast<int>(slot % 6);
            std::size_t r = slot / 6;
            int const i = static_cast<int>(r % 8);
            r /= 8;
            int const j = static_cast<int>(r % 8);
            int const k = static_cast<int>(r / 8);
            Vec3 const pos = x_tile
                             + Vec3{
                                 (i + offsets()[c].x) * cells.x,
                                 (j + offsets()[c].y) * cells.y,
                                 (k + offsets()[c].z) * cells.z};
            Vec3 E, B;
            ShapeResult sr;
            shape_field(pos, src.x, src.u, dx, dy, dz, settings, E, B, &sr);
            double const component = c < 3 ? E[c] : B[c - 3];
            contrib[idx] = -src.w * dx * dx * component;
            errs[idx] = sr.error;
            conv[idx] = sr.converged ? 1 : 0;
            isnear[idx] = sr.near ? 1 : 0;
        }

        TileReport local_report;
        for(std::size_t slot = 0; slot < nslots; ++slot)
        {
            double sum = 0.0;
            for(std::size_t si = 0; si < nsrc; ++si)
                sum += contrib[si * nslots + slot];
            tile[slot] = sum;
        }
        local_report.total = total;
        for(std::size_t idx = 0; idx < total; ++idx)
        {
            local_report.max_error = std::max(local_report.max_error, errs[idx]);
            if(!conv[idx])
                ++local_report.not_converged;
            if(isnear[idx])
                ++local_report.near_count;
        }
        local_report.all_converged = local_report.not_converged == 0;
        local_report.any_near = local_report.near_count != 0;

        if(report)
            *report = local_report;
        return tile;
    }

} // namespace picongpu::particles::neuralBunch::reference
