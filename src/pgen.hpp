// Chasqui MHD — problem generators.
//
// Initialization runs on host mirrors (it is not performance critical) and the result is
// copied to the device. Magnetic fields are set either directly at face centers (when the
// analytic field is trivially divergence-free on faces) or through a vector potential
// evaluated on cell edges, which guarantees div(B) = 0 to round-off.
#pragma once
#include <string>
#include <cmath>
#include <stdexcept>
#include "defs.hpp"
#include "mesh.hpp"
#include "params.hpp"
#include "physics.hpp"

namespace pgen {

using HArr3 = Arr3::HostMirror;
using HArr4 = Arr4::HostMirror;

struct HostFields {
  HArr4 w;              // primitives
  HArr3 b1, b2, b3;     // face fields
};

constexpr double PI = 3.14159265358979323846;

// B = curl A from a vector potential A(comp, x, y, z) sampled at edge centers.
template <class F>
void FromVectorPotential(const Mesh &m, HostFields &h, F A) {
  const Real idx = Real(1) / m.dx1, idy = Real(1) / m.dx2, idz = Real(1) / m.dx3;
  for (int k = 0; k < m.nk + 1; ++k)
    for (int j = 0; j < m.nj + 1; ++j)
      for (int i = 0; i < m.ni + 1; ++i) {
        const Real xf = m.x1f(i), yf = m.x2f(j), zf = m.x3f(k);
        const Real xv = m.x1v(i), yv = m.x2v(j), zv = m.x3v(k);
        // B1 at (xf, yv, zv)
        Real b1 = Real(0);
        if (m.two_d) b1 += (A(2, xf, m.x2f(j + 1), zv) - A(2, xf, yf, zv)) * idy;
        if (m.three_d) b1 -= (A(1, xf, yv, m.x3f(k + 1)) - A(1, xf, yv, zf)) * idz;
        // B2 at (xv, yf, zv)
        Real b2 = -(A(2, m.x1f(i + 1), yf, zv) - A(2, xf, yf, zv)) * idx;
        if (m.three_d) b2 += (A(0, xv, yf, m.x3f(k + 1)) - A(0, xv, yf, zf)) * idz;
        // B3 at (xv, yv, zf)
        Real b3 = (A(1, m.x1f(i + 1), yv, zf) - A(1, xf, yv, zf)) * idx;
        if (m.two_d) b3 -= (A(0, xv, m.x2f(j + 1), zf) - A(0, xv, yf, zf)) * idy;
        h.b1(k, j, i) = b1; h.b2(k, j, i) = b2; h.b3(k, j, i) = b3;
      }
}

// Circularly polarized Alfven wave (exact nonlinear solution of ideal MHD), propagating
// along x' = x cos(th) + y sin(th). Returns to the initial state after T = lambda/v_A.
inline void CPAW(Params &pin, const Mesh &m, HostFields &h) {
  const Real dens = pin.GetOrAddReal("problem", "dens", 1.0);
  const Real pres = pin.GetOrAddReal("problem", "pres", 0.1);
  const Real b0 = pin.GetOrAddReal("problem", "b_par", 1.0);
  const Real amp = pin.GetOrAddReal("problem", "amp", 0.1);
  const Real lambda = pin.GetOrAddReal("problem", "lambda", 1.0);
  const Real th = pin.GetOrAddReal("problem", "angle", 0.0) * PI / 180.0;
  const Real vflow = pin.GetOrAddReal("problem", "vflow", 0.0);
  const Real dir = pin.GetOrAddReal("problem", "dir", 1.0);   // +1: along +x', -1: along -x'
  const Real k = 2.0 * PI / lambda;
  const Real cth = cos(th), sth = sin(th);
  const Real fac = -dir * amp / sqrt(dens);

  FromVectorPotential(m, h, [=](int c, Real x, Real y, Real /*z*/) -> Real {
    const Real xp = x * cth + y * sth, yp = -x * sth + y * cth;
    const Real ayp = (amp / k) * sin(k * xp);          // A_y'
    const Real az = b0 * yp + (amp / k) * cos(k * xp);  // A_z
    if (c == 0) return -ayp * sth;
    if (c == 1) return ayp * cth;
    return az;
  });
  // In 1D the uniform B1 = dA_z/dy cannot come from the potential: add it by hand.
  if (!m.two_d)
    for (int kk = 0; kk < m.nk + 1; ++kk)
      for (int j = 0; j < m.nj + 1; ++j)
        for (int i = 0; i < m.ni + 1; ++i) h.b1(kk, j, i) += b0 * cth;
  for (int kk = 0; kk < m.nk; ++kk)
    for (int j = 0; j < m.nj; ++j)
      for (int i = 0; i < m.ni; ++i) {
        const Real xp = m.x1v(i) * cth + m.x2v(j) * sth;
        const Real vyp = fac * sin(k * xp), vz = fac * cos(k * xp);
        h.w(IDN, kk, j, i) = dens;
        h.w(IVX, kk, j, i) = vflow * cth - vyp * sth;
        h.w(IVY, kk, j, i) = vflow * sth + vyp * cth;
        h.w(IVZ, kk, j, i) = vz;
        h.w(IPR, kk, j, i) = pres;
      }
}

// Orszag-Tang vortex (2D), standard units: rho = 25/(36 pi), p = 5/(12 pi), B0 = 1/sqrt(4 pi)
inline void OrszagTang(Params &pin, const Mesh &m, HostFields &h) {
  const Real b0 = 1.0 / sqrt(4.0 * PI);
  FromVectorPotential(m, h, [=](int c, Real x, Real y, Real /*z*/) -> Real {
    if (c != 2) return Real(0);
    return b0 / (2.0 * PI) * cos(2.0 * PI * y) + b0 / (4.0 * PI) * cos(4.0 * PI * x);
  });
  // B1 = dAz/dy = -b0 sin(2 pi y), B2 = -dAz/dx = b0 sin(4 pi x)
  for (int k = 0; k < m.nk; ++k)
    for (int j = 0; j < m.nj; ++j)
      for (int i = 0; i < m.ni; ++i) {
        h.w(IDN, k, j, i) = 25.0 / (36.0 * PI);
        h.w(IVX, k, j, i) = -sin(2.0 * PI * m.x2v(j));
        h.w(IVY, k, j, i) = sin(2.0 * PI * m.x1v(i));
        h.w(IVZ, k, j, i) = 0.0;
        h.w(IPR, k, j, i) = 5.0 / (12.0 * PI);
      }
  (void)pin;
}

// MHD blast wave: over-pressured sphere in a uniform oblique field
inline void Blast(Params &pin, const Mesh &m, HostFields &h) {
  const Real pout = pin.GetOrAddReal("problem", "pres_out", 0.1);
  const Real pin_ = pin.GetOrAddReal("problem", "pres_in", 10.0);
  const Real rad = pin.GetOrAddReal("problem", "radius", 0.1);
  const Real b0 = pin.GetOrAddReal("problem", "b0", 1.0);
  const Real ang = pin.GetOrAddReal("problem", "angle", 45.0) * PI / 180.0;
  Real bx = b0 * cos(ang), by = b0 * sin(ang);
  if (fabs(bx) < 1e-14 * b0) bx = 0.0;   // cos(90 deg) is 6e-17, not 0: keep exact symmetry
  if (fabs(by) < 1e-14 * b0) by = 0.0;
  for (int k = 0; k < m.nk + 1; ++k)
    for (int j = 0; j < m.nj + 1; ++j)
      for (int i = 0; i < m.ni + 1; ++i) { h.b1(k, j, i) = bx; h.b2(k, j, i) = by; h.b3(k, j, i) = 0.0; }
  for (int k = 0; k < m.nk; ++k)
    for (int j = 0; j < m.nj; ++j)
      for (int i = 0; i < m.ni; ++i) {
        const Real x = m.x1v(i), y = m.two_d ? m.x2v(j) : 0.0, z = m.three_d ? m.x3v(k) : 0.0;
        const Real r = sqrt(x * x + y * y + z * z);
        h.w(IDN, k, j, i) = 1.0;
        h.w(IVX, k, j, i) = h.w(IVY, k, j, i) = h.w(IVZ, k, j, i) = 0.0;
        h.w(IPR, k, j, i) = (r < rad) ? pin_ : pout;
      }
}

// Advected field loop (Gardiner & Stone 2005): A_z = max(A0 (R - r), 0)
inline void FieldLoop(Params &pin, const Mesh &m, HostFields &h) {
  const Real a0 = pin.GetOrAddReal("problem", "amp", 1.0e-3);
  const Real rad = pin.GetOrAddReal("problem", "radius", 0.3);
  const Real vx = pin.GetOrAddReal("problem", "vx", 1.0);
  const Real vy = pin.GetOrAddReal("problem", "vy", 0.5);
  const Real vz = pin.GetOrAddReal("problem", "vz", 0.0);
  FromVectorPotential(m, h, [=](int c, Real x, Real y, Real /*z*/) -> Real {
    if (c != 2) return Real(0);
    const Real r = sqrt(x * x + y * y);
    return fmax(a0 * (rad - r), Real(0));
  });
  for (int k = 0; k < m.nk; ++k)
    for (int j = 0; j < m.nj; ++j)
      for (int i = 0; i < m.ni; ++i) {
        h.w(IDN, k, j, i) = 1.0; h.w(IVX, k, j, i) = vx; h.w(IVY, k, j, i) = vy; h.w(IVZ, k, j, i) = vz;
        h.w(IPR, k, j, i) = 1.0;
      }
}

// Brio & Wu (1988) shock tube along x (use gamma = 2)
inline void ShockTube(Params &pin, const Mesh &m, HostFields &h) {
  const Real dl = pin.GetOrAddReal("problem", "dl", 1.0), pl = pin.GetOrAddReal("problem", "pl", 1.0);
  const Real dr = pin.GetOrAddReal("problem", "dr", 0.125), pr = pin.GetOrAddReal("problem", "pr", 0.1);
  const Real byl = pin.GetOrAddReal("problem", "byl", 1.0), byr = pin.GetOrAddReal("problem", "byr", -1.0);
  const Real bx = pin.GetOrAddReal("problem", "bx", 0.75);
  const Real x0 = pin.GetOrAddReal("problem", "x0", 0.0);
  for (int k = 0; k < m.nk + 1; ++k)
    for (int j = 0; j < m.nj + 1; ++j)
      for (int i = 0; i < m.ni + 1; ++i) {
        h.b1(k, j, i) = bx;
        h.b2(k, j, i) = (m.x1v(i) < x0) ? byl : byr;
        h.b3(k, j, i) = 0.0;
      }
  for (int k = 0; k < m.nk; ++k)
    for (int j = 0; j < m.nj; ++j)
      for (int i = 0; i < m.ni; ++i) {
        const bool left = (m.x1v(i) < x0);
        h.w(IDN, k, j, i) = left ? dl : dr;
        h.w(IVX, k, j, i) = h.w(IVY, k, j, i) = h.w(IVZ, k, j, i) = 0.0;
        h.w(IPR, k, j, i) = left ? pl : pr;
      }
}


// Uniform state (tests of gravity and cooling): rho, p, v, B constant.
inline void Uniform(Params &pin, const Mesh &m, HostFields &h) {
  const Real d = pin.GetOrAddReal("problem", "dens", 1.0), p = pin.GetOrAddReal("problem", "pres", 1.0);
  const Real vx = pin.GetOrAddReal("problem", "vx", 0.0), vy = pin.GetOrAddReal("problem", "vy", 0.0), vz = pin.GetOrAddReal("problem", "vz", 0.0);
  const Real bx = pin.GetOrAddReal("problem", "bx", 0.0), by = pin.GetOrAddReal("problem", "by", 0.0), bz = pin.GetOrAddReal("problem", "bz", 0.0);
  for (int k = 0; k < m.nk + 1; ++k)
    for (int j = 0; j < m.nj + 1; ++j)
      for (int i = 0; i < m.ni + 1; ++i) { h.b1(k, j, i) = bx; h.b2(k, j, i) = by; h.b3(k, j, i) = bz; }
  for (int k = 0; k < m.nk; ++k)
    for (int j = 0; j < m.nj; ++j)
      for (int i = 0; i < m.ni; ++i) {
        h.w(IDN, k, j, i) = d; h.w(IVX, k, j, i) = vx; h.w(IVY, k, j, i) = vy; h.w(IVZ, k, j, i) = vz; h.w(IPR, k, j, i) = p;
      }
}

// Decay of a sinusoidal perturbation along one axis under one diffusive process (exact
// solution amp * exp(-D k^2 t) for a small amplitude): var = "b" (resistivity, D = eta),
// "v" (viscosity, D = nu) or "temp" (conduction, D = kappa (gamma-1)/rho). The perturbed
// vector component is the one cyclically following the axis (axis 1 -> y, 2 -> z, 3 -> x).
inline void DiffusionTest(Params &pin, const Mesh &m, HostFields &h) {
  const std::string var = pin.GetOrAddString("problem", "var", "b");
  const int axis = pin.GetOrAddInt("problem", "axis", 1);
  const Real amp = pin.GetOrAddReal("problem", "amp", 1.0e-3);
  const Real d0 = pin.GetOrAddReal("problem", "dens", 1.0), p0 = pin.GetOrAddReal("problem", "pres", 1.0);
  const Real L = (axis == 1) ? (m.x1max - m.x1min) : (axis == 2) ? (m.x2max - m.x2min) : (m.x3max - m.x3min);
  const Real k = 2.0 * PI / L;
  const int comp = axis % 3;   // perturbed vector component index (0=x,1=y,2=z)
  auto coord_v = [&](int kk, int j, int i) { return (axis == 1) ? m.x1v(i) : (axis == 2) ? m.x2v(j) : m.x3v(kk); };
  for (int kk = 0; kk < m.nk + 1; ++kk)
    for (int j = 0; j < m.nj + 1; ++j)
      for (int i = 0; i < m.ni + 1; ++i) {
        h.b1(kk, j, i) = 0.0; h.b2(kk, j, i) = 0.0; h.b3(kk, j, i) = 0.0;
        if (var == "b") {
          // face-centered B_comp as a function of the axis coordinate at that face's center
          const Real xb = (axis == 1) ? m.x1v(i) : (axis == 2) ? m.x2v(j) : m.x3v(kk);
          const Real val = amp * sin(k * xb);
          if (comp == 0) h.b1(kk, j, i) = val; else if (comp == 1) h.b2(kk, j, i) = val; else h.b3(kk, j, i) = val;
        }
      }
  for (int kk = 0; kk < m.nk; ++kk)
    for (int j = 0; j < m.nj; ++j)
      for (int i = 0; i < m.ni; ++i) {
        const Real x = coord_v(kk, j, i);
        Real d = d0, p = p0, v[3] = {0.0, 0.0, 0.0};
        if (var == "v") v[comp] = amp * sin(k * x);
        if (var == "temp") { const Real T = p0 / d0 + amp * sin(k * x); d = p0 / T; }   // uniform pressure
        if (var == "b") p = p0 - 0.5 * SQR(amp * sin(k * x));                          // uniform total pressure
        h.w(IDN, kk, j, i) = d; h.w(IVX, kk, j, i) = v[0]; h.w(IVY, kk, j, i) = v[1]; h.w(IVZ, kk, j, i) = v[2];
        h.w(IPR, kk, j, i) = p;
      }
}

// Fill primitives + faces on host, convert to conserved and copy to the device.
inline void Initialize(const std::string &name, Params &pin, Mesh &m, const Config &cfg) {
  HostFields h;
  h.w = Kokkos::create_mirror_view(m.w);
  h.b1 = Kokkos::create_mirror_view(m.b1f0);
  h.b2 = Kokkos::create_mirror_view(m.b2f0);
  h.b3 = Kokkos::create_mirror_view(m.b3f0);

  if (name == "cpaw") CPAW(pin, m, h);
  else if (name == "orszag_tang") OrszagTang(pin, m, h);
  else if (name == "blast") Blast(pin, m, h);
  else if (name == "field_loop") FieldLoop(pin, m, h);
  else if (name == "shock_tube") ShockTube(pin, m, h);
  else if (name == "uniform") Uniform(pin, m, h);
  else if (name == "diffusion_test") DiffusionTest(pin, m, h);
  else throw std::runtime_error("unknown problem: " + name);

  // conserved variables from primitives and face fields
  auto hu = Kokkos::create_mirror_view(m.u0);
  for (int k = 0; k < m.nk; ++k)
    for (int j = 0; j < m.nj; ++j)
      for (int i = 0; i < m.ni; ++i) {
        const Real bx = 0.5 * (h.b1(k, j, i) + h.b1(k, j, i + 1));
        const Real by = m.two_d ? 0.5 * (h.b2(k, j, i) + h.b2(k, j + 1, i)) : h.b2(k, j, i);
        const Real bz = m.three_d ? 0.5 * (h.b3(k, j, i) + h.b3(k + 1, j, i)) : h.b3(k, j, i);
        const Real d = h.w(IDN, k, j, i), vx = h.w(IVX, k, j, i), vy = h.w(IVY, k, j, i), vz = h.w(IVZ, k, j, i);
        hu(IDN, k, j, i) = d;
        hu(IM1, k, j, i) = d * vx; hu(IM2, k, j, i) = d * vy; hu(IM3, k, j, i) = d * vz;
        hu(IEN, k, j, i) = h.w(IPR, k, j, i) / cfg.eos.gm1 + 0.5 * d * (vx * vx + vy * vy + vz * vz)
                           + 0.5 * (bx * bx + by * by + bz * bz);
      }
  Kokkos::deep_copy(m.u0, hu);
  Kokkos::deep_copy(m.b1f0, h.b1);
  Kokkos::deep_copy(m.b2f0, h.b2);
  Kokkos::deep_copy(m.b3f0, h.b3);
  Kokkos::deep_copy(m.w, h.w);
}

}  // namespace pgen
