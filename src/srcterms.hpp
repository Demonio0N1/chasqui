// Chasqui MHD — source terms: gravity (uniform field or point mass) and optically thin
// cooling with a power-law cooling function integrated exactly (operator split).
#pragma once
#include "defs.hpp"
#include "mesh.hpp"
#include "physics.hpp"

namespace srcterms {

struct Gravity {
  bool uniform = false, point = false;
  Real g1 = 0, g2 = 0, g3 = 0;           // uniform acceleration
  Real gm = 0, x0 = 0, y0 = 0, z0 = 0;    // point mass: a = -GM r / (r^2 + r_soft^2)^{3/2}
  Real rsoft = 0;
  bool any() const { return uniform || point; }
};

// Momentum and energy sources from gravity, applied with the stage weight beta*dt using the
// stage primitives (same convention as Athena++: energy source = rho v . g).
inline void ApplyGravity(const Mesh &m, const Gravity &gr, Arr4 uout, Real beta_dt) {
  if (!gr.any()) return;
  const Geom g(m);
  const bool two_d = m.two_d, three_d = m.three_d;
  auto w = m.w;
  Kokkos::parallel_for("gravity", Range3({m.ks, m.js, m.is}, {m.ke + 1, m.je + 1, m.ie + 1}),
    KOKKOS_LAMBDA(const int k, const int j, const int i) {
      Real a1 = gr.g1, a2 = gr.g2, a3 = gr.g3;
      if (gr.point) {
        const Real dx = g.x1v(i) - gr.x0, dy = two_d ? g.x2v(j) - gr.y0 : Real(0), dz = three_d ? g.x3v(k) - gr.z0 : Real(0);
        const Real r2 = dx * dx + dy * dy + dz * dz + gr.rsoft * gr.rsoft;
        const Real f = -gr.gm / (r2 * sqrt(r2));
        a1 += f * dx; a2 += f * dy; a3 += f * dz;
      }
      const Real d = w(IDN, k, j, i);
      uout(IM1, k, j, i) += beta_dt * d * a1;
      uout(IM2, k, j, i) += beta_dt * d * a2;
      uout(IM3, k, j, i) += beta_dt * d * a3;
      uout(IEN, k, j, i) += beta_dt * d * (w(IVX, k, j, i) * a1 + w(IVY, k, j, i) * a2 + w(IVZ, k, j, i) * a3);
    });
}

struct Cooling {
  bool enabled = false;
  Real lambda0 = 0;   // Lambda(T) = lambda0 * T^alpha  (code units: dE/dt = -rho^2 Lambda)
  Real alpha = 1;
  Real tfloor = 0;
};

// Exact integration of dT/dt = -(gamma-1) rho lambda0 T^alpha over dt, applied to the
// conserved energy after the full step (operator splitting, first order in dt but with no
// stability limit on the cooling time).
inline void ApplyCooling(const Mesh &m, const EOS &eos, const Cooling &c, Arr4 u, Real dt) {
  if (!c.enabled) return;
  auto w = m.w;
  Kokkos::parallel_for("cooling", Range3({m.ks, m.js, m.is}, {m.ke + 1, m.je + 1, m.ie + 1}),
    KOKKOS_LAMBDA(const int k, const int j, const int i) {
      const Real d = w(IDN, k, j, i), p = w(IPR, k, j, i);
      const Real t0 = p / d;
      const Real cc = eos.gm1 * d * c.lambda0;
      Real t1;
      if (fabs(c.alpha - Real(1)) < Real(1e-12)) {
        t1 = t0 * exp(-cc * dt);
      } else {
        const Real e = Real(1) - c.alpha;
        const Real base = pow(t0, e) + (c.alpha - Real(1)) * cc * dt;
        t1 = (base > Real(0)) ? pow(base, Real(1) / e) : c.tfloor;
      }
      t1 = fmax(t1, c.tfloor);
      u(IEN, k, j, i) += d * (t1 - t0) / eos.gm1;
    });
}

}  // namespace srcterms
