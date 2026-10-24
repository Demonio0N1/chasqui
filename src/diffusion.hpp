// Chasqui MHD — explicit non-ideal terms: viscosity, thermal conduction, Ohmic resistivity.
//
//   viscosity   : isotropic Navier-Stokes stress, dynamic viscosity mu = rho * nu
//   conduction  : heat flux q = -kappa grad T, with T = p / rho (code units)
//   resistivity : E = -v x B + eta J, J = curl B evaluated on cell edges from face fields
//                 (keeps div B = 0 exactly), plus the Poynting flux (eta J x B) in the
//                 energy equation.
// All three are added to the hyperbolic fluxes / edge EMFs of the same stage and advanced
// with the same Runge-Kutta scheme, under the parabolic time-step limit
//   dt <= cfl * dx^2 / (2 * ndim * max(nu, eta, chi)),  chi = kappa (gamma-1) / rho.
#pragma once
#include "defs.hpp"
#include "mesh.hpp"
#include "physics.hpp"

namespace diffusion {

struct Coeffs {
  Real nu = 0, kappa = 0, eta = 0;
  bool any() const { return nu > 0 || kappa > 0 || eta > 0; }
};

// Current density components on edges (from face-centered B). Index conventions follow
// mesh.hpp: J1 at (k-1/2, j-1/2, i), J2 at (k-1/2, j, i-1/2), J3 at (k, j-1/2, i-1/2).
struct CurlB {
  Arr3 b1, b2, b3;
  Real idx1, idx2, idx3;
  bool two_d, three_d;
  KOKKOS_INLINE_FUNCTION Real J1(int k, int j, int i) const {
    Real j1 = Real(0);
    if (two_d)   j1 += (b3(k, j, i) - b3(k, j - 1, i)) * idx2;
    if (three_d) j1 -= (b2(k, j, i) - b2(k - 1, j, i)) * idx3;
    return j1;
  }
  KOKKOS_INLINE_FUNCTION Real J2(int k, int j, int i) const {
    Real j2 = -(b3(k, j, i) - b3(k, j, i - 1)) * idx1;
    if (three_d) j2 += (b1(k, j, i) - b1(k - 1, j, i)) * idx3;
    return j2;
  }
  KOKKOS_INLINE_FUNCTION Real J3(int k, int j, int i) const {
    Real j3 = (b2(k, j, i) - b2(k, j, i - 1)) * idx1;
    if (two_d) j3 -= (b1(k, j, i) - b1(k, j - 1, i)) * idx2;
    return j3;
  }
};

// Everything a face needs to add its diffusive fluxes; built once per direction on the host.
template <int DIR>
struct FaceCtx {
  Arr4 w, bcc, flx;
  CurlB J;
  Real nu, kappa, eta;
  Real idn, idt1, idt2;
  bool has_t1, has_t2;
  int di, dj, dk, di1, dj1, dk1, di2, dj2, dk2, t1, t2;

  FaceCtx(const Mesh &m, const Coeffs &c, Arr3 b1c, Arr3 b2c, Arr3 b3c) {
    w = m.w; bcc = m.bcc;
    flx = (DIR == 0) ? m.f1 : (DIR == 1) ? m.f2 : m.f3;
    const Real idx[3] = {Real(1) / m.dx1, Real(1) / m.dx2, Real(1) / m.dx3};
    J = CurlB{b1c, b2c, b3c, idx[0], idx[1], idx[2], m.two_d, m.three_d};
    nu = c.nu; kappa = c.kappa; eta = c.eta;
    t1 = (DIR + 1) % 3; t2 = (DIR + 2) % 3;
    idn = idx[DIR]; idt1 = idx[t1]; idt2 = idx[t2];
    const bool has[3] = {true, m.two_d, m.three_d};
    has_t1 = has[t1]; has_t2 = has[t2];
    di = (DIR == 0); dj = (DIR == 1); dk = (DIR == 2);
    di1 = (t1 == 0); dj1 = (t1 == 1); dk1 = (t1 == 2);
    di2 = (t2 == 0); dj2 = (t2 == 1); dk2 = (t2 == 2);
  }

  // Add viscous, conductive and resistive-Poynting fluxes to the face (k,j,i).
  KOKKOS_INLINE_FUNCTION void Add(const int k, const int j, const int i) const {
    constexpr int IVU = IVX + DIR, IVV = IVX + (DIR + 1) % 3, IVW = IVX + (DIR + 2) % 3;
    const int kL = k - dk, jL = j - dj, iL = i - di;
    if (nu > Real(0)) {
      const Real mu = nu * Real(0.5) * (w(IDN, kL, jL, iL) + w(IDN, k, j, i));
      const Real dun = (w(IVU, k, j, i) - w(IVU, kL, jL, iL)) * idn;
      const Real dvn = (w(IVV, k, j, i) - w(IVV, kL, jL, iL)) * idn;
      const Real dwn = (w(IVW, k, j, i) - w(IVW, kL, jL, iL)) * idn;
      Real dut1 = 0, dvt1 = 0, dut2 = 0, dwt2 = 0;
      if (has_t1) {
        dut1 = Real(0.25) * idt1 * ((w(IVU, k + dk1, j + dj1, i + di1) - w(IVU, k - dk1, j - dj1, i - di1))
                                  + (w(IVU, kL + dk1, jL + dj1, iL + di1) - w(IVU, kL - dk1, jL - dj1, iL - di1)));
        dvt1 = Real(0.25) * idt1 * ((w(IVV, k + dk1, j + dj1, i + di1) - w(IVV, k - dk1, j - dj1, i - di1))
                                  + (w(IVV, kL + dk1, jL + dj1, iL + di1) - w(IVV, kL - dk1, jL - dj1, iL - di1)));
      }
      if (has_t2) {
        dut2 = Real(0.25) * idt2 * ((w(IVU, k + dk2, j + dj2, i + di2) - w(IVU, k - dk2, j - dj2, i - di2))
                                  + (w(IVU, kL + dk2, jL + dj2, iL + di2) - w(IVU, kL - dk2, jL - dj2, iL - di2)));
        dwt2 = Real(0.25) * idt2 * ((w(IVW, k + dk2, j + dj2, i + di2) - w(IVW, k - dk2, j - dj2, i - di2))
                                  + (w(IVW, kL + dk2, jL + dj2, iL + di2) - w(IVW, kL - dk2, jL - dj2, iL - di2)));
      }
      const Real divv = dun + dvt1 + dwt2;
      const Real tnn = mu * (Real(2) * dun - (Real(2) / Real(3)) * divv);
      const Real tnt1 = mu * (dvn + dut1);
      const Real tnt2 = mu * (dwn + dut2);
      const Real uf = Real(0.5) * (w(IVU, kL, jL, iL) + w(IVU, k, j, i));
      const Real vf = Real(0.5) * (w(IVV, kL, jL, iL) + w(IVV, k, j, i));
      const Real wf = Real(0.5) * (w(IVW, kL, jL, iL) + w(IVW, k, j, i));
      flx(IVU, k, j, i) -= tnn;
      flx(IVV, k, j, i) -= tnt1;
      flx(IVW, k, j, i) -= tnt2;
      flx(IEN, k, j, i) -= tnn * uf + tnt1 * vf + tnt2 * wf;
    }
    if (kappa > Real(0)) {
      const Real tL = w(IPR, kL, jL, iL) / w(IDN, kL, jL, iL), tR = w(IPR, k, j, i) / w(IDN, k, j, i);
      flx(IEN, k, j, i) -= kappa * (tR - tL) * idn;
    }
    if (eta > Real(0)) {
      // resistive EMF components (along t1, t2) at the face center = eta * J averaged from
      // the adjacent edges; then (E_res x B) . n = E_t1 B_t2 - E_t2 B_t1
      Real et1, et2;
      if (DIR == 0) {
        et1 = eta * Real(0.5) * (J.J2(k, j, i) + (has_t2 ? J.J2(k + 1, j, i) : J.J2(k, j, i)));
        et2 = eta * Real(0.5) * (J.J3(k, j, i) + (has_t1 ? J.J3(k, j + 1, i) : J.J3(k, j, i)));
      } else if (DIR == 1) {
        et1 = eta * Real(0.5) * (J.J3(k, j, i) + J.J3(k, j, i + 1));
        et2 = eta * Real(0.5) * (J.J1(k, j, i) + (has_t1 ? J.J1(k + 1, j, i) : J.J1(k, j, i)));
      } else {
        et1 = eta * Real(0.5) * (J.J1(k, j, i) + J.J1(k, j + 1, i));
        et2 = eta * Real(0.5) * (J.J2(k, j, i) + J.J2(k, j, i + 1));
      }
      const Real bt1 = Real(0.5) * (bcc(t1, kL, jL, iL) + bcc(t1, k, j, i));
      const Real bt2 = Real(0.5) * (bcc(t2, kL, jL, iL) + bcc(t2, k, j, i));
      flx(IEN, k, j, i) += et1 * bt2 - et2 * bt1;
    }
  }
};

// Add diffusive fluxes on all active faces of direction DIR.
template <int DIR>
void Fluxes(const Mesh &m, const Coeffs &c, Arr3 b1c, Arr3 b2c, Arr3 b3c) {
  if (!c.any()) return;
  const FaceCtx<DIR> ctx(m, c, b1c, b2c, b3c);
  const int iu = (DIR == 0) ? m.ie + 1 : m.ie, ju = (DIR == 1) ? m.je + 1 : m.je, ku = (DIR == 2) ? m.ke + 1 : m.ke;
  Kokkos::parallel_for("diff_flux", Range3({m.ks, m.js, m.is}, {ku + 1, ju + 1, iu + 1}),
    KOKKOS_LAMBDA(const int k, const int j, const int i) { ctx.Add(k, j, i); });
}

// Add eta * J to the edge-centered EMFs (after the ideal corner EMFs are known).
inline void ResistiveEMF(const Mesh &m, const Coeffs &c, Arr3 b1c, Arr3 b2c, Arr3 b3c) {
  if (c.eta <= Real(0)) return;
  const int is = m.is, ie = m.ie, js = m.js, je = m.je, ks = m.ks, ke = m.ke;
  const bool two_d = m.two_d, three_d = m.three_d;
  const Real eta = c.eta;
  auto e1 = m.e1, e2 = m.e2, e3 = m.e3;
  CurlB J{b1c, b2c, b3c, Real(1) / m.dx1, Real(1) / m.dx2, Real(1) / m.dx3, two_d, three_d};
  const int ku = three_d ? ke + 1 : ks + 1, ju = two_d ? je + 1 : js + 1;
  Kokkos::parallel_for("resistive_emf", Range3({ks, js, is}, {ku + 1, ju + 1, ie + 2}),
    KOKKOS_LAMBDA(const int k, const int j, const int i) {
      if (i <= ie) e1(k, j, i) += eta * J.J1(k, j, i);
      if (j <= je || !two_d) e2(k, j, i) += eta * J.J2(k, j, i);
      if (k <= ke || !three_d) e3(k, j, i) += eta * J.J3(k, j, i);
    });
}

// Parabolic time-step limit over the active cells.
inline Real TimeStep(const Mesh &m, const Config &cfg, const Coeffs &c) {
  if (!c.any()) return Real(1e300);
  const int ndim = 1 + (m.two_d ? 1 : 0) + (m.three_d ? 1 : 0);
  Real dxmin = m.dx1;
  if (m.two_d) dxmin = fmin(dxmin, m.dx2);
  if (m.three_d) dxmin = fmin(dxmin, m.dx3);
  const Real nu = c.nu, eta = c.eta, kappa = c.kappa, gm1 = cfg.eos.gm1;
  auto w = m.w;
  Real rho_min = Real(1e300);
  Kokkos::parallel_reduce("diff_dt", Range3({m.ks, m.js, m.is}, {m.ke + 1, m.je + 1, m.ie + 1}),
    KOKKOS_LAMBDA(const int k, const int j, const int i, Real &r) { r = fmin(r, w(IDN, k, j, i)); },
    Kokkos::Min<Real>(rho_min));
  const Real chi = (kappa > 0) ? kappa * gm1 / rho_min : Real(0);
  const Real dmax = fmax(fmax(nu, eta), chi);
  return dxmin * dxmin / (Real(2) * ndim * dmax);
}

}  // namespace diffusion
