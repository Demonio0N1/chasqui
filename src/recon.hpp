// Chasqui MHD — face reconstruction of a single variable.
//
// All routines take the 6-cell stencil q[0..5] = (q_{i-3}, ..., q_{i+2}) surrounding the
// face i-1/2 (between q[2] = cell i-1 and q[3] = cell i) and return
//   ql : value on the LEFT  side of the face (right-face value of cell i-1)
//   qr : value on the RIGHT side of the face (left-face value of cell i)
// Uniform grid only. DC/PLM use q[1..4]; PPM/WENO-Z use all six (needs ng >= 3).
#pragma once
#include "defs.hpp"

namespace recon {

// van Leer (harmonic-mean) limited slope
KOKKOS_INLINE_FUNCTION Real VLSlope(Real qm, Real q0, Real qp) {
  const Real dl = q0 - qm, dr = qp - q0;
  const Real d2 = dl * dr;
  return (d2 > Real(0)) ? Real(2) * d2 / (dl + dr) : Real(0);
}

// Colella-Woodward monotonized central slope (CW84 eq. 1.8)
KOKKOS_INLINE_FUNCTION Real MCSlope(Real qm, Real q0, Real qp) {
  const Real dl = q0 - qm, dr = qp - q0;
  if (dl * dr <= Real(0)) return Real(0);
  const Real dc = Real(0.5) * (qp - qm);
  const Real lim = Real(2) * fmin(fabs(dl), fabs(dr));
  return SIGN(dc) * fmin(fabs(dc), lim);
}

KOKKOS_INLINE_FUNCTION void DC(const Real q[6], Real &ql, Real &qr) {
  ql = q[2];
  qr = q[3];
}

KOKKOS_INLINE_FUNCTION void PLM(const Real q[6], Real &ql, Real &qr) {
  ql = q[2] + Real(0.5) * VLSlope(q[1], q[2], q[3]);
  qr = q[3] - Real(0.5) * VLSlope(q[2], q[3], q[4]);
}

// CW84 interface value between cells a (left) and b (right) using limited slopes:
//   q_{a+1/2} = q_a + (q_b - q_a)/2 - (dq_b - dq_a)/6
// (written in a mirror-symmetric form so that reflected data give bitwise-reflected values)
KOKKOS_INLINE_FUNCTION Real PPMInterface(Real qa, Real qb, Real dqa, Real dqb) {
  return Real(0.5) * (qa + qb) + (dqa - dqb) / Real(6);
}

// CW84 monotonization of the parabola in a cell with value q and interface values (qL,qR)
KOKKOS_INLINE_FUNCTION void PPMLimit(Real q, Real &qL, Real &qR) {
  if ((qR - q) * (q - qL) <= Real(0)) {
    qL = q; qR = q;
    return;
  }
  const Real d = qR - qL;
  const Real c = q - Real(0.5) * (qL + qR);
  const Real d2over6 = d * d / Real(6);
  if (d * c > d2over6)       qL = Real(3) * q - Real(2) * qR;
  else if (d * c < -d2over6) qR = Real(3) * q - Real(2) * qL;
}

// Piecewise parabolic (Colella & Woodward 1984) with monotonized slopes.
KOKKOS_INLINE_FUNCTION void PPM(const Real q[6], Real &ql, Real &qr) {
  // limited slopes in cells i-2 .. i+1 (q[1] .. q[4])
  const Real d1 = MCSlope(q[0], q[1], q[2]);
  const Real d2 = MCSlope(q[1], q[2], q[3]);
  const Real d3 = MCSlope(q[2], q[3], q[4]);
  const Real d4 = MCSlope(q[3], q[4], q[5]);
  // interface values at i-3/2, i-1/2, i+1/2
  const Real qm32 = PPMInterface(q[1], q[2], d1, d2);
  const Real qm12 = PPMInterface(q[2], q[3], d2, d3);
  const Real qp12 = PPMInterface(q[3], q[4], d3, d4);
  // cell i-1: parabola between qm32 and qm12 ; we need its right value
  Real aL = qm32, aR = qm12;
  PPMLimit(q[2], aL, aR);
  ql = aR;
  // cell i: parabola between qm12 and qp12 ; we need its left value
  Real bL = qm12, bR = qp12;
  PPMLimit(q[3], bL, bR);
  qr = bL;
}


// Extremum-preserving PPM (Colella & Sekora 2008; McCorquodale & Colella 2011): fourth-order
// interface values limited so that smooth extrema are not clipped, which keeps the full
// accuracy of the parabola on smooth flows while remaining non-oscillatory at shocks.
// Written in mirror-symmetric arithmetic.
KOKKOS_INLINE_FUNCTION Real PPMXInterface(Real qm1, Real q0, Real qp1, Real qp2) {
  // 4th-order interface value between q0 and qp1, then limit it if it is a local extremum
  const Real C = Real(1.25);
  Real qf = (Real(7) / Real(12)) * (q0 + qp1) - (Real(1) / Real(12)) * (qm1 + qp2);
  if ((qf - q0) * (qp1 - qf) < Real(0)) {
    const Real d2 = Real(3) * ((q0 + qp1) - Real(2) * qf);
    const Real d2l = (qm1 + qp1) - Real(2) * q0;
    const Real d2r = (q0 + qp2) - Real(2) * qp1;
    Real d2lim = Real(0);
    if (d2 * d2l > Real(0) && d2 * d2r > Real(0)) {
      d2lim = SIGN(d2) * fmin(fmin(C * fabs(d2l), C * fabs(d2r)), fabs(d2));
    }
    qf = Real(0.5) * (q0 + qp1) - d2lim / Real(6);
  }
  return qf;
}

// Limit the parabola of a cell with value q, neighbours (qm2, qm1, qp1, qp2) and interface
// values (qL, qR); modifies qL, qR in place.
KOKKOS_INLINE_FUNCTION void PPMXLimit(Real qm2, Real qm1, Real q, Real qp1, Real qp2, Real &qL, Real &qR) {
  const Real C = Real(1.25);
  const Real dqm = q - qL, dqp = qR - q;          // jumps to the two faces
  const bool extremum = (dqm * dqp <= Real(0)) || ((qp1 - q) * (q - qm1) <= Real(0));
  if (extremum) {
    const Real d2 = Real(6) * ((qL + qR) - Real(2) * q);   // second derivative of the parabola
    const Real d2c = (qm1 + qp1) - Real(2) * q;
    const Real d2l = (qm2 + q) - Real(2) * qm1;
    const Real d2r = (q + qp2) - Real(2) * qp1;
    Real d2lim = Real(0);
    if (d2 * d2c > Real(0) && d2 * d2l > Real(0) && d2 * d2r > Real(0)) {
      d2lim = SIGN(d2) * fmin(fmin(C * fabs(d2l), C * fabs(d2r)), fmin(C * fabs(d2c), fabs(d2)));
    }
    const Real rho = (fabs(d2) > Real(1e-12) * (fabs(q) + fabs(qm1) + fabs(qp1)) + Real(1e-300)) ? d2lim / d2 : Real(0);
    if (rho < Real(1) - Real(1e-12)) {
      qL = q - rho * dqm;
      qR = q + rho * dqp;
    }
  } else {
    // monotone cell: keep the parabola from overshooting
    if (fabs(dqm) >= Real(2) * fabs(dqp)) qL = q - Real(2) * dqp;
    if (fabs(dqp) >= Real(2) * fabs(dqm)) qR = q + Real(2) * dqm;
  }
}

KOKKOS_INLINE_FUNCTION void PPMX(const Real q[6], Real &ql, Real &qr) {
  // interface values at i-3/2, i-1/2, i+1/2
  const Real qm32 = PPMXInterface(q[0], q[1], q[2], q[3]);
  const Real qm12 = PPMXInterface(q[1], q[2], q[3], q[4]);
  const Real qp12 = PPMXInterface(q[2], q[3], q[4], q[5]);
  Real aL = qm32, aR = qm12;
  PPMXLimit(q[0], q[1], q[2], q[3], q[4], aL, aR);   // cell i-1
  ql = aR;
  Real bL = qm12, bR = qp12;
  PPMXLimit(q[1], q[2], q[3], q[4], q[5], bL, bR);   // cell i
  qr = bL;
}

// WENO-Z (Borges et al. 2008) right-interface value of the cell whose 5-point
// stencil is f[0..4] = (f_{-2}, f_{-1}, f_0, f_1, f_2).
KOKKOS_INLINE_FUNCTION Real WENOZRight(Real fm2, Real fm1, Real f0, Real fp1, Real fp2) {
#ifdef CHASQUI_SINGLE_PRECISION
  const Real eps = Real(1e-20);
#else
  const Real eps = Real(1e-36);
#endif
  const Real c13 = Real(13) / Real(12);
  // smoothness indicators in a mirror-symmetric arithmetic form
  const Real b0 = c13 * SQR((fm2 + f0) - Real(2) * fm1) + Real(0.25) * SQR((Real(3) * f0 + fm2) - Real(4) * fm1);
  const Real b1 = c13 * SQR((fm1 + fp1) - Real(2) * f0) + Real(0.25) * SQR(fm1 - fp1);
  const Real b2 = c13 * SQR((fp2 + f0) - Real(2) * fp1) + Real(0.25) * SQR((Real(3) * f0 + fp2) - Real(4) * fp1);
  const Real tau = fabs(b0 - b2);
  Real a0 = Real(0.1) * (Real(1) + tau / (b0 + eps));
  Real a1 = Real(0.6) * (Real(1) + tau / (b1 + eps));
  Real a2 = Real(0.3) * (Real(1) + tau / (b2 + eps));
  const Real inv = Real(1) / (a0 + a1 + a2);
  const Real q0 = (Real(2) * fm2 - Real(7) * fm1 + Real(11) * f0) / Real(6);
  const Real q1 = (-fm1 + Real(5) * f0 + Real(2) * fp1) / Real(6);
  const Real q2 = (Real(2) * f0 + Real(5) * fp1 - fp2) / Real(6);
  return inv * (a0 * q0 + a1 * q1 + a2 * q2);
}

KOKKOS_INLINE_FUNCTION void WENOZ(const Real q[6], Real &ql, Real &qr) {
  // left state: right interface of cell i-1 (stencil q[0..4])
  ql = WENOZRight(q[0], q[1], q[2], q[3], q[4]);
  // right state: left interface of cell i = mirrored right interface (stencil q[5..1])
  qr = WENOZRight(q[5], q[4], q[3], q[2], q[1]);
}


// Shock flattening (Colella & Woodward 1984, Appendix): a cell inside a strong compressive
// pressure jump gets its reconstructed face values pulled back toward the cell value.
// Inputs: pressures p[0..4] = (p_{j-2}, ..., p_{j+2}) and normal velocities u_{j-1}, u_{j+1}
// of the cell j. Returns the flattening coefficient in [0,1] (1 = donor cell).
// Written in mirror-symmetric arithmetic.
KOKKOS_INLINE_FUNCTION Real Flattening(const Real p[5], Real um1, Real up1) {
  const Real dp = p[3] - p[1];                 // p_{j+1} - p_{j-1}
  const Real dp2 = p[4] - p[0];                // p_{j+2} - p_{j-2}
  const Real pmin = fmin(p[1], p[3]);
  const bool strong = (fabs(dp) > Real(0.33) * pmin) && (um1 > up1);   // steep and compressive
  if (!strong) return Real(0);
  const Real ratio = (fabs(dp2) > Real(1e-300)) ? dp / dp2 : Real(0);
  return fmin(Real(1), fmax(Real(0), Real(10) * (ratio - Real(0.75))));
}

template <Recon R>
KOKKOS_INLINE_FUNCTION void Face(const Real q[6], Real &ql, Real &qr) {
  if constexpr (R == Recon::dc) DC(q, ql, qr);
  else if constexpr (R == Recon::plm) PLM(q, ql, qr);
  else if constexpr (R == Recon::ppm) PPM(q, ql, qr);
  else if constexpr (R == Recon::ppmx) PPMX(q, ql, qr);
  else WENOZ(q, ql, qr);
}

}  // namespace recon
