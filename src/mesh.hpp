// Chasqui MHD — uniform Cartesian mesh (single block with ghost zones) and field storage.
//
// Index conventions (all arrays include ghost zones):
//   cells:   i in [0, ni), active cells in [is, ie] with is = ng, ie = ng + nx1 - 1
//   faces:   face i sits at x_{i-1/2}, i.e. the LEFT face of cell i; faces of the
//            active cells are i in [is, ie+1]
//   edges:   edge arrays follow the same "left/lower" convention in both transverse dirs.
// Face/edge arrays are allocated with one extra entry in every direction so that a
// single index convention serves cells, faces and edges.
#pragma once
#include "defs.hpp"

struct Mesh {
  int nx1, nx2, nx3;       // active cells
  int ng;                  // ghost cells
  int ni, nj, nk;          // total cells including ghosts
  int is, ie, js, je, ks, ke;
  Real x1min, x1max, x2min, x2max, x3min, x3max;
  Real dx1, dx2, dx3;
  bool two_d, three_d;

  // conserved variables (two buffers for multi-stage RK without copies)
  Arr4 u0, u1;
  // primitives, cell-centered B, cell-centered EMF E = -(v x B)
  Arr4 w, bcc, ecc;
  // face-centered B: b1f(k,j,i) at x_{i-1/2}; b2f at y_{j-1/2}; b3f at z_{k-1/2}
  Arr3 b1f0, b2f0, b3f0, b1f1, b2f1, b3f1;
  // fluxes of conserved variables through faces
  Arr4 f1, f2, f3;
  // area-averaged EMFs at faces (named e<comp>x<facedir>)
  Arr3 e2x1, e3x1, e1x2, e3x2, e1x3, e2x3;
  // edge-centered EMFs used by CT
  Arr3 e1, e2, e3;

  void Allocate() {
    ni = nx1 + 2 * ng;
    nj = (nx2 > 1) ? nx2 + 2 * ng : 1;
    nk = (nx3 > 1) ? nx3 + 2 * ng : 1;
    is = ng; ie = ng + nx1 - 1;
    js = (nx2 > 1) ? ng : 0; je = (nx2 > 1) ? ng + nx2 - 1 : 0;
    ks = (nx3 > 1) ? ng : 0; ke = (nx3 > 1) ? ng + nx3 - 1 : 0;
    two_d = (nx2 > 1);
    three_d = (nx3 > 1);
    dx1 = (x1max - x1min) / nx1;
    dx2 = (x2max - x2min) / nx2;
    dx3 = (x3max - x3min) / nx3;
    const int nip = ni + 1, njp = nj + 1, nkp = nk + 1;
    u0 = Arr4("u0", NVAR, nk, nj, ni);
    u1 = Arr4("u1", NVAR, nk, nj, ni);
    w = Arr4("w", NVAR, nk, nj, ni);
    bcc = Arr4("bcc", 3, nk, nj, ni);
    ecc = Arr4("ecc", 3, nk, nj, ni);
    b1f0 = Arr3("b1f0", nkp, njp, nip); b2f0 = Arr3("b2f0", nkp, njp, nip); b3f0 = Arr3("b3f0", nkp, njp, nip);
    b1f1 = Arr3("b1f1", nkp, njp, nip); b2f1 = Arr3("b2f1", nkp, njp, nip); b3f1 = Arr3("b3f1", nkp, njp, nip);
    f1 = Arr4("f1", NVAR, nkp, njp, nip); f2 = Arr4("f2", NVAR, nkp, njp, nip); f3 = Arr4("f3", NVAR, nkp, njp, nip);
    e2x1 = Arr3("e2x1", nkp, njp, nip); e3x1 = Arr3("e3x1", nkp, njp, nip);
    e1x2 = Arr3("e1x2", nkp, njp, nip); e3x2 = Arr3("e3x2", nkp, njp, nip);
    e1x3 = Arr3("e1x3", nkp, njp, nip); e2x3 = Arr3("e2x3", nkp, njp, nip);
    e1 = Arr3("e1", nkp, njp, nip); e2 = Arr3("e2", nkp, njp, nip); e3 = Arr3("e3", nkp, njp, nip);
  }

  // cell-center and face coordinates
  KOKKOS_INLINE_FUNCTION Real x1v(int i) const { return x1min + (i - is + Real(0.5)) * dx1; }
  KOKKOS_INLINE_FUNCTION Real x2v(int j) const { return x2min + (j - js + Real(0.5)) * dx2; }
  KOKKOS_INLINE_FUNCTION Real x3v(int k) const { return x3min + (k - ks + Real(0.5)) * dx3; }
  KOKKOS_INLINE_FUNCTION Real x1f(int i) const { return x1min + (i - is) * dx1; }
  KOKKOS_INLINE_FUNCTION Real x2f(int j) const { return x2min + (j - js) * dx2; }
  KOKKOS_INLINE_FUNCTION Real x3f(int k) const { return x3min + (k - ks) * dx3; }
};

// Small POD copy of the geometry for use inside kernels (Views are captured separately).
struct Geom {
  int is, ie, js, je, ks, ke, ni, nj, nk;
  Real dx1, dx2, dx3, x1min, x2min, x3min;
  bool two_d, three_d;
  Geom() = default;
  explicit Geom(const Mesh &m)
      : is(m.is), ie(m.ie), js(m.js), je(m.je), ks(m.ks), ke(m.ke), ni(m.ni), nj(m.nj), nk(m.nk),
        dx1(m.dx1), dx2(m.dx2), dx3(m.dx3), x1min(m.x1min), x2min(m.x2min), x3min(m.x3min),
        two_d(m.two_d), three_d(m.three_d) {}
  KOKKOS_INLINE_FUNCTION Real x1v(int i) const { return x1min + (i - is + Real(0.5)) * dx1; }
  KOKKOS_INLINE_FUNCTION Real x2v(int j) const { return x2min + (j - js + Real(0.5)) * dx2; }
  KOKKOS_INLINE_FUNCTION Real x3v(int k) const { return x3min + (k - ks + Real(0.5)) * dx3; }
  KOKKOS_INLINE_FUNCTION Real x1f(int i) const { return x1min + (i - is) * dx1; }
  KOKKOS_INLINE_FUNCTION Real x2f(int j) const { return x2min + (j - js) * dx2; }
  KOKKOS_INLINE_FUNCTION Real x3f(int k) const { return x3min + (k - ks) * dx3; }
};
