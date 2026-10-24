// Chasqui MHD — outputs: legacy VTK (PLUTO-compatible layout) and a history file.
#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include "defs.hpp"
#include "mesh.hpp"

namespace output {

inline void WriteBigEndianFloats(std::ofstream &f, const std::vector<float> &v) {
  std::vector<unsigned char> buf(v.size() * 4);
  for (size_t n = 0; n < v.size(); ++n) {
    uint32_t u; std::memcpy(&u, &v[n], 4);
    buf[4 * n + 0] = (u >> 24) & 0xff; buf[4 * n + 1] = (u >> 16) & 0xff;
    buf[4 * n + 2] = (u >> 8) & 0xff;  buf[4 * n + 3] = u & 0xff;
  }
  f.write(reinterpret_cast<const char *>(buf.data()), buf.size());
}

inline void WriteBigEndianInt(std::ofstream &f, int32_t v) {
  unsigned char b[4] = {static_cast<unsigned char>((v >> 24) & 0xff), static_cast<unsigned char>((v >> 16) & 0xff),
                        static_cast<unsigned char>((v >> 8) & 0xff), static_cast<unsigned char>(v & 0xff)};
  f.write(reinterpret_cast<const char *>(b), 4);
}

// Writes rho, prs, vx1, vx2, vx3, Bx1, Bx2, Bx3 as cell-centered scalars, i fastest,
// in the same layout PLUTO uses for its .vtk dumps.
inline void WriteVTK(const Mesh &m, const std::string &fname, Real time, int cycle) {
  auto hw = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), m.w);
  auto hb = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), m.bcc);
  std::ofstream f(fname, std::ios::binary);
  const int nx1 = m.nx1, nx2 = m.nx2, nx3 = m.nx3;
  f << "# vtk DataFile Version 2.0\n";
  f << "Chasqui MHD VTK Data\n";
  f << "BINARY\n";
  f << "DATASET RECTILINEAR_GRID\n";
  f << "FIELD FieldData 2\n";
  f << "TIME 1 1 float\n";
  WriteBigEndianFloats(f, {static_cast<float>(time)});
  f << "\nCYCLE 1 1 int\n";
  WriteBigEndianInt(f, cycle);
  f << "\n";
  f << "DIMENSIONS " << nx1 + 1 << " " << (m.two_d ? nx2 + 1 : 1) << " " << (m.three_d ? nx3 + 1 : 1) << "\n";
  {
    std::vector<float> x(nx1 + 1);
    for (int i = 0; i <= nx1; ++i) x[i] = static_cast<float>(m.x1f(m.is + i));
    f << "X_COORDINATES " << nx1 + 1 << " float\n"; WriteBigEndianFloats(f, x); f << "\n";
  }
  {
    const int n = m.two_d ? nx2 + 1 : 1;
    std::vector<float> y(n);
    for (int j = 0; j < n; ++j) y[j] = static_cast<float>(m.two_d ? m.x2f(m.js + j) : m.x2min);
    f << "Y_COORDINATES " << n << " float\n"; WriteBigEndianFloats(f, y); f << "\n";
  }
  {
    const int n = m.three_d ? nx3 + 1 : 1;
    std::vector<float> z(n);
    for (int k = 0; k < n; ++k) z[k] = static_cast<float>(m.three_d ? m.x3f(m.ks + k) : m.x3min);
    f << "Z_COORDINATES " << n << " float\n"; WriteBigEndianFloats(f, z); f << "\n";
  }
  const size_t ncells = static_cast<size_t>(nx1) * nx2 * nx3;
  f << "CELL_DATA " << ncells << "\n";
  std::vector<float> buf(ncells);
  auto dump = [&](const char *name, auto getter) {
    size_t n = 0;
    for (int k = m.ks; k <= m.ke; ++k)
      for (int j = m.js; j <= m.je; ++j)
        for (int i = m.is; i <= m.ie; ++i) buf[n++] = static_cast<float>(getter(k, j, i));
    f << "SCALARS " << name << " float\nLOOKUP_TABLE default\n";
    WriteBigEndianFloats(f, buf);
    f << "\n";
  };
  dump("rho", [&](int k, int j, int i) { return hw(IDN, k, j, i); });
  dump("prs", [&](int k, int j, int i) { return hw(IPR, k, j, i); });
  dump("vx1", [&](int k, int j, int i) { return hw(IVX, k, j, i); });
  dump("vx2", [&](int k, int j, int i) { return hw(IVY, k, j, i); });
  dump("vx3", [&](int k, int j, int i) { return hw(IVZ, k, j, i); });
  dump("Bx1", [&](int k, int j, int i) { return hb(IBX, k, j, i); });
  dump("Bx2", [&](int k, int j, int i) { return hb(IBY, k, j, i); });
  dump("Bx3", [&](int k, int j, int i) { return hb(IBZ, k, j, i); });
}

struct HistoryData {
  Real mass, mom1, mom2, mom3, etot, ekin, emag, eint, divb_max, divb_l1;
};

inline HistoryData History(const Mesh &m, Arr4 u, Arr3 b1, Arr3 b2, Arr3 b3) {
  const int is = m.is, ie = m.ie, js = m.js, je = m.je, ks = m.ks, ke = m.ke;
  const bool two_d = m.two_d, three_d = m.three_d;
  const Real idx1 = Real(1) / m.dx1, idx2 = Real(1) / m.dx2, idx3 = Real(1) / m.dx3;
  auto w = m.w, bcc = m.bcc;
  Real s[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  Real dbmax = 0, dbl1 = 0;
  Kokkos::parallel_reduce("hst", Range3({ks, js, is}, {ke + 1, je + 1, ie + 1}),
    KOKKOS_LAMBDA(const int k, const int j, const int i, Real &m0, Real &m1, Real &m2, Real &m3,
                  Real &et, Real &ek, Real &em, Real &ei, Real &dbm, Real &dbs) {
      m0 += u(IDN, k, j, i); m1 += u(IM1, k, j, i); m2 += u(IM2, k, j, i); m3 += u(IM3, k, j, i);
      et += u(IEN, k, j, i);
      const Real d = w(IDN, k, j, i);
      ek += Real(0.5) * d * (SQR(w(IVX, k, j, i)) + SQR(w(IVY, k, j, i)) + SQR(w(IVZ, k, j, i)));
      em += Real(0.5) * (SQR(bcc(0, k, j, i)) + SQR(bcc(1, k, j, i)) + SQR(bcc(2, k, j, i)));
      ei += u(IEN, k, j, i) - Real(0.5) * d * (SQR(w(IVX, k, j, i)) + SQR(w(IVY, k, j, i)) + SQR(w(IVZ, k, j, i)))
            - Real(0.5) * (SQR(bcc(0, k, j, i)) + SQR(bcc(1, k, j, i)) + SQR(bcc(2, k, j, i)));
      Real db = (b1(k, j, i + 1) - b1(k, j, i)) * idx1;
      if (two_d)   db += (b2(k, j + 1, i) - b2(k, j, i)) * idx2;
      if (three_d) db += (b3(k + 1, j, i) - b3(k, j, i)) * idx3;
      dbm = fmax(dbm, fabs(db));
      dbs += fabs(db);
    },
    s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], Kokkos::Max<Real>(dbmax), dbl1);
  const Real vol = m.dx1 * (two_d ? m.dx2 : Real(1)) * (three_d ? m.dx3 : Real(1));
  const Real ncell = Real(m.nx1) * m.nx2 * m.nx3;
  return {s[0] * vol, s[1] * vol, s[2] * vol, s[3] * vol, s[4] * vol, s[5] * vol, s[6] * vol, s[7] * vol,
          dbmax, dbl1 / ncell};
}

}  // namespace output
