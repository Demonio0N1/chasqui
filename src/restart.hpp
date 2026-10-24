// Chasqui MHD — binary restart dumps (conserved variables + face fields, all cells).
#pragma once
#include <string>
#include <fstream>
#include <stdexcept>
#include "defs.hpp"
#include "mesh.hpp"

namespace restart {

struct Header {
  int magic = 0x43485351;   // "CHSQ"
  int nx1, nx2, nx3, ng;
  int cycle;
  double time;
  int real_size;
};

inline void Write(const Mesh &m, const std::string &fname, Real time, int cycle) {
  Header h;
  h.nx1 = m.nx1; h.nx2 = m.nx2; h.nx3 = m.nx3; h.ng = m.ng; h.cycle = cycle; h.time = time;
  h.real_size = sizeof(Real);
  std::ofstream f(fname, std::ios::binary);
  f.write(reinterpret_cast<const char *>(&h), sizeof(h));
  auto hu = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), m.u0);
  auto hb1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), m.b1f0);
  auto hb2 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), m.b2f0);
  auto hb3 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), m.b3f0);
  f.write(reinterpret_cast<const char *>(hu.data()), hu.size() * sizeof(Real));
  f.write(reinterpret_cast<const char *>(hb1.data()), hb1.size() * sizeof(Real));
  f.write(reinterpret_cast<const char *>(hb2.data()), hb2.size() * sizeof(Real));
  f.write(reinterpret_cast<const char *>(hb3.data()), hb3.size() * sizeof(Real));
}

// Reads into u0 / b?f0; returns time and cycle through the arguments.
inline void Read(const Mesh &m, const std::string &fname, Real &time, int &cycle) {
  std::ifstream f(fname, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open restart file " + fname);
  Header h;
  f.read(reinterpret_cast<char *>(&h), sizeof(h));
  if (h.magic != 0x43485351) throw std::runtime_error("not a Chasqui restart file: " + fname);
  if (h.nx1 != m.nx1 || h.nx2 != m.nx2 || h.nx3 != m.nx3 || h.ng != m.ng || h.real_size != int(sizeof(Real)))
    throw std::runtime_error("restart file does not match the mesh (nx, nghost or precision differ)");
  auto hu = Kokkos::create_mirror_view(m.u0);
  auto hb1 = Kokkos::create_mirror_view(m.b1f0);
  auto hb2 = Kokkos::create_mirror_view(m.b2f0);
  auto hb3 = Kokkos::create_mirror_view(m.b3f0);
  f.read(reinterpret_cast<char *>(hu.data()), hu.size() * sizeof(Real));
  f.read(reinterpret_cast<char *>(hb1.data()), hb1.size() * sizeof(Real));
  f.read(reinterpret_cast<char *>(hb2.data()), hb2.size() * sizeof(Real));
  f.read(reinterpret_cast<char *>(hb3.data()), hb3.size() * sizeof(Real));
  Kokkos::deep_copy(m.u0, hu); Kokkos::deep_copy(m.b1f0, hb1);
  Kokkos::deep_copy(m.b2f0, hb2); Kokkos::deep_copy(m.b3f0, hb3);
  time = static_cast<Real>(h.time);
  cycle = h.cycle;
}

}  // namespace restart
