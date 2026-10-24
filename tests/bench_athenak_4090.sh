#!/usr/bin/env bash
# bench_athenak_4090.sh
# Compara en tu GPU tres compilaciones de AthenaK sobre el mismo problema (MHD 3D,
# onda lineal, PLM + HLLD + RK2, 256x128x128 = 4.2 M celdas, 200 pasos):
#   1) HEAD en doble precision
#   2) commit anterior al split recon/Riemann (#757, jul-2026) en doble precision
#   3) HEAD en precision simple
# y para la 1) dos tamaños de MeshBlock (32^3 vs bloque unico) para ver el costo de
# las celdas fantasma. Imprime zone-cycles/s de cada corrida al final.
#
# Requisitos: git, cmake >= 3.16, gcc, CUDA Toolkit 12.x (nvcc en el PATH).
# Uso:  bash bench_athenak_4090.sh          (tarda ~20-30 min la primera vez)
set -euo pipefail

ROOT="${ROOT:-$HOME/athenak-bench}"
NLIM="${NLIM:-200}"
ARCH="${ARCH:-ADA89}"          # RTX 4090 = sm_89. Para otra GPU: AMPERE86 (30xx), ADA89 (40xx)
PRESPLIT_REF="7f05899^"        # ultimo commit antes de "Split recon rsolver clean (#757)"

mkdir -p "$ROOT"; cd "$ROOT"
if [ ! -d athenak ]; then
  git clone --recursive https://github.com/IAS-Astrophysics/athenak.git
fi
git -C athenak fetch -q --unshallow 2>/dev/null || git -C athenak fetch -q

# ---------- input: onda lineal MHD 3D sin salidas pesadas ----------
cat > "$ROOT/linwave_bench.athinput" <<'EOF'
<comment>
problem   = mhd linear wave benchmark

<job>
basename  = LinWaveBench

<mesh>
nghost    = 2
nx1       = 256
x1min     = 0.0
x1max     = 3.0
ix1_bc    = periodic
ox1_bc    = periodic
nx2       = 128
x2min     = 0.0
x2max     = 1.5
ix2_bc    = periodic
ox2_bc    = periodic
nx3       = 128
x3min     = 0.0
x3max     = 1.5
ix3_bc    = periodic
ox3_bc    = periodic

<meshblock>
nx1       = 256
nx2       = 128
nx3       = 128

<time>
evolution  = dynamic
integrator = rk2
cfl_number = 0.3
nlim       = 200
tlim       = 100.0
ndiag      = 50

<mhd>
eos         = ideal
reconstruct = plm
rsolver     = hlld
gamma       = 1.66666666667

<problem>
pgen_name = linear_wave
wave_flag = 0
amp       = 1.0e-3
dens      = 1.0
pgas      = 0.6
vx0       = 0.0
vy0       = 0.0
vz0       = 0.0
bx0       = 1.0
by0       = 1.4142136
bz0       = 0.5
along_x1  = false
along_x2  = false
along_x3  = false

<output1>
file_type = hst
dt        = 1.0e10
EOF

# ---------- helpers ----------
checkout_src () {   # $1 = nombre, $2 = ref git
  local name=$1 ref=$2 dir="$ROOT/src-$name"
  if [ ! -d "$dir" ]; then
    git -C "$ROOT/athenak" worktree add "$dir" "$ref" >/dev/null
    git -C "$dir" submodule update --init --recursive -q
  fi
}

build () {          # $1 = nombre, $2... = flags extra de cmake
  local name=$1; shift
  local src="$ROOT/src-$name" bld="$ROOT/build-$name"
  if [ -x "$bld/src/athena" ]; then return; fi
  cmake -S "$src" -B "$bld" \
    -D CMAKE_BUILD_TYPE=Release \
    -D Kokkos_ENABLE_CUDA=ON -D "Kokkos_ARCH_${ARCH}=ON" \
    -D CMAKE_CXX_COMPILER="$src/kokkos/bin/nvcc_wrapper" \
    "$@" > "$bld.cmake.log" 2>&1
  cmake --build "$bld" -j"$(nproc)" > "$bld.make.log" 2>&1
}

run () {            # $1 = etiqueta, $2 = build, $3... = overrides del input
  local label=$1 name=$2; shift 2
  local out="$ROOT/run-$label"; mkdir -p "$out"
  ( cd "$out" && "$ROOT/build-$name/src/athena" -i "$ROOT/linwave_bench.athinput" \
      "time/nlim=$NLIM" "$@" > log.txt 2>&1 ) || { echo "  [$label] FALLO, ver $out/log.txt"; return; }
  local z; z=$(grep -o 'zone-cycles/cpu_second = [0-9.e+]*' "$out/log.txt" | tail -1 | awk '{print $3}')
  awk -v l="$label" -v z="$z" 'BEGIN{printf "  %-34s %12.3e zone-cycles/s  (%6.1f M)\n", l, z, z/1e6}'
}

# ---------- fuentes ----------
checkout_src head      HEAD
checkout_src presplit  "$PRESPLIT_REF"
cp -r "$ROOT/src-head" "$ROOT/src-head32" 2>/dev/null || true   # misma fuente, otro build

# ---------- compilaciones ----------
echo ">> compilando HEAD (FP64)...";      build head
echo ">> compilando pre-split (FP64)..."; build presplit
echo ">> compilando HEAD (FP32)...";      build head32 -D Athena_SINGLE_PRECISION=ON

# ---------- corridas ----------
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
echo ">> corriendo (${NLIM} pasos cada una, 256x128x128)..."
run "head-fp64-bloque-unico"      head
run "head-fp64-bloques-32^3"      head     meshblock/nx1=32 meshblock/nx2=32 meshblock/nx3=32
run "head-fp64-bloques-64^3"      head     meshblock/nx1=64 meshblock/nx2=64 meshblock/nx3=64
run "presplit-fp64-bloque-unico"  presplit
run "head-fp32-bloque-unico"      head32
echo
echo "Logs en $ROOT/run-*/log.txt. Para repetir solo las corridas: borra run-* y vuelve a ejecutar."
