# Chasqui MHD

Un código de magnetohidrodinámica ideal (3D, volúmenes finitos, transporte restringido)
escrito sobre [Kokkos](https://github.com/kokkos/kokkos): el mismo fuente compila para CPU
(OpenMP), GPUs NVIDIA (CUDA), AMD (HIP) e Intel (SYCL).

Chasqui nace de una revisión del camino MHD de AthenaK y hereda las decisiones correctas de
Athena++/AthenaK y PLUTO (bloques en caché, CT con promedio *upwind*, HLLD, RK3 SSP),
pero reorganiza los kernels para mover el mínimo de bytes por celda, que es lo que manda
en un código limitado por ancho de banda de memoria.

## Esquema numérico

| Componente | Opciones | Por defecto |
|---|---|---|
| Reconstrucción | `dc`, `plm` (van Leer), `ppm` (Colella-Woodward), `wenoz` (Borges 2008, 5.º orden) | `ppm` |
| Solver de Riemann | `hlld` (Miyoshi & Kusano 2005, 7 ondas), `hlle` | `hlld` |
| Integración temporal | `rk1`, `rk2`, `rk3` (SSP) | `rk3` |
| ∇·B | Transporte restringido, EMF de esquina con promedio *upwind* (Gardiner & Stone 2005) | — |
| Precisión | `double` (defecto) o `float` (`-D CHASQUI_SINGLE_PRECISION=ON`) | `double` |

Unidades: presión magnética = B²/2 (μ₀ = 1), las mismas que Athena y PLUTO.

### Diseño de los kernels (una etapa RK = 5 lanzamientos)

1. **Flujos fusionados.** Por cada cara se reconstruyen los dos lados a partir de las
   primitivas (las lecturas del *stencil* caen en caché), se resuelve el problema de Riemann
   en registros y se escriben el flujo y las dos componentes de EMF de esa cara. Los estados
   L/R nunca pasan por memoria global.
2. **EMF de esquina** (promedio *upwind* de las EMF de cara usando el signo del flujo de masa).
3. **Actualización**: combinación RK + divergencia de flujos para U, y CT para B, en un lanzamiento.
4. **Fronteras** (periódicas / *outflow*).
5. **Con2prim** + B centrado + EMF centrada + paso de tiempo (una pasada, una reducción).

Las etapas RK alternan dos *buffers* (u0/u1, b0/b1) sin copias.

Tráfico aproximado de DRAM por celda y etapa en 3D: ~130 doubles (≈1 KB), frente a
≈250 doubles del camino MHD de AthenaK (sept. 2026), donde los estados L/R se materializan
en memoria global.

## Compilar

```bash
git clone --recursive https://github.com/<usuario>/chasqui.git
cd chasqui
# CPU (OpenMP)
cmake -S . -B build -D Kokkos_ENABLE_OPENMP=ON
cmake --build build -j
# GPU NVIDIA (RTX 4090 = ADA89; A100 = AMPERE80; H100 = HOPPER90)
cmake -S . -B build-cuda -D Kokkos_ENABLE_CUDA=ON -D Kokkos_ARCH_ADA89=ON \
      -D CMAKE_CXX_COMPILER=$PWD/external/kokkos/bin/nvcc_wrapper
cmake --build build-cuda -j
```

Si ya tienes Kokkos instalado: `-D Kokkos_ROOT=/ruta/a/kokkos-install` y no hace falta el submódulo.

## Correr

```bash
./build/chasqui -i inputs/orszag_tang.in            # parámetros del fichero
./build/chasqui -i inputs/cpaw_2d.in mesh.nx1=256 mesh.nx2=256 mhd.reconstruct=wenoz
```

Cualquier parámetro `bloque.clave=valor` de la línea de comandos sobrescribe al fichero.

Salidas: `basename.NNNN.vtk` (VTK *legacy*, malla rectilínea, datos de celda `rho prs vx1 vx2
vx3 Bx1 Bx2 Bx3`, mismo diseño que PLUTO, así que los lectores de PLUTO funcionan) y
`basename.hst` (masa, momento, energías, máx|∇·B|).

## Validación (`tests/convergence.sh`)

Onda de Alfvén circularmente polarizada (solución exacta no lineal; vuelve al estado inicial
tras un periodo). Error L1 total tras T = 1, HLLD + RK3, CFL 0.4:

| 1D, N | PLM | PPM | WENO-Z |
|---|---|---|---|
| 32  | 1.42e-2 | 1.38e-2 | 3.93e-5 |
| 64  | 3.99e-3 | 3.20e-3 | 4.21e-6 |
| 128 | 9.95e-4 | 6.65e-4 | 5.03e-7 |
| 256 | 2.39e-4 | 1.46e-4 | 6.22e-8 |
| orden | 2.0 | 2.2 | 3.0 (limitado por RK3) |

2D oblicua (45°): PLM orden 2.0; WENO-Z orden 2.0 con error 16× menor (el promedio de
cara→centro de B y la EMF de esquina son de 2.º orden, como en Athena++). 3D: orden → 2.
∇·B ≤ 5e-13 y energía total constante a 1e-12 en todas las corridas (CPAW 3D, Orszag-Tang,
Brio-Wu, blast MHD 3D).

## Problemas incluidos

`cpaw` (onda de Alfvén, cualquier ángulo en x-y), `orszag_tang`, `brio_wu`/`shock_tube`,
`field_loop`, `blast`. Añadir uno nuevo = una función en `src/pgen.hpp` que llene
primitivas y campos de cara (hay un ayudante para inicializar B desde un potencial vector,
que garantiza ∇·B = 0 a nivel de redondeo).

## Hoja de ruta

- [ ] Medir en GPU (RTX 4090) contra AthenaK en el mismo problema.
- [ ] Reconstrucción característica para WENO-Z (menos oscilaciones en choques fuertes).
- [ ] Límite extremo-preservante para PPM (Colella & Sekora 2008).
- [ ] Corrección de flujo de primer orden como red de seguridad.
- [ ] MPI (descomposición de dominio) y salida HDF5.
- [ ] Fronteras reflectantes y definidas por el usuario; términos fuente.

## Licencia

BSD 3-Clause.
