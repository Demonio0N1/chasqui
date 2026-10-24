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
| Reconstrucción | `dc`, `plm` (van Leer), `ppm` (Colella-Woodward 1984), `ppmx` (PPM extremo-preservante, Colella & Sekora 2008 / McCorquodale & Colella 2011), `wenoz` (Borges 2008) | `ppm` |
| Solver de Riemann | `hlld` (Miyoshi & Kusano 2005, 7 ondas), `hlle` | `hlld` |
| Integración temporal | `rk1`, `rk2`, `rk3` (SSP), `rk4` = SSPRK(4,3) de Spiteri & Ruuth 2002 (3.º orden, estable hasta CFL 2× mayor, 4 etapas) | `rk3` |
| Aplanado de choques | `flatten` (Colella & Woodward 1984) para ppm/ppmx/wenoz | `true` |
| Bajo Mach | `low_mach` (Thornber et al. 2008): velocidades de cara mezcladas con z = min(1, M) | `false` |
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

## Pruebas

- `tests/convergence.sh`: órdenes de convergencia (abajo).
- `tests/physics.sh`: cada módulo físico contra su solución exacta.
- `ctest` (o `./build/symmetry_test`): simetría especular bit a bit de reconstrucción y
  solvers de Riemann.

## Validación (`tests/convergence.sh`)

Onda de Alfvén circularmente polarizada (solución exacta no lineal; vuelve al estado inicial
tras un periodo). Error L1 total tras T = 1, HLLD + RK3, CFL 0.4:

| 1D, N | PLM | PPM | PPMX | WENO-Z |
|---|---|---|---|---|
| 32  | 1.42e-2 | 1.38e-2 | 8.33e-5 | 3.93e-5 |
| 64  | 3.99e-3 | 3.20e-3 | 6.25e-6 | 4.21e-6 |
| 128 | 9.95e-4 | 6.65e-4 | 5.80e-7 | 5.03e-7 |
| 256 | 2.39e-4 | 1.46e-4 | 6.47e-8 | 6.22e-8 |
| orden | 2.0 | 2.2 | 3.2 | 3.0 (ambos limitados por RK3) |

2D oblicua (45°): PLM orden 2.0; WENO-Z orden 2.0 con error 16× menor (el promedio de
cara→centro de B y la EMF de esquina son de 2.º orden, como en Athena++). 3D: orden → 2.
∇·B ≤ 5e-13 y energía total constante a 1e-12 en todas las corridas (CPAW 3D, Orszag-Tang,
Brio-Wu, blast MHD 3D).

Vórtice de Gresho a bajo Mach (64², PPMX + HLLD, t = 1): energía cinética retenida
99.08 % a M = 0.01 sin corrección y 99.89 % con `low_mach = true` (igual que a M = 0.1), es
decir, la disipación deja de crecer al bajar el Mach.

## Física

| Módulo | Bloque del input | Validación (`tests/physics.sh`) |
|---|---|---|
| MHD ideal (3D, CT) | `[mhd]` | convergencia CPAW, Orszag-Tang, Brio-Wu, blast |
| Resistividad óhmica η (EMF = ηJ en aristas + calentamiento óhmico) | `[diffusion] eta` | decaimiento exacto `exp(-η k² t)` en x, y, z: error 3e-4 |
| Viscosidad ν (tensor Navier-Stokes isótropo, μ = ρν) | `[diffusion] nu` | decaimiento exacto en x, y, z: error 3e-4 |
| Conducción térmica κ (q = −κ∇T) | `[diffusion] kappa` | difusividad isóbara κ/(ρ c_p): error 2e-3 |
| Gravedad uniforme y de masa puntual (suavizada) | `[gravity]` | momento ρ g t V exacto |
| Enfriamiento ópticamente delgado Λ = Λ₀Tᵅ, integrado exactamente | `[cooling]` | T(t) exacta a 3e-8 |
| Fronteras: periódicas, *outflow*, reflectantes | `[mesh] bc_x1..3` | masa y energía conservadas a 1e-12 en caja cerrada |
| Corrección de flujo de primer orden (positividad) | `[mhd] fofc` | se activa solo en celdas con ρ o p bajo el piso |
| Reinicios binarios | `[output] rst_dt`, `-r fichero` | reanudación *bitwise* |

Los términos difusivos se suman a los flujos/EMF de la misma etapa RK (explícitos), con el
límite parabólico de paso de tiempo `dt ≤ cfl·dx²/(2·ndim·max(ν, η, χ))`.

Limitaciones conocidas:
- Las paredes reflectantes requieren B normal nulo en la pared (como en Athena++); con
  B_n ≠ 0 no son un espejo y no conservan masa.
- Simetría especular *bitwise*: toda la aritmética de PLM/PPM/WENO-Z, HLLE/HLLD y las EMF de
  esquina está escrita en forma simétrica (sumas agrupadas en pares que son imágenes
  especulares exactas, productos en el mismo orden a ambos lados, ramas con promedio en los
  empates), siguiendo Fleischmann, Adami & Adams (2019). `tests/symmetry_test.cpp` lo
  verifica con 200 000 estados aleatorios por kernel, y un blast MHD conserva su simetría a
  0.0 durante toda la corrida. Requiere compilar sin contracción FMA (`-ffp-contract=off`,
  el valor por defecto; `-D CHASQUI_ALLOW_FMA=ON` la reactiva a cambio de perder la simetría
  exacta). En GPU, `nvcc` contrae FMA por defecto: usar `-fmad=false` si se quiere la
  simetría exacta allí.
- Sin MPI, sin AMR, sin geometrías curvilíneas, sin EOS no ideales todavía.

## Problemas incluidos

`cpaw` (onda de Alfvén, cualquier ángulo en x-y), `orszag_tang`, `shock_tube` (Brio-Wu),
`field_loop`, `blast`, `uniform`, `diffusion_test`. Añadir uno nuevo = una función en
`src/pgen.hpp` que llene primitivas y campos de cara (hay un ayudante para inicializar B
desde un potencial vector, que garantiza ∇·B = 0 a nivel de redondeo).

## Hoja de ruta

- [ ] Medir en GPU (RTX 4090) contra AthenaK en el mismo problema.
- [ ] MPI (descomposición de dominio) y salida HDF5.
- [ ] EOS isoterma y general; conducción anisótropa (a lo largo de B).
- [ ] Reconstrucción en variables características (reduce las oscilaciones post-choque de
      PPMX/WENO-Z en Brio-Wu, que el aplanado solo recorta ~15 %).
- [ ] HLLD de baja disipación de Minoshima & Miyoshi (2021): quita la inestabilidad de choque
      multidimensional y mejora los flujos casi incompresibles sin recurrir al parche de Thornber.
- [ ] Formalismo de doble energía / interruptor de entropía para regiones de β muy bajo.
- [ ] Super-time-stepping RKL2 (Meyer, Balsara & Aslam 2014) para los términos difusivos.
- [ ] Limitador de positividad de Hu, Adams & Shu (2013) como alternativa al FOFC.
- [ ] Fronteras definidas por el usuario; términos fuente de usuario; escalares pasivos.
- [ ] Vectorización del bucle interno en CPU.

## Licencia

BSD 3-Clause.
