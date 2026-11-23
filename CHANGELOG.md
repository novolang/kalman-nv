# Changelog

All notable changes to kalman-nv are recorded here. The format is
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
package follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
with the pre-1.0 rule that a breaking change bumps the MINOR number.

## 0.1.0 — 2026-11-23

### Added

- **`kfmat`** — the vector `KVec` and the matrix `Mat` over inline storage
  of at most six rows and columns, in single precision: construction,
  sum, difference, scale, product, product with a transpose, transpose,
  matrix-vector products, trace, the symmetry check and the symmetric
  average, the `L D Lᵀ` factorisation, the positive-definiteness test and
  the solve for a symmetric positive-definite matrix. The host-only
  `vec_of` and `from_list` take a list and carry `@tier(app)`.
- **`kalman`** — the filter `Kf`, the linear prediction with and without a
  control input, the update with the Joseph form of the covariance, the
  scalar update, the sequential update of independent measurements, the
  innovation with its covariance and its normalised squared value, the
  gate, and the five fault codes.
- **`ekf`** — the extended filter's prediction, update, scalar update and
  innovation, taking the model's value and Jacobian from the caller.
- **`tests/embedded_probe.nv`** and **`tests/device_build.sh`** — the device
  claim as a program, built for the nRF52840-DK, the nRF54LM20-DK and the
  emulated board, with the image sizes beside a baseline and a QEMU run.
- **`tests/fixtures/kalman_reference.py`** — the float64 numpy reference
  that writes `tests/kalman_vector_tests.nv`.

### Known

- The capacity, `kfmat.MAX_DIM = 6`, is a constant of the package and not a
  configuration key. A key reaches a package through a generated module,
  and the shard audit's device build stages the package's sources without
  one, so a `Vec` sized by a key is refused there with E2920.
- `linalg-nv` is not a dependency: its matrix is a heap value in `Float`,
  and a device build of a program that factors one 2-by-2 array is refused
  with 505 tier errors.
- The embedded build of `kfmat.asymmetry` converts one binary32 value to
  binary64 and back, which links two routines from libgcc.
