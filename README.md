# kalman-nv

A **Kalman filter** estimates the state of a system from measurements that
are noisy and arrive one at a time. It keeps two things: the best estimate of
the state, and a covariance matrix that says how uncertain that estimate is.
Each step has two halves. The prediction moves both forward through a model
of how the system behaves. The update takes one measurement, weighs it
against the prediction by their covariances, and pulls the estimate toward
it. The **extended Kalman filter** applies the same steps to a nonlinear
model by linearising it at the current estimate. The filter is R. E. Kalman's
(1960); the treatment followed here is Bar-Shalom, Li and Kirubarajan,
*Estimation with Applications to Tracking and Navigation* (2001), and Dan
Simon, *Optimal State Estimation* (2006).

This package is both filters in single precision, over matrices whose
capacity is fixed when the package is built, with no allocation. It is for a
microcontroller that fuses a gyroscope with wheel odometry, tracks a target
from a range sensor, or estimates a battery's charge, and it runs unchanged
on a host.

## What the vocabulary means

The **state** `x` is the vector of quantities being estimated: a position and
a velocity, or a pose. The **covariance** `P` is a symmetric matrix whose
diagonal holds the variance of each state and whose other entries say how the
errors of two states move together.

The **process model** `F` says how the state moves in one step, and `Q` is the
covariance of what the model leaves out. The **measurement model** `H` says
what a sensor reads given the state, and `R` is the sensor's noise
covariance.

The **innovation** `y` is the measurement less what the filter predicted it
would read. Its covariance is `S = H P Hᵀ + R`. The **gain** `K = P Hᵀ S⁻¹` is
how far one unit of innovation moves each state. The **normalised innovation
squared**, `yᵀ S⁻¹ y`, measures how surprising a measurement is, in units of
how surprising the filter expects measurements to be. For a Gaussian
measurement it follows a chi-square law with one degree of freedom for each
measured quantity, so comparing it with a quantile of that law rejects an
outlier. That comparison is a **gate**.

The **Joseph form** of the covariance update, `P = (I − K H) P (I − K H)ᵀ +
K R Kᵀ`, equals the short form `(I − K H) P` in exact arithmetic. The short
form is a difference of nearly equal matrices; the Joseph form is a sum of two
positive semi-definite matrices, so rounding cannot make it indefinite.

| Quantity | Value |
| --- | --- |
| Number format | single precision (`fp32`) |
| Largest state or measurement | 6 (`kfmat.MAX_DIM`) |
| Storage of one matrix | 168 bytes, in the caller's frame |
| Storage of a filter of any size up to the capacity | 448 bytes |
| Allocations | none |
| Matrix inverses taken | at most the size of the measurement; none for a scalar update |

## Install

```
novo pkg add kalman-nv
```

## Example

A two-wheeled robot knows how fast it drives and turns from its wheel
encoders, and how fast it turns from a gyroscope. The wheels over-read the
turn by ten per cent because they slip; the gyroscope is accurate. The filter
estimates the pose, the speed and the turn rate, and weighs each sensor by
its variance, so the turn rate settles on the gyroscope's value and the pose
follows it.

```novo
use kalman
use ekf
use kfmat

// The state is x, y, heading, speed and turn rate. The sensors never
// measure the pose: they measure the speed and the turn rate, and the
// filter integrates those into the pose.
const DT = 0.1

// The motion model at the estimate: the pose advances along the heading
// by the speed, and the heading by the turn rate.
fn motion(s: KVec) -> KVec
    let th = s.a[2] as Float
    kfmat.vec_of([
        (s.a[0] as Float) + (s.a[3] as Float) * math.cos(th) * DT,
        (s.a[1] as Float) + (s.a[3] as Float) * math.sin(th) * DT,
        th + (s.a[4] as Float) * DT,
        s.a[3] as Float,
        s.a[4] as Float
    ])

// Its Jacobian at the same estimate.
fn motion_jacobian(s: KVec) -> Mat
    let th = s.a[2] as Float
    let v = s.a[3] as Float
    var f = kfmat.identity(5)
    f = kfmat.set(f, 0, 2, (0.0 - v * math.sin(th) * DT) as fp32)
    f = kfmat.set(f, 0, 3, (math.cos(th) * DT) as fp32)
    f = kfmat.set(f, 1, 2, (v * math.cos(th) * DT) as fp32)
    f = kfmat.set(f, 1, 3, (math.sin(th) * DT) as fp32)
    f = kfmat.set(f, 2, 4, DT as fp32)
    f

// A scalar measurement, applied when it passes the gate. 6.63 is the
// 99 per cent quantile of the chi-square law with one degree of freedom.
fn fuse(k: Kf, z: fp32, row: KVec, variance: fp32) -> Kf
    let inn = kalman.innovate_scalar(k, z, row, variance)
    if kalman.gate(inn, 6.63)
        return kalman.update_scalar(k, z, row, variance)
    k

fn main() [io]
    var k = kalman.filter(kfmat.vec_of([0.0, 0.0, 0.0, 0.0, 0.0]), kfmat.diagonal(5, 1.0))
    let q = kfmat.from_list(5, 5, [
        0.0001, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0001, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0001, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.01, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.01
    ])
    let speed_row = kfmat.vec_of([0.0, 0.0, 0.0, 1.0, 0.0])
    let turn_row = kfmat.vec_of([0.0, 0.0, 0.0, 0.0, 1.0])
    for step in 0..100
        // Predict with the model's value and Jacobian at the estimate.
        k = ekf.predict(k, motion(kalman.state(k)), motion_jacobian(kalman.state(k)), q)
        // The wheels: 0.5 m/s, and a turn of 0.33 rad/s that reads ten per
        // cent high.
        k = fuse(k, 0.5, speed_row, 0.0025)
        k = fuse(k, 0.33, turn_row, 0.04)
        // The gyroscope: 0.30 rad/s, with a much smaller variance.
        k = fuse(k, 0.3, turn_row, 0.0004)
    let s = kalman.state(k)
    println("x ${s.a[0]} y ${s.a[1]} heading ${s.a[2]} turn rate ${s.a[4]}")
    println("fault ${kalman.fault(k)}")
```

After ten seconds the turn rate is 0.3003 rad/s and the heading is 3.0027
rad, where the wheels alone would give 3.3. The pose is the point on the
circle of radius 1.67 m that a heading of three radians reaches.

The example takes the wheel speed and turn rate as plain numbers. A program
that uses [odometry-nv](https://novo-lang.org/packages/odometry-nv) reads them
from the `twist` of the `Odometer` that `odometry.update` answers, as `v` and
`w`, and this package does not depend on it: a filter that fuses a twist needs
no knowledge of how the twist was measured, and an odometer needs none of the
filter. A program on a microcontroller replaces `math.sin` and `math.cos` with
the single-precision ones of [fpmath-nv](https://novo-lang.org/packages/fpmath-nv)
(`fpmath.sin`, `fpmath.cos`), which ahrs-nv and odometry-nv take as well, since
the embedded tier refuses a call into the host's maths library.

Build and test with `novo pkg build` and `novo test`.

## What the package contains

| Module | Contents |
| --- | --- |
| `kfmat` | The vector `KVec` and the matrix `Mat`, with sum, difference, product, product with a transpose, transpose, trace, the symmetry and positive-definiteness checks, and a solve for a symmetric positive-definite matrix. |
| `kalman` | The filter `Kf`, the linear prediction and update, the scalar update, the sequential update of independent measurements, the innovation and its gate, and the fault codes. |
| `ekf` | The extended filter: the prediction and the update with the model's value and Jacobian supplied by the caller. |

## How to choose an entry point

**`kalman` when the model is linear.** `predict(k, f, q)` and `update(k, z, h,
r)` take the matrices `F`, `Q`, `H` and `R`.

**`ekf` when it is not.** The caller evaluates the model's function and its
Jacobian at the estimate and passes both. The filter takes the values, never
a function: a function value that captured anything would allocate, and
evaluating the model where the model lives keeps this package a function of
numbers.

**`update_scalar` when the measurements are independent.** A vector of
measurements with uncorrelated noise carries the same information as its
components taken one at a time. The innovation covariance is then a single
number and the gain a vector divided by it, so no matrix is inverted at all.
`update_each` does this for the rows of `H` and the diagonal of `R`.

**`update` when the noise is correlated.** `R` has off-diagonal entries, so
the components cannot be taken apart. The only inverse is of the `m`-by-`m`
innovation covariance, and it is solved by factors, never formed.

**`innovate` before `update` when a measurement may be wrong.** It answers the
innovation, its covariance and the normalised innovation squared and changes
nothing; `gate` compares that with a chi-square quantile; the caller then
applies the update or drops the measurement.

## The rules a user needs

1. **The filter is a value, and a step answers a new one.** `k =
   kalman.predict(k, f, q)`. A `@value` struct is not part of a `Result`
   without being boxed, so nothing here answers a `Result`.
2. **A fault is a number carried in the filter.** `kalman.fault(k)` is
   `KF_OK` when the covariance is finite, symmetric and positive definite
   after the last step. The other four codes are `KF_NOT_FINITE`,
   `KF_NOT_SYMMETRIC`, `KF_NOT_POSITIVE` and `KF_SINGULAR_INNOVATION`.
3. **An update with no gain is refused.** When the innovation covariance is
   not positive definite, the state and the covariance stay as they were, the
   code is `KF_SINGULAR_INNOVATION`, and the innovation is still readable.
4. **The covariance is checked after every step.** Symmetry is judged against
   the largest diagonal element, with a relative tolerance of 1e-4, so a
   covariance in metres squared and one in millimetres squared get the same
   verdict. Positive definiteness is the pivot test of an `L D Lᵀ`
   factorisation with a relative floor of 1e-6.
5. **The check costs one factorisation of `P` in every prediction and every
   update.** An `L D Lᵀ` factorisation of an `n`-by-`n` matrix is about
   `n³/6` multiply-adds, 36 for six states. There is no switch to turn it
   off: a covariance that has gone indefinite is the fault that most needs
   reporting.
6. **The normalised innovation squared is chi-square.** One measurement has
   one degree of freedom, and a gate of 3.84 passes 95 per cent of good
   measurements; 6.63 passes 99 per cent. A vector measurement has `m`
   degrees of freedom: 5.99 and 9.21 for two, 7.81 and 11.34 for three.
7. **The prediction needs the model's Jacobian at the estimate before the
   step, and the update needs the measurement Jacobian at the predicted
   state.** For several scalar updates of a nonlinear model, the caller
   re-evaluates the Jacobian after each one.
8. **A dimension that does not match stops the program.** A matrix beyond
   the capacity, a product of shapes that do not multiply, a covariance that
   is not `n` by `n` all panic with a message that names the function. These
   are programming errors, and a filter that continued would produce a wrong
   estimate silently.
9. **The covariance symmetrises after every step.** The Joseph form's two
   terms are symmetric in exact arithmetic; the average with the transpose
   removes what rounding adds.
10. **Single precision limits the dynamic range.** A covariance whose largest
    and smallest eigenvalues differ by more than about 1e6 loses the small
    one. State scaled to similar magnitudes keeps the filter in range.

## Design

**Capacity, not dimension, is fixed when the package is built.** `kfmat.MAX_DIM`
is 6. A `Mat` is `r`, `c` and a `Vec[fp32; 36]`, and a `KVec` is `n` and a
`Vec[fp32; 6]`: inline storage that lives in the frame of the function that
holds it. A filter of two states and one of six are the same type, and the
actual sizes are set when the filter is made. Six covers a position and a
velocity in three dimensions and the common fusion filters. The constant is
not a configuration key of the package: a key reaches a package only through
a generated module, and the device build of the shard audit stages the
package's sources without one, so a `Vec` sized by a key fails there with
E2920. A program that needs a larger capacity vendors the package with a
different constant.

**linalg-nv does not fit a device build, and so it is not used.** Measured: a
program that builds one 2-by-2 array with `ndarray-nv` and factors it with
`lalu.factor`, built with `--target=nrf52-qemu`, is refused with 505 tier
errors, 445 of them in `ndarray-nv` and 60 in `linalg-nv`. Its matrix is an
`NdFloat`, a heap value; its faults are enums with payloads, which are heap
cells at the embedded tier; its numbers are `Float`, which is binary64 and a
software routine on a Cortex-M4. The routines the filter needs are five
operations on matrices of at most six rows, and they are written out here
over inline storage in single precision.

**The factorisation is `L D Lᵀ`, not Cholesky.** Cholesky takes a square
root, and the embedded tier refuses `math.sqrt` because it lowers to a libm
call. `L D Lᵀ` takes none, and its diagonal `D` is the pivot test for
positive definiteness.

**The gain is `P Hᵀ S⁻¹`, computed as `(S⁻¹ H P)ᵀ`.** `S` and `P` are
symmetric, so one solve against the columns of `H P` gives the transposed gain
without forming `S⁻¹`.

**Integer index arithmetic is not overflow-checked.** The functions of
`kfmat` are `@wrapping`: a dimension is at most six, so no index can reach
the range at which a checked `Int` would trap, and the check costs about a
tenth of the image.

## Running on a microcontroller

`tests/embedded_probe.nv` builds a filter, runs eight steps of a
constant-velocity track through the matrix update, the gate, the scalar
update and the extended form, and prints `PASS: kalman-embedded`. It is the
package's device claim as a program: `--target=nrf52-qemu` stamps
`@tier(embedded)` on every function it compiles, so a build that links has
checked the whole package. It also runs under QEMU.

`bash tests/device_build.sh` builds it for the nRF52840-DK and the
nRF54LM20-DK and reports the sizes beside a program that does everything but
call this package. Nothing is flashed.

| Image | Flash | Of which this package | `.data` + `.bss` | Worst-case stack |
| --- | --- | --- | --- | --- |
| `nrf52840-dk` | 27,932 B | 21,048 B | 2,652 B | 6,136 B |
| `nrf54lm20-dk` | 29,236 B | 21,040 B | 2,652 B | 6,136 B |

The probe calls every public function, so the flash figure is the cost of the
whole package; a program that uses the scalar update and no matrix update
links less. `kfmat.vec_of` and `kfmat.from_list` take a list, which
allocates, and carry `@tier(app)`: a device builds its vectors with `vzeros`
and `vset` and its matrices with `zeros` and `set`. Every other function
checks clean at the embedded tier, so the registry shows the embedded, `rt`,
system, wasm and app tiers.

## What is not included

- **An unscented or a particle filter.** They propagate sample points through
  the model, which needs a function value, and so a closure.
- **A square-root or an information form.** They trade code for dynamic
  range. The Joseph form with a symmetrised covariance is the form that
  single precision needs and a six-state filter fits.
- **Smoothing and fixed-lag estimation.** They keep a history of states.
- **A model library.** A constant-velocity or a coordinated-turn model is a
  few lines of the caller's, and the one the example shows is the pattern.
- **Trigonometry.** The extended filter takes the model's values, so the
  caller supplies the sine and the cosine. [ahrs-nv](https://novo-lang.org/packages/ahrs-nv)
  has single-precision ones.

## Related packages

- [odometry-nv](https://novo-lang.org/packages/odometry-nv) turns encoder
  counts into the twist the example fuses.
- [ahrs-nv](https://novo-lang.org/packages/ahrs-nv) estimates orientation from
  an IMU with a complementary and a Madgwick filter, which is the filter to
  use when orientation is the only quantity wanted.
- [pid-nv](https://novo-lang.org/packages/pid-nv) closes the loop on the
  estimate.
- [linalg-nv](https://novo-lang.org/packages/linalg-nv) is dense linear
  algebra on a host, over `ndarray-nv` arrays.

## Tests

```bash
novo test                                # 47 tests
novo test tests/kfmat_tests.nv           # 21: the matrix routines against hand-worked answers
novo test tests/kalman_vector_tests.nv   #  7: both filters against a numpy reference, step by step
novo test tests/kalman_edge_tests.nv     # 19: faults, gating, refusal, the dimension checks
bash tests/device_build.sh               # the device probe, sizes and a QEMU run
```

`tests/fixtures/kalman_reference.py` writes `kalman_vector_tests.nv`. It is a
float64 Kalman filter in numpy whose gain comes from numpy's matrix inverse,
whose covariance update is the Joseph form written out, and whose noise is a
fixed linear congruential sequence, so the vectors are the same on every run.
It covers a constant-velocity track measured in one dimension and in two, a
correlated measurement covariance, and a pendulum measured by its horizontal
displacement through an extended filter. The tolerance is 1e-4 relative to one
plus the value, about ten times the error measured.
