#!/usr/bin/env python3
"""The reference filter for kalman-nv's vector tests.

Run it to write tests/kalman_vector_tests.nv:

    python3 tests/fixtures/kalman_reference.py > tests/kalman_vector_tests.nv

The reference is a textbook Kalman filter in float64 with numpy, written
with none of the package's choices: the gain comes from numpy's matrix
inverse, not from an L D L^T solve; the covariance update is the Joseph
form written out; and the normalised innovation squared is y^T S^-1 y
with the same inverse.  Three scenarios are generated:

  cv1   a one-dimensional constant-velocity track, position measured
  cv2d  a two-dimensional track, both positions measured, diagonal R
  pend  a pendulum, extended filter, a horizontal displacement measured

The measurement noise is a fixed pseudo-random sequence from a linear
congruential generator, so the vectors are the same on every run and
every machine.  The pendulum's truth is integrated with fourth-order
Runge-Kutta.  Every number the test file needs is printed with repr, so
nothing is rounded between this script and the suite.
"""

import math
import numpy as np


class Lcg:
    """A 31-bit linear congruential generator and a bell-shaped draw."""

    def __init__(self, seed):
        self.s = seed

    def uniform(self):
        self.s = (1103515245 * self.s + 12345) % 2147483648
        return self.s / 2147483648.0

    def normal(self):
        return sum(self.uniform() for _ in range(12)) - 6.0


def predict(x, p, f_val, f_jac, q):
    return f_val, f_jac @ p @ f_jac.T + q


def update(x, p, z, hx, h_jac, r):
    y = z - hx
    s = h_jac @ p @ h_jac.T + r
    s_inv = np.linalg.inv(s)
    k = p @ h_jac.T @ s_inv
    nis = float(y @ s_inv @ y)
    i_kh = np.eye(len(x)) - k @ h_jac
    p_new = i_kh @ p @ i_kh.T + k @ r @ k.T
    p_new = 0.5 * (p_new + p_new.T)
    return x + k @ y, p_new, nis


def flat(a):
    return [float(v) for v in np.asarray(a).reshape(-1)]


def lst(xs):
    return "[" + ", ".join(repr(float(v)) for v in xs) + "]"


def cv_model(dt, sigma_a):
    f = np.array([[1.0, dt], [0.0, 1.0]])
    q = sigma_a ** 2 * np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]])
    return f, q


def scenario_cv1():
    dt, sigma_a, r_var, steps = 0.1, 0.5, 0.25, 30
    f, q = cv_model(dt, sigma_a)
    h = np.array([[1.0, 0.0]])
    r = np.array([[r_var]])
    rng = Lcg(20261123)
    zs = []
    for k in range(1, steps + 1):
        truth = 1.0 + 2.0 * (k * dt)
        zs.append(truth + math.sqrt(r_var) * rng.normal())
    x, p = np.zeros(2), np.diag([10.0, 10.0])
    out = []
    for z in zs:
        x, p = predict(x, p, f @ x, f, q)
        x, p, nis = update(x, p, np.array([z]), h @ x, h, r)
        out += [x[0], x[1], p[0, 0], p[0, 1], p[1, 1], nis]
    return dict(dt=dt, f=flat(f), q=flat(q), h=flat(h), r=r_var, z=zs, want=out)


def scenario_cv2d():
    dt, sigma_a, steps = 0.2, 0.4, 20
    f1, q1 = cv_model(dt, sigma_a)
    f = np.zeros((4, 4))
    q = np.zeros((4, 4))
    for a in (0, 2):
        f[a:a + 2, a:a + 2] = f1
        q[a:a + 2, a:a + 2] = q1
    h = np.array([[1.0, 0, 0, 0], [0, 0, 1.0, 0]])
    r = np.diag([0.5, 0.4])
    rng = Lcg(7)
    zs = []
    for k in range(1, steps + 1):
        t = k * dt
        zs.append(3.0 + 1.5 * t + math.sqrt(0.5) * rng.normal())
        zs.append(-2.0 + 0.5 * t + math.sqrt(0.4) * rng.normal())
    x, p = np.zeros(4), np.diag([10.0, 4.0, 10.0, 4.0])
    out = []
    for i in range(steps):
        x, p = predict(x, p, f @ x, f, q)
        z = np.array([zs[2 * i], zs[2 * i + 1]])
        x, p, nis = update(x, p, z, h @ x, h, r)
        out += [x[0], x[1], x[2], x[3], p[0, 0], p[1, 1], p[2, 2], p[3, 3], nis]
    return dict(dt=dt, f=flat(f), q=flat(q), h=flat(h), r=[0.5, 0.4], z=zs, want=out)


def scenario_correlated():
    """One matrix update with a correlated R, for the cross terms of S."""
    x = np.array([1.0, -0.5, 2.0, 0.25])
    p = np.array([[2.0, 0.3, 0.2, 0.0],
                  [0.3, 1.0, 0.0, 0.1],
                  [0.2, 0.0, 3.0, 0.4],
                  [0.0, 0.1, 0.4, 0.5]])
    h = np.array([[1.0, 0, 0, 0], [0, 0, 1.0, 0], [0, 1.0, 0, 1.0]])
    r = np.array([[0.5, 0.1, 0.0], [0.1, 0.4, 0.05], [0.0, 0.05, 0.3]])
    z = np.array([1.4, 1.7, -0.1])
    x2, p2, nis = update(x, p, z, h @ x, h, r)
    return dict(x=flat(x), p=flat(p), h=flat(h), r=flat(r), z=flat(z),
                want_x=flat(x2), want_p=flat(p2), want_nis=nis)


def pend_f(x, dt, g_over_l):
    th, om = x
    val = np.array([th + om * dt, om - g_over_l * math.sin(th) * dt])
    jac = np.array([[1.0, dt], [-g_over_l * math.cos(th) * dt, 1.0]])
    return val, jac


def pend_h(x, length):
    return (np.array([length * math.sin(x[0])]),
            np.array([[length * math.cos(x[0]), 0.0]]))


def scenario_pend():
    dt, g_over_l, length, steps, sigma = 0.05, 9.81, 1.0, 40, 0.02
    rng = Lcg(424242)

    def deriv(s):
        return np.array([s[1], -g_over_l * math.sin(s[0])])

    truth = np.array([0.8, 0.0])
    zs = []
    for _ in range(steps):
        k1 = deriv(truth)
        k2 = deriv(truth + 0.5 * dt * k1)
        k3 = deriv(truth + 0.5 * dt * k2)
        k4 = deriv(truth + dt * k3)
        truth = truth + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        zs.append(length * math.sin(truth[0]) + sigma * rng.normal())
    q = np.diag([1e-5, 1e-4])
    r = np.array([[sigma ** 2]])
    x, p = np.array([0.5, 0.0]), np.diag([0.5, 0.5])
    out = []
    for z in zs:
        fv, fj = pend_f(x, dt, g_over_l)
        x, p = predict(x, p, fv, fj, q)
        hv, hj = pend_h(x, length)
        x, p, nis = update(x, p, np.array([z]), hv, hj, r)
        out += [x[0], x[1], p[0, 0], p[1, 1], nis]
    return dict(dt=dt, g_over_l=g_over_l, length=length, sigma=sigma, q=flat(q),
                z=zs, want=out)


HEADER = '''// kalman_vector_tests.nv — the filters against a reference filter, step
// by step.
//
// GENERATED by tests/fixtures/kalman_reference.py; do not edit by hand.
// The reference is a float64 Kalman filter written with numpy: the gain
// comes from numpy's matrix inverse, the covariance update is the Joseph
// form written out, and the normalised innovation squared is y' S^-1 y.
// None of that shares a line with the package, whose gain comes from an
// L D L' solve in single precision.  The measurement noise is a fixed
// linear congruential sequence, so the vectors are the same everywhere.
//
// The tolerance is `TOL`, relative to one plus the magnitude of the
// number: binary32 carries about seven digits and these filters lose
// one or two of them over thirty steps, so 1e-4 is ten times the
// error measured, and a wrong term in a gain or a covariance moves the
// answer by far more.

use std.test
use kfmat
use kalman
use ekf

const TOL = 0.0001

// Whether a binary32 answer is within `TOL` of the reference's, relative
// to one plus the reference's magnitude.
fn near(got: fp32, want: Float) -> Bool
    let g = got as Float
    let scale = 1.0 + math.abs(want)
    math.abs(g - want) <= TOL * scale
'''


def emit():
    a = scenario_cv1()
    b = scenario_cv2d()
    c = scenario_correlated()
    d = scenario_pend()
    out = [HEADER]
    out.append(f'''
// ── The vectors ──────────────────────────────────────────────────────

fn cv1_f() -> [Float]
    {lst(a["f"])}

fn cv1_q() -> [Float]
    {lst(a["q"])}

// Thirty position measurements of a target at 1 + 2t, noise variance 0.25.
fn cv1_z() -> [Float]
    {lst(a["z"])}

// Per step: x0, x1, P00, P01, P11 after the update, then the normalised
// innovation squared before it.
fn cv1_want() -> [Float]
    {lst(a["want"])}

fn cv2_f() -> [Float]
    {lst(b["f"])}

fn cv2_q() -> [Float]
    {lst(b["q"])}

// Twenty pairs of position measurements, noise variances 0.5 and 0.4.
fn cv2_z() -> [Float]
    {lst(b["z"])}

// Per step: x0..x3, P00, P11, P22, P33 after the update, then the
// normalised innovation squared before it.
fn cv2_want() -> [Float]
    {lst(b["want"])}

fn corr_x() -> [Float]
    {lst(c["x"])}

fn corr_p() -> [Float]
    {lst(c["p"])}

fn corr_h() -> [Float]
    {lst(c["h"])}

fn corr_r() -> [Float]
    {lst(c["r"])}

fn corr_z() -> [Float]
    {lst(c["z"])}

fn corr_want_x() -> [Float]
    {lst(c["want_x"])}

fn corr_want_p() -> [Float]
    {lst(c["want_p"])}

fn corr_want_nis() -> Float
    {repr(c["want_nis"])}

// Forty horizontal displacements of a pendulum released at 0.8 rad.
fn pend_z() -> [Float]
    {lst(d["z"])}

// Per step: theta, omega, P00, P11 after the update, then the normalised
// innovation squared before it.
fn pend_want() -> [Float]
    {lst(d["want"])}
''')
    out.append(f'''
// ── The linear filter ────────────────────────────────────────────────

fn cv1_filter() -> Kf
    kalman.filter(kfmat.vec2(0.0, 0.0), kfmat.diagonal(2, 10.0))

@test
fn test_a_constant_velocity_track_follows_the_reference_step_by_step()
    let z = cv1_z()
    let want = cv1_want()
    let f = kfmat.from_list(2, 2, cv1_f())
    let q = kfmat.from_list(2, 2, cv1_q())
    let h = kfmat.from_list(1, 2, [1.0, 0.0])
    let r = kfmat.diagonal(1, {a["r"]!r})
    var k = cv1_filter()
    for i in 0..list.len(z)
        k = kalman.predict(k, f, q)
        let zi = kfmat.vec1(z[i] as fp32)
        test.assert(near(kalman.innovate(k, zi, h, r).nis, want[i * 6 + 5]))
        k = kalman.update(k, zi, h, r)
        let x = kalman.state(k)
        let p = kalman.covariance(k)
        test.assert(near(x.a[0], want[i * 6]))
        test.assert(near(x.a[1], want[i * 6 + 1]))
        test.assert(near(kfmat.get(p, 0, 0), want[i * 6 + 2]))
        test.assert(near(kfmat.get(p, 0, 1), want[i * 6 + 3]))
        test.assert(near(kfmat.get(p, 1, 1), want[i * 6 + 4]))
        test.assert_eq(kalman.fault(k), KF_OK)

@test
fn test_the_scalar_update_gives_the_matrix_update_for_one_measurement()
    let z = cv1_z()
    let want = cv1_want()
    let f = kfmat.from_list(2, 2, cv1_f())
    let q = kfmat.from_list(2, 2, cv1_q())
    let row = kfmat.vec2(1.0, 0.0)
    var k = cv1_filter()
    for i in 0..list.len(z)
        k = kalman.predict(k, f, q)
        let inn = kalman.innovate_scalar(k, z[i] as fp32, row, {a["r"]!r})
        test.assert(near(inn.nis, want[i * 6 + 5]))
        k = kalman.update_scalar(k, z[i] as fp32, row, {a["r"]!r})
        test.assert(near(kalman.state(k).a[0], want[i * 6]))
        test.assert(near(kalman.state(k).a[1], want[i * 6 + 1]))
        test.assert(near(kfmat.get(kalman.covariance(k), 1, 1), want[i * 6 + 4]))
        test.assert(near(kalman.nis(k), want[i * 6 + 5]))

fn cv2_filter() -> Kf
    kalman.filter(kfmat.vec4(0.0, 0.0, 0.0, 0.0), kfmat.from_list(4, 4, [10.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 4.0]))

fn cv2_h() -> Mat
    kfmat.from_list(2, 4, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])

fn cv2_check(k: Kf, want: [Float], i: Int) -> Bool
    let x = kalman.state(k)
    let p = kalman.covariance(k)
    near(x.a[0], want[i * 9]) and near(x.a[1], want[i * 9 + 1]) and near(x.a[2], want[i * 9 + 2]) and near(x.a[3], want[i * 9 + 3]) and near(kfmat.get(p, 0, 0), want[i * 9 + 4]) and near(kfmat.get(p, 1, 1), want[i * 9 + 5]) and near(kfmat.get(p, 2, 2), want[i * 9 + 6]) and near(kfmat.get(p, 3, 3), want[i * 9 + 7])

@test
fn test_a_two_dimensional_track_with_a_matrix_update_follows_the_reference()
    let z = cv2_z()
    let want = cv2_want()
    let f = kfmat.from_list(4, 4, cv2_f())
    let q = kfmat.from_list(4, 4, cv2_q())
    let r = kfmat.from_list(2, 2, [0.5, 0.0, 0.0, 0.4])
    var k = cv2_filter()
    for i in 0..list.len(want) / 9
        k = kalman.predict(k, f, q)
        let zi = kfmat.vec2(z[2 * i] as fp32, z[2 * i + 1] as fp32)
        test.assert(near(kalman.innovate(k, zi, cv2_h(), r).nis, want[i * 9 + 8]))
        k = kalman.update(k, zi, cv2_h(), r)
        test.assert(cv2_check(k, want, i))
        test.assert(near(kalman.nis(k), want[i * 9 + 8]))
        test.assert_eq(kalman.innovation(k).n, 2)
        test.assert_eq(kfmat.get(kalman.innovation_cov(k), 0, 1) == kfmat.get(kalman.innovation_cov(k), 1, 0), true)

@test
fn test_uncorrelated_measurements_taken_one_at_a_time_give_the_matrix_answer()
    let z = cv2_z()
    let want = cv2_want()
    let f = kfmat.from_list(4, 4, cv2_f())
    let q = kfmat.from_list(4, 4, cv2_q())
    var k = cv2_filter()
    for i in 0..list.len(want) / 9
        k = kalman.predict(k, f, q)
        let zi = kfmat.vec2(z[2 * i] as fp32, z[2 * i + 1] as fp32)
        k = kalman.update_each(k, zi, cv2_h(), kfmat.vec2(0.5, 0.4))
        test.assert(cv2_check(k, want, i))
        test.assert_eq(kalman.fault(k), KF_OK)

@test
fn test_a_correlated_measurement_covariance_gives_the_reference_cross_terms()
    let x0 = kfmat.vec_of(corr_x())
    let p0 = kfmat.from_list(4, 4, corr_p())
    let h = kfmat.from_list(3, 4, corr_h())
    let r = kfmat.from_list(3, 3, corr_r())
    let k = kalman.filter(x0, p0)
    let z = kfmat.vec_of(corr_z())
    test.assert(near(kalman.innovate(k, z, h, r).nis, corr_want_nis()))
    let k2 = kalman.update(k, z, h, r)
    let wx = corr_want_x()
    let wp = corr_want_p()
    for i in 0..4
        test.assert(near(kalman.state(k2).a[i], wx[i]))
    for i in 0..4
        for j in 0..4
            test.assert(near(kfmat.get(kalman.covariance(k2), i, j), wp[i * 4 + j]))
    test.assert(kfmat.is_symmetric(kalman.covariance(k2), 0.0))
''')
    out.append(f'''
// ── The extended filter ──────────────────────────────────────────────

const DT = {d["dt"]!r}
const G_OVER_L = {d["g_over_l"]!r}
const LENGTH = {d["length"]!r}
const SIGMA = {d["sigma"]!r}

// The pendulum's move at the estimate: its value and its Jacobian.
fn pend_value(x: KVec) -> KVec
    let th = x.a[0] as Float
    kfmat.vec2((th + (x.a[1] as Float) * DT) as fp32, ((x.a[1] as Float) - G_OVER_L * math.sin(th) * DT) as fp32)

fn pend_jac(x: KVec) -> Mat
    let th = x.a[0] as Float
    kfmat.from_list(2, 2, [1.0, DT, 0.0 - G_OVER_L * math.cos(th) * DT, 1.0])

fn pend_filter() -> Kf
    kalman.filter(kfmat.vec2(0.5, 0.0), kfmat.diagonal(2, 0.5))

@test
fn test_a_pendulum_follows_the_reference_extended_filter_step_by_step()
    let z = pend_z()
    let want = pend_want()
    let q = kfmat.from_list(2, 2, {lst(d["q"])})
    let r = kfmat.diagonal(1, (SIGMA * SIGMA) as fp32)
    var k = pend_filter()
    for i in 0..list.len(z)
        k = ekf.predict(k, pend_value(kalman.state(k)), pend_jac(kalman.state(k)), q)
        let th = kalman.state(k).a[0] as Float
        let hx = kfmat.vec1((LENGTH * math.sin(th)) as fp32)
        let hj = kfmat.from_list(1, 2, [LENGTH * math.cos(th), 0.0])
        let zi = kfmat.vec1(z[i] as fp32)
        test.assert(near(ekf.innovate(k, zi, hx, hj, r).nis, want[i * 5 + 4]))
        k = ekf.update(k, zi, hx, hj, r)
        test.assert(near(kalman.state(k).a[0], want[i * 5]))
        test.assert(near(kalman.state(k).a[1], want[i * 5 + 1]))
        test.assert(near(kfmat.get(kalman.covariance(k), 0, 0), want[i * 5 + 2]))
        test.assert(near(kfmat.get(kalman.covariance(k), 1, 1), want[i * 5 + 3]))
        test.assert_eq(kalman.fault(k), KF_OK)

@test
fn test_the_extended_scalar_update_follows_the_same_reference()
    let z = pend_z()
    let want = pend_want()
    let q = kfmat.from_list(2, 2, {lst(d["q"])})
    var k = pend_filter()
    for i in 0..list.len(z)
        k = ekf.predict(k, pend_value(kalman.state(k)), pend_jac(kalman.state(k)), q)
        let th = kalman.state(k).a[0] as Float
        let hx = (LENGTH * math.sin(th)) as fp32
        let row = kfmat.vec2((LENGTH * math.cos(th)) as fp32, 0.0)
        let inn = ekf.innovate_scalar(k, z[i] as fp32, hx, row, (SIGMA * SIGMA) as fp32)
        test.assert(near(inn.nis, want[i * 5 + 4]))
        k = ekf.update_scalar(k, z[i] as fp32, hx, row, (SIGMA * SIGMA) as fp32)
        test.assert(near(kalman.state(k).a[0], want[i * 5]))
        test.assert(near(kalman.state(k).a[1], want[i * 5 + 1]))
        test.assert(near(kfmat.get(kalman.covariance(k), 0, 0), want[i * 5 + 2]))
''')
    return "".join(out)


if __name__ == "__main__":
    print(emit())
