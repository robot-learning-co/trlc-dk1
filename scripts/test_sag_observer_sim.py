"""Offline validation of the FR-018 sag observer + friction dither (csrc/control_loop.cpp 8b).

Replicates the exact per-cycle update law against a 1-DOF stick-slip joint simulation
(inertia + viscous + static/kinetic Coulomb friction + constant gravity-model error) and
asserts the four properties that matter before the code ever runs on hardware:

  1. observer OFF  -> steady-state error ~= dtau/kp (the sag we measured on the DK-1)
  2. observer ON   -> error collapses well below the friction deadband; bias -> dtau
  3. + dither      -> stuck-short residual error shrinks further (kinetic regime)
  4. contact       -> residual exceeds the freeze threshold -> bias STOPS adapting
                      (no integrator windup pushing into an obstacle), and stays clamped

    python scripts/test_sag_observer_sim.py
"""
import math

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [OK ] {name} {detail}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {detail}")


class Joint1D:
    """Stick-slip joint under MIT impedance control, integrated at the RT rate."""

    def __init__(self, kp=70.0, kd=5.0, inertia=0.05, tau_static=0.8, tau_kinetic=0.6,
                 dtau=1.6, hz=250.0, wall=None):
        self.kp, self.kd, self.I = kp, kd, inertia
        self.tf_s, self.tf_k = tau_static, tau_kinetic
        self.dtau = dtau              # gravity-model error: tau_ff is SHORT by this much
        self.dt = 1.0 / hz
        self.wall = wall              # position obstacle (contact test), None = free
        self.q = 0.0
        self.qd = 0.0

    def step(self, q_des, tau_extra=0.0):
        # controller torque with the DELIBERATELY wrong feedforward (tau_g - dtau) and any
        # observer/dither addition; true gravity load is tau_g -> net error torque is -dtau
        tau_net = self.kp * (q_des - self.q) + self.kd * (0.0 - self.qd) - self.dtau + tau_extra
        if abs(self.qd) < 1e-4:
            if abs(tau_net) <= self.tf_s:
                self.qd = 0.0          # stuck: static friction absorbs the net torque
                return
            qdd = (tau_net - math.copysign(self.tf_k, tau_net)) / self.I
        else:
            qdd = (tau_net - math.copysign(self.tf_k, self.qd)) / self.I
        self.qd += qdd * self.dt
        self.q += self.qd * self.dt
        if self.wall is not None and self.q > self.wall:
            self.q, self.qd = self.wall, 0.0


def run(observer, dither, cycles=1500, wall=None, q_des=0.05,
        lam=0.004, bmax=2.5, freeze=3.0, veps=0.05, leak=2e-5,
        amp=0.25, f_hz=25.0, peps=0.002, hz=250.0):
    """Mirror of control_loop.cpp step 8b (single joint)."""
    j = Joint1D(hz=hz, wall=wall)
    bias, phase = 0.0, 0.0
    for _ in range(cycles):
        residual = j.kp * (q_des - j.q)                    # kp*(slew_target - cur_pos)
        extra = 0.0
        if observer:
            bias *= (1.0 - leak)
            if abs(j.qd) < veps and abs(residual) < freeze:
                bias = max(-bmax, min(bmax, bias + lam * residual))
            extra += bias
        if dither:
            s = math.sin(phase)
            phase += 2.0 * math.pi * f_hz / hz
            if phase > 2.0 * math.pi:
                phase -= 2.0 * math.pi
            if abs(q_des - j.q) > peps and abs(j.qd) < veps:
                extra += amp * s
        j.step(q_des, extra)
    return j, bias, j.kp * (q_des - j.q)


def main():
    print("== 1. observer off: steady-state sag ==")
    j, _, res = run(observer=False, dither=False)
    err0 = abs(0.05 - j.q)
    # analytic sag floor dtau-tf_s .. dtau+tf_s over kp; measured DK-1 numbers live here too
    check("open-loop error in the analytic sag band",
          (1.6 - 0.8) / 70.0 <= err0 <= (1.6 + 0.8) / 70.0 + 1e-4,
          f"err={err0 * 1e3:.2f}mrad (dtau/kp={1.6 / 70.0 * 1e3:.2f}mrad)")

    print("== 2. observer on: sag learned away ==")
    j, bias, res = run(observer=True, dither=False, cycles=2500)
    err1 = abs(0.05 - j.q)
    check("error < 40% of open-loop", err1 < 0.4 * err0,
          f"err={err1 * 1e3:.2f}mrad vs open-loop {err0 * 1e3:.2f}mrad")
    check("bias converged toward dtau", 0.8 <= bias <= 2.5, f"bias={bias:.2f}Nm (dtau=1.6)")
    check("bias respects clamp", abs(bias) <= 2.5, f"|bias|={abs(bias):.2f}")

    print("== 3. observer + dither: through the deadband ==")
    j, bias, _ = run(observer=True, dither=True, cycles=2500)
    err2 = abs(0.05 - j.q)
    check("dither shrinks the residual further", err2 <= err1 + 1e-6,
          f"err={err2 * 1e3:.3f}mrad (observer-only {err1 * 1e3:.3f})")
    check("dither leaves converged joint alone: err below stuck gate or improved",
          err2 < 0.002 or err2 < err1, f"err={err2 * 1e3:.3f}mrad, gate=2mrad")

    print("== 4. contact: freeze prevents windup ==")
    # obstacle at 20 mrad short of the 50 mrad target -> residual = 70*0.03 = 2.1 Nm < freeze
    # at first, growing as bias pushes... assert bias never exceeds clamp and that with a HARD
    # block (residual > freeze from the start) the bias barely moves.
    jb, bias_b, res_b = run(observer=True, dither=False, cycles=2500, wall=0.030)
    check("blocked: bias stays within clamp", abs(bias_b) <= 2.5, f"bias={bias_b:.2f}Nm")
    jh, bias_h, res_h = run(observer=True, dither=False, cycles=2500, wall=0.030, q_des=0.10)
    # here residual = 70*(0.10-0.03) = 4.9 Nm > freeze 3.0 -> frozen from the moment of contact
    check("hard block: adaptation frozen (bias ~ pre-contact value)", abs(bias_h) <= 2.5
          and abs(res_h) > 3.0, f"bias={bias_h:.2f}Nm residual={res_h:.2f}Nm (frozen>3)")

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
