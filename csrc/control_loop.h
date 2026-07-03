#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

#include "dm_protocol.h"
#include "gravity_comp.h"
#include "motor_defs.h"
#include "perf_counters.h"
#include "safety.h"
#include "serial_port.h"

namespace trlc {

enum class CommLossAction : int {
    HOLD = 0,     // Send MIT with kp=0 (damping only) — motors gently brake
    DISABLE = 1,  // Send disable frames to all motors — motors go limp
};

struct RtLoopConfig {
    std::string serial_port = "/dev/ttyACM0";
    double loop_hz = 250.0;

    std::vector<MotorDescriptor> motors;  // 7 entries (6 arm + gripper)

    std::array<double, 6> default_kp = {80, 70, 60, 20, 20, 10};
    std::array<double, 6> default_kd = {5, 5, 4, 1, 1, 1};

    // Flat [lo0,hi0,lo1,hi1,...] for 6 joints
    std::array<double, 12> joint_pos_limits = {
        -3.14159265, 3.14159265,   // joint_1
        -3.14159265, 3.14159265,   // joint_2
        -3.14159265, 3.14159265,   // joint_3
        -1.74532925, 1.74532925,   // joint_4 (100 deg)
        -1.57079633, 1.57079633,   // joint_5 (90 deg)
        -3.14159265, 3.14159265,   // joint_6
    };
    std::array<double, 6> joint_torque_limits = {28, 28, 28, 10, 10, 10};
    std::array<double, 6> joint_velocity_limits = {5.0, 5.0, 5.0, 15.0, 15.0, 15.0};
    double limit_buffer = 0.05;

    std::string model_path;  // URDF/XML for gravity comp, empty = disabled
    double gravity_comp_scale = 1.0;
    // Evaluate the gravity model at q + offset: the motors' hardware zero is not the model
    // zero (FR-008 calibrated joint offsets, ~1-3 deg on j2/j3). Zeros = old behaviour.
    std::array<double, 6> gravity_q_offset = {0, 0, 0, 0, 0, 0};

    // FR-018 sag observer. At rest the missing feedforward torque is directly observable as
    // the position-error torque kp*(slew_target - q); a gated leaky integrator folds it into
    // tau_ff so the arm settles ON target instead of tau_err/kp away from it.
    // Update gate: |vel| < sag_vel_eps (near rest) AND |residual| < sag_freeze_residual_nm
    // (an obstructed/contacting joint shows a LARGE residual — adapting there would push).
    // The bias is clamped to +-sag_max_nm and leaks to zero over ~1/(leak*loop_hz) seconds
    // so a stale estimate fades instead of persisting at a new pose.
    bool sag_observer_enable = false;
    double sag_lambda = 0.004;            // per-cycle integration gain (~1 s at 250 Hz)
    double sag_max_nm = 2.5;              // |bias| clamp per joint (Nm)
    double sag_freeze_residual_nm = 3.0;  // no adaptation when |residual| exceeds this (contact)
    double sag_vel_eps = 0.05;            // rad/s: adapt only near rest
    double sag_leak = 2e-5;               // per-cycle bias decay (~200 s at 250 Hz)

    // FR-018 friction dither. A joint stuck short of its target inside the static-friction
    // deadband (|err| > dither_pos_eps while |vel| < sag_vel_eps) gets a small sinusoidal
    // tau_ff so it stays in the kinetic regime and creeps onto the target. Amplitudes of 0
    // (default) disable per joint; keep them well below breakaway (~0.3-0.8 Nm).
    bool dither_enable = false;
    std::array<double, 6> dither_amp_nm = {0, 0, 0, 0, 0, 0};
    double dither_hz = 25.0;
    double dither_pos_eps = 0.002;        // rad: position error that counts as "stuck"

    double command_timeout_s = 0.5;
    int overcurrent_threshold = 20;
    int overspeed_threshold = 5;

    // Motor initialization
    int min_motors_required = 6;          // throw if fewer arm motors respond

    // Gripper
    // gripper_open_pos: offset from the calibrated zero (hard stop) in the
    // close direction.  A small negative value backs off from the hard stop
    // so the gripper rests force-free when commanded fully open.
    double gripper_open_pos = 0.0;
    double gripper_closed_pos = -4.7;
    double max_gripper_torque_nm = 1.0;
    double torque_constant = 0.945;
    double emit_velocity_scale = 100.0;
    double emit_current_scale = 1000.0;
    double gripper_cal_timeout_s = 10.0;  // wall-clock timeout for calibration

    // Slew rate limiting (max position change per cycle in radians)
    // At 250 Hz, 0.01 rad/cycle = 2.5 rad/s max slew rate.
    // Set to 0.0 to disable rate limiting for a joint.
    std::array<double, 6> max_pos_delta_per_cycle = {0.02, 0.02, 0.02, 0.06, 0.06, 0.06};

    // Communication loss detection
    int max_consecutive_empty_cycles = 50;  // 50 cycles = 200ms at 250Hz
    CommLossAction comm_loss_action = CommLossAction::DISABLE;
    int per_motor_stale_threshold = 100;    // flag motor after 100 cycles (~400ms)

    // Shutdown
    bool disable_torque_on_disconnect = true;

    // RT
    int rt_priority = 80;
    int rt_cpu_affinity = -1;
    bool rt_use_mlockall = true;
};

struct JointState {
    std::array<double, 6> pos = {};
    std::array<double, 6> vel = {};
    std::array<double, 6> torque = {};
    // Raw temperature bytes from MIT-mode replies (data[6] = MOSFET, data[7] = rotor).
    // Units per DAMIAO protocol are °C; not yet hardware-verified.
    std::array<uint8_t, 6> t_mos = {};
    std::array<uint8_t, 6> t_rotor = {};
    // FR-018 telemetry (one cycle stale): kp*(slew_target - q) = the feedforward torque the
    // model is missing at rest, and the sag observer's current tau_ff correction.
    std::array<double, 6> sag_residual = {};
    std::array<double, 6> sag_bias = {};
};

struct GripperState {
    double pos = 0.0;
    double torque = 0.0;
};

struct HealthState {
    // Safety
    bool damping_mode = false;
    int overcurrent_count = 0;
    int overspeed_count = 0;

    // Communication
    bool comm_loss = false;
    int consecutive_empty_cycles = 0;
    uint64_t total_rx_bytes = 0;
    uint64_t total_tx_frames = 0;
    uint64_t total_write_errors = 0;
    std::array<uint64_t, 7> motor_last_seen_cycle = {};
    std::array<bool, 7> motor_stale = {};

    // Loop metadata
    uint64_t loop_count = 0;
};

class RtControlLoop {
public:
    explicit RtControlLoop(const RtLoopConfig& cfg);
    ~RtControlLoop();

    RtControlLoop(const RtControlLoop&) = delete;
    RtControlLoop& operator=(const RtControlLoop&) = delete;

    void start();
    void stop();

    // Commands (lock-free writes via seqlock)
    void command_joint_pos(const double* q6);
    void command_gripper(double normalized);  // 0=open, 1=closed

    // State (lock-free reads via seqlock)
    JointState get_joint_state() const;
    GripperState get_gripper_state() const;

    // Health and error state (lock-free read via seqlock)
    HealthState get_health() const;

    // Reset safety errors (damping mode, overcurrent, overspeed, comm loss).
    // Blocks until the RT thread processes the reset, with a timeout.
    // Throws std::runtime_error if not acknowledged within timeout_ms.
    // Pass timeout_ms=0 for fire-and-forget (non-blocking).
    void reset_errors(int timeout_ms = 100);

    // Diagnostics
    PerfSnapshot get_perf() const;
    size_t read_cycle_times(float* buf, size_t max) const;
    void reset_perf(int timeout_ms = 100);
    bool is_running() const { return running_.load(std::memory_order_acquire); }
    bool is_rt_active() const { return rt_active_; }

private:
    void configure_motors();
    void calibrate_gripper();
    void rt_thread_func();

    // Seqlock helpers
    struct CommandBuffer {
        std::array<double, 6> q_des = {};
        double gripper_des = 0.0;
        uint64_t timestamp_ns = 0;
    };

    struct StateBuffer {
        std::array<double, 7> pos = {};
        std::array<double, 7> vel = {};
        std::array<double, 7> torque = {};
        // Raw temperature bytes per motor (index 0..5 arm, 6 gripper).
        std::array<uint8_t, 7> t_mos = {};
        std::array<uint8_t, 7> t_rotor = {};
        std::array<double, 6> sag_residual = {};
        std::array<double, 6> sag_bias = {};
    };

    RtLoopConfig cfg_;
    SerialPort serial_;
    GravityCompensator grav_comp_;
    PacketParser parser_;
    PerfCounters perf_;
    SafetyState safety_state_;

    // Command seqlock (Python writes, RT reads)
    alignas(64) CommandBuffer cmd_buf_;
    alignas(64) std::atomic<uint64_t> cmd_seq_{0};

    // State seqlock (RT writes, Python reads)
    alignas(64) StateBuffer state_buf_;
    alignas(64) std::atomic<uint64_t> state_seq_{0};

    // Health seqlock (RT writes, Python reads)
    alignas(64) HealthState health_buf_;
    alignas(64) std::atomic<uint64_t> health_seq_{0};

    // RT-thread-only mutable health tracking
    HealthState health_rt_;

    // Error reset coordination (Python sets, RT thread increments ack after processing)
    std::atomic<bool> error_reset_requested_{false};
    std::atomic<uint64_t> error_reset_ack_{0};

    std::atomic<bool> running_{false};
    std::thread thread_;
    bool rt_active_ = false;

    // Gripper calibration result
    double gripper_open_pos_ = 0.0;

    // Slew rate limiter state (RT thread only)
    std::array<double, 6> slew_target_ = {};

    // FR-018 sag observer / dither state (RT thread only)
    std::array<double, 6> sag_bias_ = {};
    std::array<double, 6> sag_residual_ = {};
    double dither_phase_ = 0.0;
};

} // namespace trlc
