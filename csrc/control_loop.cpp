#include "control_loop.h"
#include "rt_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <sstream>
#include <stdexcept>
#include <time.h>

#ifdef __APPLE__
#include <mach/mach_time.h>
#endif

namespace trlc {

static uint64_t now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL + static_cast<uint64_t>(ts.tv_nsec);
}

static void sleep_until_ns(uint64_t target_ns) {
#ifdef __APPLE__
    // macOS lacks clock_nanosleep; use mach_wait_until for absolute sleep
    static mach_timebase_info_data_t tb = {};
    if (tb.denom == 0) mach_timebase_info(&tb);
    // Convert wall-clock ns to mach absolute time units
    uint64_t mach_target = target_ns * tb.denom / tb.numer;
    mach_wait_until(mach_target);
#else
    struct timespec ts;
    ts.tv_sec = static_cast<time_t>(target_ns / 1000000000ULL);
    ts.tv_nsec = static_cast<long>(target_ns % 1000000000ULL);
    clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, nullptr);
#endif
}

static void sleep_ms(int ms) {
    struct timespec ts = {ms / 1000, (ms % 1000) * 1000000};
    nanosleep(&ts, nullptr);
}

// Write a frame then wait for and read the response.
static size_t send_and_recv(SerialPort& serial, const uint8_t* tx, size_t tx_len,
                            uint8_t* rx_buf, size_t rx_max, int timeout_us = 2000) {
    serial.write(tx, tx_len);
    return serial.read_with_timeout(rx_buf, rx_max, timeout_us);
}

RtControlLoop::RtControlLoop(const RtLoopConfig& cfg) : cfg_(cfg) {}

RtControlLoop::~RtControlLoop() {
    stop();
}

void RtControlLoop::start() {
    if (running_.load()) {
        throw std::runtime_error("RtControlLoop already running");
    }

    // Validate configuration
    if (cfg_.loop_hz <= 0.0 || cfg_.loop_hz > 10000.0) {
        throw std::invalid_argument("loop_hz must be in (0, 10000], got: " +
                                    std::to_string(cfg_.loop_hz));
    }
    if (cfg_.motors.size() < 7) {
        throw std::invalid_argument("Need at least 7 motor descriptors (6 arm + 1 gripper), got: " +
                                    std::to_string(cfg_.motors.size()));
    }
    if (cfg_.command_timeout_s <= 0.0) {
        throw std::invalid_argument("command_timeout_s must be positive, got: " +
                                    std::to_string(cfg_.command_timeout_s));
    }
    if (cfg_.min_motors_required < 0 || cfg_.min_motors_required > 6) {
        throw std::invalid_argument("min_motors_required must be in [0, 6], got: " +
                                    std::to_string(cfg_.min_motors_required));
    }

    if (!serial_.open(cfg_.serial_port, 921600)) {
        throw std::runtime_error("Failed to open serial port: " + cfg_.serial_port);
    }

    sleep_ms(500);
    configure_motors();

    if (!cfg_.model_path.empty()) {
        if (!grav_comp_.load(cfg_.model_path, 6)) {
            std::fprintf(stderr, "Warning: gravity compensation disabled (failed to load model)\n");
        }
    }

    calibrate_gripper();

    // Initialize command buffer to current position
    {
        uint64_t s = cmd_seq_.load(std::memory_order_relaxed);
        cmd_seq_.store(s + 1, std::memory_order_release);
        for (int i = 0; i < 6; ++i) {
            cmd_buf_.q_des[static_cast<size_t>(i)] = state_buf_.pos[static_cast<size_t>(i)];
        }
        cmd_buf_.gripper_des = 0.0;
        cmd_buf_.timestamp_ns = now_ns();
        cmd_seq_.store(s + 2, std::memory_order_release);
    }

    std::fprintf(stderr, "Command buffer initialized to current pos:\n");
    for (int i = 0; i < 6; ++i) {
        std::fprintf(stderr, "  joint[%d] q_des=%.4f (from state_buf_.pos=%.4f)\n",
                     i, cmd_buf_.q_des[static_cast<size_t>(i)],
                     state_buf_.pos[static_cast<size_t>(i)]);
    }

    running_.store(true, std::memory_order_release);
    thread_ = std::thread(&RtControlLoop::rt_thread_func, this);
}

void RtControlLoop::stop() {
    if (!running_.load()) return;
    running_.store(false, std::memory_order_release);
    if (thread_.joinable()) thread_.join();

    uint8_t frame[30];
    uint8_t rx_buf[256];

    // Drain any pending data from the serial buffer
    while (serial_.read_all(rx_buf, sizeof(rx_buf)) > 0) {}

    // Disable all motors — send all frames back-to-back without waiting
    // for individual responses.  The motors will process the disable
    // commands regardless of whether we read the acks.
    if (cfg_.disable_torque_on_disconnect) {
        for (const auto& m : cfg_.motors) {
            build_disable_frame(frame, m.slave_id);
            serial_.write(frame, 30);
        }
        // Brief pause for the last frame to be transmitted over USB
        sleep_ms(20);
    }

    serial_.close();
    std::fprintf(stderr, "RtControlLoop stopped\n");
}

void RtControlLoop::command_joint_pos(const double* q6) {
    uint64_t s = cmd_seq_.load(std::memory_order_relaxed);
    cmd_seq_.store(s + 1, std::memory_order_release);
    std::memcpy(cmd_buf_.q_des.data(), q6, 6 * sizeof(double));
    cmd_buf_.timestamp_ns = now_ns();
    cmd_seq_.store(s + 2, std::memory_order_release);
}

void RtControlLoop::command_gripper(double normalized) {
    normalized = std::clamp(normalized, 0.0, 1.0);
    uint64_t s = cmd_seq_.load(std::memory_order_relaxed);
    cmd_seq_.store(s + 1, std::memory_order_release);
    cmd_buf_.gripper_des = normalized;
    cmd_buf_.timestamp_ns = now_ns();
    cmd_seq_.store(s + 2, std::memory_order_release);
}

JointState RtControlLoop::get_joint_state() const {
    JointState result;
    for (int attempt = 0; attempt < 100; ++attempt) {
        uint64_t s1 = state_seq_.load(std::memory_order_acquire);
        if (s1 & 1) continue;
        for (int i = 0; i < 6; ++i) {
            result.pos[static_cast<size_t>(i)] = state_buf_.pos[static_cast<size_t>(i)];
            result.vel[static_cast<size_t>(i)] = state_buf_.vel[static_cast<size_t>(i)];
            result.torque[static_cast<size_t>(i)] = state_buf_.torque[static_cast<size_t>(i)];
            result.t_mos[static_cast<size_t>(i)]   = state_buf_.t_mos[static_cast<size_t>(i)];
            result.t_rotor[static_cast<size_t>(i)] = state_buf_.t_rotor[static_cast<size_t>(i)];
            result.sag_residual[static_cast<size_t>(i)] = state_buf_.sag_residual[static_cast<size_t>(i)];
            result.sag_bias[static_cast<size_t>(i)]     = state_buf_.sag_bias[static_cast<size_t>(i)];
        }
        uint64_t s2 = state_seq_.load(std::memory_order_acquire);
        if (s1 == s2) return result;
    }
    return result;
}

GripperState RtControlLoop::get_gripper_state() const {
    GripperState result;
    for (int attempt = 0; attempt < 100; ++attempt) {
        uint64_t s1 = state_seq_.load(std::memory_order_acquire);
        if (s1 & 1) continue;
        result.pos = state_buf_.pos[6];
        result.torque = state_buf_.torque[6];
        uint64_t s2 = state_seq_.load(std::memory_order_acquire);
        if (s1 == s2) {
            double range = cfg_.gripper_closed_pos - gripper_open_pos_;
            if (std::abs(range) > 1e-6) {
                result.pos = std::clamp((result.pos - gripper_open_pos_) / range, 0.0, 1.0);
            }
            return result;
        }
    }
    return result;
}

HealthState RtControlLoop::get_health() const {
    HealthState result;
    for (int attempt = 0; attempt < 100; ++attempt) {
        uint64_t s1 = health_seq_.load(std::memory_order_acquire);
        if (s1 & 1) continue;
        result = health_buf_;
        uint64_t s2 = health_seq_.load(std::memory_order_acquire);
        if (s1 == s2) return result;
    }
    return result;  // best effort
}

void RtControlLoop::reset_errors(int timeout_ms) {
    if (!running_.load(std::memory_order_acquire)) return;
    uint64_t before = error_reset_ack_.load(std::memory_order_acquire);
    error_reset_requested_.store(true, std::memory_order_release);
    if (timeout_ms <= 0) return;  // fire-and-forget
    for (int waited = 0; waited < timeout_ms; ++waited) {
        if (error_reset_ack_.load(std::memory_order_acquire) != before) return;
        if (!running_.load(std::memory_order_acquire)) return;
        struct timespec ts = {0, 1000000};  // 1ms
        nanosleep(&ts, nullptr);
    }
    throw std::runtime_error("reset_errors() timed out waiting for RT thread acknowledgment");
}

PerfSnapshot RtControlLoop::get_perf() const { return perf_.snapshot(); }

size_t RtControlLoop::read_cycle_times(float* buf, size_t max) const {
    return perf_.read_ring(buf, max);
}

void RtControlLoop::reset_perf(int timeout_ms) { perf_.reset(timeout_ms); }

// --- Motor configuration ---

void RtControlLoop::configure_motors() {
    if (cfg_.motors.size() < 7) {
        throw std::runtime_error("Need at least 7 motor descriptors (6 arm + 1 gripper)");
    }

    uint8_t frame[30];
    uint8_t rx_buf[256];

    for (const auto& m : cfg_.motors) {
        for (int attempt = 0; attempt < 3; ++attempt) {
            build_refresh_frame(frame, m.slave_id);
            send_and_recv(serial_, frame, 30, rx_buf, sizeof(rx_buf), 10000);
        }
        std::fprintf(stderr, "%s (slave=%d master=0x%02X) connected\n",
                     m.name.c_str(), m.slave_id, m.master_id);
    }

    // Switch arm joints to MIT mode and enable
    for (size_t i = 0; i < 6; ++i) {
        const auto& m = cfg_.motors[i];
        build_switch_mode_frame(frame, m.slave_id, 1);
        send_and_recv(serial_, frame, 30, rx_buf, sizeof(rx_buf), 100000);
        build_enable_frame(frame, m.slave_id);
        send_and_recv(serial_, frame, 30, rx_buf, sizeof(rx_buf), 100000);
    }

    // Drain any leftover bytes from mode-switch / enable responses
    parser_.clear();
    sleep_ms(50);
    while (true) {
        size_t n = serial_.read_all(rx_buf, sizeof(rx_buf));
        if (n == 0) break;
    }

    // Read initial arm state — retry multiple times to ensure we get all 6 motors
    std::array<bool, 6> got_initial = {};
    std::array<int, 6> param_resp_count = {};  // track motors stuck returning param responses
    int total_got = 0;

    for (int round = 0; round < 8 && total_got < 6; ++round) {
        std::vector<std::array<uint8_t, RX_PACKET_LEN>> packets;
        for (size_t i = 0; i < 6; ++i) {
            if (got_initial[i]) continue;
            const auto& m = cfg_.motors[i];

            // If a motor keeps returning param responses, re-cycle it
            if (round > 0 && param_resp_count[i] >= 2) {
                std::fprintf(stderr, "  [init round %d] motor %zu (%s) stuck in param mode — "
                             "disable/re-enable\n", round, i, m.name.c_str());
                build_disable_frame(frame, m.slave_id);
                send_and_recv(serial_, frame, 30, rx_buf, sizeof(rx_buf), 50000);
                sleep_ms(50);
                build_switch_mode_frame(frame, m.slave_id, 1);
                send_and_recv(serial_, frame, 30, rx_buf, sizeof(rx_buf), 100000);
                build_enable_frame(frame, m.slave_id);
                send_and_recv(serial_, frame, 30, rx_buf, sizeof(rx_buf), 100000);
                sleep_ms(50);
                // Drain residual param ack packets
                while (serial_.read_all(rx_buf, sizeof(rx_buf)) > 0) {}
                parser_.clear();
                param_resp_count[i] = 0;
            }

            build_refresh_frame(frame, m.slave_id);
            size_t n = send_and_recv(serial_, frame, 30, rx_buf, sizeof(rx_buf), 10000);
            std::fprintf(stderr, "  [init round %d] refresh motor %zu (%s): rx %zu bytes\n",
                         round, i, m.name.c_str(), n);
            if (n > 0) {
                parser_.feed(rx_buf, n);
                parser_.extract(packets);
            }
        }

        for (const auto& pkt : packets) {
            uint32_t can_id = decode_can_id(pkt.data());
            uint8_t cmd = decode_cmd(pkt.data());
            if (cmd != 0x11) {
                std::fprintf(stderr, "  [init] skip packet can_id=0x%03X cmd=0x%02X (not motor state)\n", can_id, cmd);
                continue;
            }
            if (is_param_response(pkt.data())) {
                std::fprintf(stderr, "  [init] skip param response can_id=0x%03X\n", can_id);
                // Track which motor is stuck
                for (size_t i = 0; i < 6; ++i) {
                    if (got_initial[i]) continue;
                    const auto& m = cfg_.motors[i];
                    if (can_id == m.slave_id || can_id == m.master_id ||
                        (can_id == 0 && decode_response_id(pkt.data()) == (m.master_id & 0x0F))) {
                        ++param_resp_count[i];
                        break;
                    }
                }
                continue;
            }

            for (size_t i = 0; i < 6; ++i) {
                if (got_initial[i]) continue;
                const auto& m = cfg_.motors[i];
                bool match = (can_id == m.slave_id) ||
                             (can_id == m.master_id) ||
                             (can_id == 0 && decode_response_id(pkt.data()) == (m.master_id & 0x0F));
                if (match) {
                    auto st = decode_motor_state(pkt.data(), get_limits(m.type));
                    state_buf_.pos[i] = st.q;
                    state_buf_.vel[i] = st.dq;
                    state_buf_.torque[i] = st.tau;
                    state_buf_.t_mos[i] = st.t_mos;
                    state_buf_.t_rotor[i] = st.t_rotor;
                    got_initial[i] = true;
                    ++total_got;
                    std::fprintf(stderr, "  %s initial pos=%.4f vel=%.4f tau=%.4f (matched can_id=0x%03X)\n",
                                 m.name.c_str(), st.q, st.dq, st.tau, can_id);
                    break;
                }
            }
        }
    }

    if (total_got < cfg_.min_motors_required) {
        // Disable motors we already enabled before throwing
        for (size_t i = 0; i < 6; ++i) {
            build_disable_frame(frame, cfg_.motors[i].slave_id);
            send_and_recv(serial_, frame, 30, rx_buf, sizeof(rx_buf), 50000);
        }

        char hex_buf[16];
        std::ostringstream msg;
        msg << "Motor initialization failed: only " << total_got
            << "/6 arm motors responded (minimum required: "
            << cfg_.min_motors_required
            << "). Check that motors are powered on and CAN bus is connected.";
        for (size_t i = 0; i < 6; ++i) {
            if (!got_initial[i]) {
                std::snprintf(hex_buf, sizeof(hex_buf), "0x%02X", cfg_.motors[i].master_id);
                msg << "\n  MISSING: " << cfg_.motors[i].name
                    << " (slave=" << cfg_.motors[i].slave_id
                    << " master=" << hex_buf << ")";
            }
        }
        throw std::runtime_error(msg.str());
    } else if (total_got < 6) {
        std::fprintf(stderr, "WARNING: only got initial state for %d/6 arm motors (min_required=%d)!\n",
                     total_got, cfg_.min_motors_required);
        for (size_t i = 0; i < 6; ++i) {
            if (!got_initial[i]) {
                std::fprintf(stderr, "  MISSING: %s (slave=%d master=0x%02X) — pos will be 0!\n",
                             cfg_.motors[i].name.c_str(), cfg_.motors[i].slave_id,
                             cfg_.motors[i].master_id);
            }
        }
    } else {
        std::fprintf(stderr, "All 6 arm motors initial state read successfully\n");
    }

    parser_.clear();
}

void RtControlLoop::calibrate_gripper() {
    if (cfg_.motors.size() < 7) return;
    const auto& gm = cfg_.motors[6];

    uint8_t frame[30];
    uint8_t rx_buf[256];
    const auto& lim = get_limits(gm.type);

    build_switch_mode_frame(frame, gm.slave_id, 3);  // VEL
    send_and_recv(serial_, frame, 30, rx_buf, sizeof(rx_buf), 100000);

    build_enable_frame(frame, gm.slave_id);
    send_and_recv(serial_, frame, 30, rx_buf, sizeof(rx_buf), 100000);

    std::fprintf(stderr, "Gripper calibrating (opening until torque spike)...\n");
    std::vector<std::array<uint8_t, RX_PACKET_LEN>> packets;
    int cal_iterations = 0;
    constexpr int MAX_CAL_ITERATIONS = 2000;  // ~10s at 5ms per iteration
    constexpr float TORQUE_THRESHOLD = 0.7f;  // match Python threshold
    constexpr float CAL_VELOCITY = 5.0f;      // gentle homing speed

    const uint64_t cal_deadline_ns = now_ns() +
        static_cast<uint64_t>(cfg_.gripper_cal_timeout_s * 1e9);

    // Send VEL command ONCE to start the gripper moving (open direction)
    build_vel_frame(frame, gm.slave_id, CAL_VELOCITY);
    send_and_recv(serial_, frame, 30, rx_buf, sizeof(rx_buf), 5000);

    // Poll with refresh frames to read torque (like Python version)
    while (cal_iterations < MAX_CAL_ITERATIONS && now_ns() < cal_deadline_ns) {
        build_refresh_frame(frame, gm.slave_id);
        size_t n = send_and_recv(serial_, frame, 30, rx_buf, sizeof(rx_buf), 10000);

        if (n > 0) {
            parser_.feed(rx_buf, n);
            packets.clear();
            parser_.extract(packets);

            for (const auto& pkt : packets) {
                if (decode_cmd(pkt.data()) == 0x11) {
                    auto st = decode_motor_state(pkt.data(), lim);
                    ++cal_iterations;
                    if (cal_iterations % 20 == 0) {
                        std::fprintf(stderr, "  gripper cal [%d/%d]: pos=%.3f tau=%.3f (threshold=%.1f)\n",
                                     cal_iterations, MAX_CAL_ITERATIONS, st.q, st.tau, TORQUE_THRESHOLD);
                    }
                    if (st.tau > TORQUE_THRESHOLD) {
                        build_vel_frame(frame, gm.slave_id, 0.0f);
                        send_and_recv(serial_, frame, 30, rx_buf, sizeof(rx_buf), 50000);
                        build_disable_frame(frame, gm.slave_id);
                        send_and_recv(serial_, frame, 30, rx_buf, sizeof(rx_buf), 100000);
                        build_set_zero_frame(frame, gm.slave_id);
                        send_and_recv(serial_, frame, 30, rx_buf, sizeof(rx_buf), 200000);
                        sleep_ms(200);

                        // Back off slightly from the hard stop so the open
                        // position is force-free.  cfg_.gripper_open_pos is
                        // a small offset (default 0.0) in the close direction
                        // (negative = towards closed).
                        gripper_open_pos_ = cfg_.gripper_open_pos;

                        build_enable_frame(frame, gm.slave_id);
                        send_and_recv(serial_, frame, 30, rx_buf, sizeof(rx_buf), 100000);
                        goto calibration_done;
                    }
                }
            }
        }
        sleep_ms(10);  // match Python's time.sleep(0.01)
    }

    // Timeout — stop gripper and throw
    build_vel_frame(frame, gm.slave_id, 0.0f);
    send_and_recv(serial_, frame, 30, rx_buf, sizeof(rx_buf), 50000);
    build_disable_frame(frame, gm.slave_id);
    send_and_recv(serial_, frame, 30, rx_buf, sizeof(rx_buf), 100000);
    {
        std::string msg = "Gripper calibration timed out after " +
            std::to_string(cfg_.gripper_cal_timeout_s) + "s (" +
            std::to_string(cal_iterations) + " valid packets received). ";
        if (cal_iterations == 0) {
            msg += "No motor state packets were received — check that the gripper motor "
                   "(slave=" + std::to_string(gm.slave_id) + ") is powered on.";
        } else {
            msg += "Torque never exceeded threshold " + std::to_string(TORQUE_THRESHOLD) +
                   ". The gripper may be unobstructed or the threshold may need adjustment.";
        }
        throw std::runtime_error(msg);
    }

calibration_done:
    parser_.clear();

    build_switch_mode_frame(frame, gm.slave_id, 4);  // Torque_Pos
    send_and_recv(serial_, frame, 30, rx_buf, sizeof(rx_buf), 100000);

    // Read back actual position after set_zero + enable (like Python version)
    build_refresh_frame(frame, gm.slave_id);
    size_t n = send_and_recv(serial_, frame, 30, rx_buf, sizeof(rx_buf), 10000);
    if (n > 0) {
        std::vector<std::array<uint8_t, RX_PACKET_LEN>> pkts;
        parser_.feed(rx_buf, n);
        parser_.extract(pkts);
        for (const auto& pkt : pkts) {
            if (decode_cmd(pkt.data()) == 0x11) {
                auto st = decode_motor_state(pkt.data(), lim);
                state_buf_.pos[6] = st.q;
                state_buf_.vel[6] = st.dq;
                state_buf_.torque[6] = st.tau;
                state_buf_.t_mos[6] = st.t_mos;
                state_buf_.t_rotor[6] = st.t_rotor;
                // Use actual encoder reading + configured offset as open pos.
                // After set_zero, st.q is near 0.  The offset backs off from
                // the hard stop so the open position is force-free.
                gripper_open_pos_ = static_cast<double>(st.q) + cfg_.gripper_open_pos;
                std::fprintf(stderr, "  gripper read-back: pos=%.4f, open_pos=%.4f (offset=%.3f)\n",
                             st.q, gripper_open_pos_, cfg_.gripper_open_pos);
                break;
            }
        }
    }
    parser_.clear();
    std::fprintf(stderr, "Gripper calibrated: open position = %f\n", gripper_open_pos_);
}

// --- RT thread ---
//
// Pipeline strategy: responses from cycle N arrive during cycle N+1.
//   1. Read whatever responses have arrived (from previous cycle's commands)
//   2. Parse and update state
//   3. Compute control
//   4. Send all 7 commands in a burst (fast — just copies to kernel buffer)
//   5. Sleep to next period
//
// This avoids blocking on tcdrain or read_with_timeout in the hot path.
// State is one cycle (4ms) behind, which is acceptable for impedance control.

void RtControlLoop::rt_thread_func() {
    rt_active_ = apply_rt_scheduling(cfg_.rt_priority, cfg_.rt_cpu_affinity, cfg_.rt_use_mlockall, cfg_.loop_hz);
    if (rt_active_) {
        std::fprintf(stderr, "RT scheduling active (SCHED_FIFO priority %d)\n", cfg_.rt_priority);
    } else {
        std::fprintf(stderr, "RT scheduling not available, using default scheduler\n");
    }

    const uint64_t period_ns = static_cast<uint64_t>(1e9 / cfg_.loop_hz);
    const double target_us = 1e6 / cfg_.loop_hz;
    uint64_t next_wakeup = now_ns();

    std::array<double, 7> cur_pos = {};
    std::array<double, 7> cur_vel = {};
    std::array<double, 7> cur_torque = {};
    std::array<uint8_t, 7> cur_t_mos = {};
    std::array<uint8_t, 7> cur_t_rotor = {};

    for (int i = 0; i < 7; ++i) {
        cur_pos[static_cast<size_t>(i)] = state_buf_.pos[static_cast<size_t>(i)];
        cur_vel[static_cast<size_t>(i)] = state_buf_.vel[static_cast<size_t>(i)];
        cur_torque[static_cast<size_t>(i)] = state_buf_.torque[static_cast<size_t>(i)];
        cur_t_mos[static_cast<size_t>(i)]   = state_buf_.t_mos[static_cast<size_t>(i)];
        cur_t_rotor[static_cast<size_t>(i)] = state_buf_.t_rotor[static_cast<size_t>(i)];
    }

    uint8_t tx_frame[30];
    uint8_t rx_buf[512];
    std::vector<std::array<uint8_t, RX_PACKET_LEN>> packets;

    std::array<double, 6> pos_lo, pos_hi;
    for (int i = 0; i < 6; ++i) {
        pos_lo[static_cast<size_t>(i)] = cfg_.joint_pos_limits[static_cast<size_t>(2 * i)];
        pos_hi[static_cast<size_t>(i)] = cfg_.joint_pos_limits[static_cast<size_t>(2 * i + 1)];
    }

    const auto& dm4310_lim = get_limits(MotorType::DM4310);
    double gripper_vel_emit = static_cast<double>(dm4310_lim.dq_max) * cfg_.emit_velocity_scale;
    double gripper_i_des_emit = cfg_.max_gripper_torque_nm / cfg_.torque_constant * cfg_.emit_current_scale;

    uint64_t loop_count = 0;

    // Comm loss state machine
    bool comm_error_disable_sent = false;

    // Warmup phase: send refresh frames for a few cycles to fill the pipeline
    // before sending MIT commands. This ensures cur_pos is populated from
    // actual motor feedback, not zeros.
    constexpr int WARMUP_CYCLES = 10;

    while (running_.load(std::memory_order_acquire)) {
        uint64_t t0 = now_ns();
        ++loop_count;
        bool debug = false;

        // 1. Read responses from PREVIOUS cycle (non-blocking — grab everything available)
        size_t total_rx = 0;
        for (int pass = 0; pass < 3; ++pass) {
            size_t n = serial_.read_all(rx_buf, sizeof(rx_buf));
            if (n == 0) break;
            total_rx += n;
            parser_.feed(rx_buf, n);
        }

        // Communication loss tracking
        health_rt_.total_rx_bytes += total_rx;
        if (total_rx == 0) {
            ++health_rt_.consecutive_empty_cycles;
        } else {
            health_rt_.consecutive_empty_cycles = 0;
        }

        // 2. Parse received packets and update motor state
        packets.clear();
        parser_.extract(packets);

        if (debug) {
            std::fprintf(stderr, "[cycle %llu] rx_bytes=%zu packets=%zu\n",
                         (unsigned long long)loop_count, total_rx, packets.size());
        }

        uint8_t motor_responded = 0;  // bitmask, bit i = motor i responded this cycle

        for (const auto& pkt : packets) {
            uint32_t can_id = decode_can_id(pkt.data());
            uint8_t pkt_cmd = decode_cmd(pkt.data());
            if (pkt_cmd != 0x11) {
                if (debug) std::fprintf(stderr, "  skip pkt can_id=0x%03X cmd=0x%02X\n", can_id, pkt_cmd);
                continue;
            }
            if (is_param_response(pkt.data())) continue;

            bool matched = false;
            for (size_t i = 0; i < cfg_.motors.size(); ++i) {
                const auto& m = cfg_.motors[i];
                bool match = (can_id == m.slave_id) ||
                             (can_id == m.master_id) ||
                             (can_id == 0 && decode_response_id(pkt.data()) == (m.master_id & 0x0F));
                if (match) {
                    auto st = decode_motor_state(pkt.data(), get_limits(m.type));
                    cur_pos[i] = st.q;
                    cur_vel[i] = st.dq;
                    cur_torque[i] = st.tau;
                    cur_t_mos[i] = st.t_mos;
                    cur_t_rotor[i] = st.t_rotor;
                    if (i < 7) motor_responded |= (1u << i);
                    matched = true;
                    if (debug) {
                        std::fprintf(stderr, "  motor[%zu] %s: pos=%.4f vel=%.4f tau=%.4f (can_id=0x%03X)\n",
                                     i, m.name.c_str(), st.q, st.dq, st.tau, can_id);
                    }
                    break;
                }
            }
            if (!matched && debug) {
                std::fprintf(stderr, "  UNMATCHED pkt can_id=0x%03X\n", can_id);
            }
        }

        // Per-motor staleness tracking
        for (size_t i = 0; i < cfg_.motors.size() && i < 7; ++i) {
            if (motor_responded & (1u << i)) {
                health_rt_.motor_last_seen_cycle[i] = loop_count;
                health_rt_.motor_stale[i] = false;
            } else {
                if (loop_count - health_rt_.motor_last_seen_cycle[i] >=
                    static_cast<uint64_t>(cfg_.per_motor_stale_threshold)) {
                    health_rt_.motor_stale[i] = true;
                }
            }
        }

        // Comm error state machine (only active after warmup)
        if (loop_count > WARMUP_CYCLES) {
            if (!health_rt_.comm_loss) {
                if (health_rt_.consecutive_empty_cycles >= cfg_.max_consecutive_empty_cycles) {
                    health_rt_.comm_loss = true;
                    comm_error_disable_sent = false;
                    std::fprintf(stderr, "[cycle %llu] COMM ERROR: %d consecutive empty cycles, "
                                 "entering comm loss state (action=%s)\n",
                                 (unsigned long long)loop_count,
                                 health_rt_.consecutive_empty_cycles,
                                 cfg_.comm_loss_action == CommLossAction::DISABLE ? "DISABLE" : "HOLD");
                }
            }
        }

        // 3. Write state (seqlock) — so Python sees the latest even during computation
        {
            uint64_t s = state_seq_.load(std::memory_order_relaxed);
            state_seq_.store(s + 1, std::memory_order_release);
            state_buf_.pos = cur_pos;
            state_buf_.vel = cur_vel;
            state_buf_.torque = cur_torque;
            state_buf_.t_mos = cur_t_mos;
            state_buf_.t_rotor = cur_t_rotor;
            state_buf_.sag_residual = sag_residual_;   // from the previous cycle's step 8b
            state_buf_.sag_bias = sag_bias_;
            state_seq_.store(s + 2, std::memory_order_release);
        }

        // Publish health state (seqlock)
        health_rt_.loop_count = loop_count;
        {
            uint64_t s = health_seq_.load(std::memory_order_relaxed);
            health_seq_.store(s + 1, std::memory_order_release);
            health_buf_ = health_rt_;
            health_seq_.store(s + 2, std::memory_order_release);
        }

        // During warmup, send refresh frames only (no MIT commands).
        // This populates cur_pos from actual motor feedback before we start controlling.
        if (loop_count <= WARMUP_CYCLES) {
            for (size_t i = 0; i < cfg_.motors.size(); ++i) {
                const auto& m = cfg_.motors[i];
                build_refresh_frame(tx_frame, m.slave_id);
                if (!serial_.write(tx_frame, 30)) {
                    ++health_rt_.total_write_errors;
                }
                ++health_rt_.total_tx_frames;
            }
            if (debug) {
                std::fprintf(stderr, "  [warmup %llu/%d] sent refresh frames, cur_pos=[",
                             (unsigned long long)loop_count, WARMUP_CYCLES);
                for (int i = 0; i < 6; ++i) std::fprintf(stderr, "%.4f%s", cur_pos[static_cast<size_t>(i)], i<5?", ":"");
                std::fprintf(stderr, "]\n");
            }

            // At end of warmup, update command buffer and slew target to match actual positions
            if (loop_count == WARMUP_CYCLES) {
                uint64_t s = cmd_seq_.load(std::memory_order_relaxed);
                cmd_seq_.store(s + 1, std::memory_order_release);
                for (int i = 0; i < 6; ++i) {
                    cmd_buf_.q_des[static_cast<size_t>(i)] = cur_pos[static_cast<size_t>(i)];
                }
                cmd_buf_.timestamp_ns = now_ns();
                cmd_seq_.store(s + 2, std::memory_order_release);

                // Initialize slew target to actual motor positions
                for (int i = 0; i < 6; ++i) {
                    slew_target_[static_cast<size_t>(i)] = cur_pos[static_cast<size_t>(i)];
                }

                // Reset comm counters so warmup's empty cycles don't trigger comm loss
                health_rt_.consecutive_empty_cycles = 0;

                std::fprintf(stderr, "  [warmup done] command buffer and slew target initialized to actual pos\n");
            }

            uint64_t t1 = now_ns();
            double cycle_us = static_cast<double>(t1 - t0) / 1000.0;
            perf_.record(cycle_us, target_us);
            next_wakeup += period_ns;
            if (next_wakeup < t1) next_wakeup = t1 + period_ns;
            sleep_until_ns(next_wakeup);
            continue;
        }

        // 4. Handle error reset request (from Python thread)
        if (error_reset_requested_.exchange(false, std::memory_order_acq_rel)) {
            safety_state_.damping_mode = false;
            safety_state_.overcurrent_count = 0;
            safety_state_.overspeed_count = 0;
            health_rt_.comm_loss = false;
            health_rt_.consecutive_empty_cycles = 0;
            comm_error_disable_sent = false;
            // Snap slew target to current position to prevent jump after reset
            for (int i = 0; i < 6; ++i) {
                slew_target_[static_cast<size_t>(i)] = cur_pos[static_cast<size_t>(i)];
            }
            // A fault may have been caused by (or corrupted) the learned sag bias — restart clean
            sag_bias_ = {};
            error_reset_ack_.fetch_add(1, std::memory_order_release);
            std::fprintf(stderr, "[cycle %llu] Error reset acknowledged\n",
                         (unsigned long long)loop_count);
        }

        // 5. Read commands (seqlock)
        CommandBuffer cmd;
        for (int attempt = 0; attempt < 100; ++attempt) {
            uint64_t s1 = cmd_seq_.load(std::memory_order_acquire);
            if (s1 & 1) continue;
            cmd = cmd_buf_;
            uint64_t s2 = cmd_seq_.load(std::memory_order_acquire);
            if (s1 == s2) break;
        }

        // 6. Watchdog
        uint64_t cmd_age_ns = t0 - cmd.timestamp_ns;
        double cmd_age_s = static_cast<double>(cmd_age_ns) / 1e9;
        if (cmd_age_s > cfg_.command_timeout_s) {
            for (int i = 0; i < 6; ++i) {
                cmd.q_des[static_cast<size_t>(i)] = cur_pos[static_cast<size_t>(i)];
            }
        }

        // 7. Slew rate limiting
        std::array<bool, 6> cmd_moving;   // did this joint's command move this cycle? (for 8b)
        for (int i = 0; i < 6; ++i) {
            const double before = slew_target_[static_cast<size_t>(i)];
            double max_delta = cfg_.max_pos_delta_per_cycle[static_cast<size_t>(i)];
            if (max_delta > 0.0) {
                double diff = cmd.q_des[static_cast<size_t>(i)] - slew_target_[static_cast<size_t>(i)];
                if (diff > max_delta) diff = max_delta;
                else if (diff < -max_delta) diff = -max_delta;
                slew_target_[static_cast<size_t>(i)] += diff;
            } else {
                // Rate limiting disabled for this joint — pass through
                slew_target_[static_cast<size_t>(i)] = cmd.q_des[static_cast<size_t>(i)];
            }
            cmd_moving[static_cast<size_t>(i)] =
                std::abs(slew_target_[static_cast<size_t>(i)] - before) > 1e-9;
        }

        // 8. Gravity compensation — evaluated at the MODEL configuration (hardware q +
        // calibrated zero offset), not the raw encoder reading
        std::array<double, 6> tau_ff = {};
        if (grav_comp_.is_loaded()) {
            std::array<double, 6> q_model;
            for (int i = 0; i < 6; ++i) {
                q_model[static_cast<size_t>(i)] =
                    cur_pos[static_cast<size_t>(i)] + cfg_.gravity_q_offset[static_cast<size_t>(i)];
            }
            grav_comp_.compute(q_model.data(), tau_ff.data());
            for (int i = 0; i < 6; ++i) {
                tau_ff[static_cast<size_t>(i)] *= cfg_.gravity_comp_scale;
            }
        }

        // 8b. FR-018 sag observer + friction dither (see control_loop.h for the contract).
        // Runs on the pre-safety tau_ff; apply_safety still clamps the sum to the joint
        // torque limits and zeroes it in damping mode.
        for (int i = 0; i < 6; ++i) {
            sag_residual_[static_cast<size_t>(i)] = cfg_.default_kp[static_cast<size_t>(i)] *
                (slew_target_[static_cast<size_t>(i)] - cur_pos[static_cast<size_t>(i)]);
        }
        const bool adapt_ok = !safety_state_.damping_mode && !health_rt_.comm_loss;
        if (cfg_.sag_observer_enable && adapt_ok) {
            for (int i = 0; i < 6; ++i) {
                const size_t j = static_cast<size_t>(i);
                sag_bias_[j] *= (1.0 - cfg_.sag_leak);
                // adapt only while HOLDING: a joint stick-slipping through a slow commanded
                // move has |vel|~0 during every stick phase — adapting there deepens the
                // sawtooth until the breakaway looks like a contact jump (false guard trips)
                if (!cmd_moving[j] && std::abs(cur_vel[j]) < cfg_.sag_vel_eps &&
                    std::abs(sag_residual_[j]) < cfg_.sag_freeze_residual_nm) {
                    sag_bias_[j] = std::clamp(sag_bias_[j] + cfg_.sag_lambda * sag_residual_[j],
                                              -cfg_.sag_max_nm, cfg_.sag_max_nm);
                }
                tau_ff[j] += sag_bias_[j];
            }
        }
        if (cfg_.dither_enable && adapt_ok) {
            const double s = std::sin(dither_phase_);
            dither_phase_ += 2.0 * M_PI * cfg_.dither_hz / cfg_.loop_hz;
            if (dither_phase_ > 2.0 * M_PI) dither_phase_ -= 2.0 * M_PI;
            for (int i = 0; i < 6; ++i) {
                const size_t j = static_cast<size_t>(i);
                // only a joint that is STUCK while HOLDING (error beyond encoder noise, not
                // moving, command stationary) gets dithered — a converged, moving, or
                // commanded-through joint stays clean
                if (!cmd_moving[j] &&
                    std::abs(slew_target_[j] - cur_pos[j]) > cfg_.dither_pos_eps &&
                    std::abs(cur_vel[j]) < cfg_.sag_vel_eps) {
                    tau_ff[j] += cfg_.dither_amp_nm[j] * s;
                }
            }
        }

        // 9. Safety (uses slew-limited target, not raw q_des)
        std::array<double, 6> kp = cfg_.default_kp;
        std::array<double, 6> q_des;
        std::copy(slew_target_.begin(), slew_target_.end(), q_des.begin());
        apply_safety(cur_pos.data(), cur_vel.data(), cur_torque.data(),
                     q_des.data(), kp.data(), tau_ff.data(),
                     pos_lo.data(), pos_hi.data(),
                     cfg_.joint_torque_limits.data(), cfg_.joint_velocity_limits.data(),
                     cfg_.limit_buffer,
                     cfg_.overcurrent_threshold, cfg_.overspeed_threshold,
                     6, safety_state_);

        // Update health with safety state
        health_rt_.damping_mode = safety_state_.damping_mode;
        health_rt_.overcurrent_count = safety_state_.overcurrent_count;
        health_rt_.overspeed_count = safety_state_.overspeed_count;

        if (debug) {
            std::fprintf(stderr, "  [cycle %llu] q_des=[", (unsigned long long)loop_count);
            for (int i = 0; i < 6; ++i) std::fprintf(stderr, "%.4f%s", q_des[static_cast<size_t>(i)], i<5?", ":"");
            std::fprintf(stderr, "] cur_pos=[");
            for (int i = 0; i < 6; ++i) std::fprintf(stderr, "%.4f%s", cur_pos[static_cast<size_t>(i)], i<5?", ":"");
            std::fprintf(stderr, "] tau_ff=[");
            for (int i = 0; i < 6; ++i) std::fprintf(stderr, "%.3f%s", tau_ff[static_cast<size_t>(i)], i<5?", ":"");
            std::fprintf(stderr, "] cmd_age=%.3fs\n", cmd_age_s);
        }

        // 10. Send commands — gated by comm error state
        if (health_rt_.comm_loss) {
            if (cfg_.comm_loss_action == CommLossAction::DISABLE && !comm_error_disable_sent) {
                // Send disable frames ONCE when first entering error state
                for (size_t i = 0; i < cfg_.motors.size(); ++i) {
                    build_disable_frame(tx_frame, cfg_.motors[i].slave_id);
                    if (!serial_.write(tx_frame, 30)) {
                        ++health_rt_.total_write_errors;
                    }
                    ++health_rt_.total_tx_frames;
                }
                comm_error_disable_sent = true;
            } else if (cfg_.comm_loss_action == CommLossAction::HOLD) {
                // Send MIT with kp=0 (damping only) to gently decelerate
                for (size_t i = 0; i < 6 && i < cfg_.motors.size(); ++i) {
                    const auto& m = cfg_.motors[i];
                    const auto& lim = get_limits(m.type);
                    build_mit_frame(tx_frame, m.slave_id,
                                   0.0f,  // kp = 0
                                   static_cast<float>(cfg_.default_kd[i]),
                                   static_cast<float>(cur_pos[i]),
                                   0.0f, 0.0f, lim);
                    if (!serial_.write(tx_frame, 30)) {
                        ++health_rt_.total_write_errors;
                    }
                    ++health_rt_.total_tx_frames;
                }
            }
            // In DISABLE mode after first cycle: send nothing (motors already disabled)
        } else {
            // Normal operation — send MIT frames for arm joints
            for (size_t i = 0; i < 6 && i < cfg_.motors.size(); ++i) {
                const auto& m = cfg_.motors[i];
                const auto& lim = get_limits(m.type);
                build_mit_frame(tx_frame, m.slave_id,
                               static_cast<float>(kp[i]),
                               static_cast<float>(cfg_.default_kd[i]),
                               static_cast<float>(q_des[i]),
                               0.0f,
                               static_cast<float>(tau_ff[i]),
                               lim);
                if (!serial_.write(tx_frame, 30)) {
                    ++health_rt_.total_write_errors;
                }
                ++health_rt_.total_tx_frames;
            }

            if (cfg_.motors.size() >= 7) {
                const auto& gm = cfg_.motors[6];
                double gripper_q = gripper_open_pos_ +
                    cmd.gripper_des * (cfg_.gripper_closed_pos - gripper_open_pos_);
                build_emit_frame(tx_frame, gm.slave_id,
                                static_cast<float>(gripper_q),
                                static_cast<float>(gripper_vel_emit),
                                static_cast<float>(gripper_i_des_emit));
                if (!serial_.write(tx_frame, 30)) {
                    ++health_rt_.total_write_errors;
                }
                ++health_rt_.total_tx_frames;
            }
        }

        // 11. Record perf
        uint64_t t1 = now_ns();
        double cycle_us = static_cast<double>(t1 - t0) / 1000.0;
        perf_.record(cycle_us, target_us);

        // 12. Sleep to next period
        next_wakeup += period_ns;
        if (next_wakeup < t1) {
            next_wakeup = t1 + period_ns;
        }
        sleep_until_ns(next_wakeup);
    }
}

} // namespace trlc
