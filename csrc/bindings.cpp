#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/array.h>
#include <nanobind/ndarray.h>

#include "control_loop.h"
#include "rt_utils.h"

namespace nb = nanobind;
using namespace trlc;

NB_MODULE(_trlc_dk1_rt, m) {
    m.doc() = "TRLC-DK1 real-time control loop (C++ with nanobind)";

    // MotorType enum
    nb::enum_<MotorType>(m, "MotorType")
        .value("DM4310", MotorType::DM4310)
        .value("DM4310_48V", MotorType::DM4310_48V)
        .value("DM4340", MotorType::DM4340)
        .value("DM4340_48V", MotorType::DM4340_48V)
        .value("DM6006", MotorType::DM6006)
        .value("DM8006", MotorType::DM8006)
        .value("DM8009", MotorType::DM8009)
        .value("DM10010L", MotorType::DM10010L)
        .value("DM10010", MotorType::DM10010)
        .value("DMH3510", MotorType::DMH3510)
        .value("DMH6215", MotorType::DMH6215)
        .value("DMG6220", MotorType::DMG6220);

    // CommLossAction enum
    nb::enum_<CommLossAction>(m, "CommLossAction")
        .value("HOLD", CommLossAction::HOLD)
        .value("DISABLE", CommLossAction::DISABLE);

    // MotorDescriptor
    nb::class_<MotorDescriptor>(m, "MotorDescriptor")
        .def(nb::init<>())
        .def_rw("name", &MotorDescriptor::name)
        .def_rw("type", &MotorDescriptor::type)
        .def_rw("slave_id", &MotorDescriptor::slave_id)
        .def_rw("master_id", &MotorDescriptor::master_id);

    // PerfSnapshot
    nb::class_<PerfSnapshot>(m, "PerfSnapshot")
        .def(nb::init<>())
        .def_ro("loop_count", &PerfSnapshot::loop_count)
        .def_ro("min_cycle_us", &PerfSnapshot::min_cycle_us)
        .def_ro("max_cycle_us", &PerfSnapshot::max_cycle_us)
        .def_ro("mean_cycle_us", &PerfSnapshot::mean_cycle_us)
        .def_ro("deadline_misses", &PerfSnapshot::deadline_misses)
        .def_prop_ro("histogram", [](const PerfSnapshot& s) {
            return nb::ndarray<nb::numpy, const uint64_t, nb::shape<6>>(
                s.histogram.data(), {6});
        });

    // RtLoopConfig
    nb::class_<RtLoopConfig>(m, "RtLoopConfig")
        .def(nb::init<>())
        .def_rw("serial_port", &RtLoopConfig::serial_port)
        .def_rw("loop_hz", &RtLoopConfig::loop_hz)
        .def_rw("motors", &RtLoopConfig::motors)
        .def_rw("limit_buffer", &RtLoopConfig::limit_buffer)
        .def_rw("model_path", &RtLoopConfig::model_path)
        .def_rw("gravity_comp_scale", &RtLoopConfig::gravity_comp_scale)
        .def_rw("sag_observer_enable", &RtLoopConfig::sag_observer_enable)
        .def_rw("sag_lambda", &RtLoopConfig::sag_lambda)
        .def_rw("sag_max_nm", &RtLoopConfig::sag_max_nm)
        .def_rw("sag_freeze_residual_nm", &RtLoopConfig::sag_freeze_residual_nm)
        .def_rw("sag_vel_eps", &RtLoopConfig::sag_vel_eps)
        .def_rw("sag_leak", &RtLoopConfig::sag_leak)
        .def_rw("dither_enable", &RtLoopConfig::dither_enable)
        .def_rw("dither_hz", &RtLoopConfig::dither_hz)
        .def_rw("dither_pos_eps", &RtLoopConfig::dither_pos_eps)
        .def_rw("command_timeout_s", &RtLoopConfig::command_timeout_s)
        .def_rw("overcurrent_threshold", &RtLoopConfig::overcurrent_threshold)
        .def_rw("overspeed_threshold", &RtLoopConfig::overspeed_threshold)
        .def_rw("min_motors_required", &RtLoopConfig::min_motors_required)
        .def_rw("gripper_open_pos", &RtLoopConfig::gripper_open_pos)
        .def_rw("gripper_closed_pos", &RtLoopConfig::gripper_closed_pos)
        .def_rw("max_gripper_torque_nm", &RtLoopConfig::max_gripper_torque_nm)
        .def_rw("torque_constant", &RtLoopConfig::torque_constant)
        .def_rw("emit_velocity_scale", &RtLoopConfig::emit_velocity_scale)
        .def_rw("emit_current_scale", &RtLoopConfig::emit_current_scale)
        .def_rw("gripper_cal_timeout_s", &RtLoopConfig::gripper_cal_timeout_s)
        .def_rw("disable_torque_on_disconnect", &RtLoopConfig::disable_torque_on_disconnect)
        .def_rw("rt_priority", &RtLoopConfig::rt_priority)
        .def_rw("rt_cpu_affinity", &RtLoopConfig::rt_cpu_affinity)
        .def_rw("rt_use_mlockall", &RtLoopConfig::rt_use_mlockall)
        .def_rw("max_consecutive_empty_cycles", &RtLoopConfig::max_consecutive_empty_cycles)
        .def_rw("comm_loss_action", &RtLoopConfig::comm_loss_action)
        .def_rw("per_motor_stale_threshold", &RtLoopConfig::per_motor_stale_threshold)
        // std::array properties exposed via lambdas for numpy compatibility
        .def_prop_rw("default_kp",
            [](const RtLoopConfig& c) {
                return nb::ndarray<nb::numpy, const double, nb::shape<6>>(
                    c.default_kp.data(), {6});
            },
            [](RtLoopConfig& c, nb::ndarray<nb::numpy, const double, nb::shape<6>> arr) {
                const double* p = arr.data();
                for (int i = 0; i < 6; ++i) c.default_kp[static_cast<size_t>(i)] = p[i];
            })
        .def_prop_rw("default_kd",
            [](const RtLoopConfig& c) {
                return nb::ndarray<nb::numpy, const double, nb::shape<6>>(
                    c.default_kd.data(), {6});
            },
            [](RtLoopConfig& c, nb::ndarray<nb::numpy, const double, nb::shape<6>> arr) {
                const double* p = arr.data();
                for (int i = 0; i < 6; ++i) c.default_kd[static_cast<size_t>(i)] = p[i];
            })
        .def_prop_rw("joint_pos_limits",
            [](const RtLoopConfig& c) {
                return nb::ndarray<nb::numpy, const double, nb::shape<12>>(
                    c.joint_pos_limits.data(), {12});
            },
            [](RtLoopConfig& c, nb::ndarray<nb::numpy, const double, nb::shape<12>> arr) {
                const double* p = arr.data();
                for (int i = 0; i < 12; ++i) c.joint_pos_limits[static_cast<size_t>(i)] = p[i];
            })
        .def_prop_rw("joint_torque_limits",
            [](const RtLoopConfig& c) {
                return nb::ndarray<nb::numpy, const double, nb::shape<6>>(
                    c.joint_torque_limits.data(), {6});
            },
            [](RtLoopConfig& c, nb::ndarray<nb::numpy, const double, nb::shape<6>> arr) {
                const double* p = arr.data();
                for (int i = 0; i < 6; ++i) c.joint_torque_limits[static_cast<size_t>(i)] = p[i];
            })
        .def_prop_rw("joint_velocity_limits",
            [](const RtLoopConfig& c) {
                return nb::ndarray<nb::numpy, const double, nb::shape<6>>(
                    c.joint_velocity_limits.data(), {6});
            },
            [](RtLoopConfig& c, nb::ndarray<nb::numpy, const double, nb::shape<6>> arr) {
                const double* p = arr.data();
                for (int i = 0; i < 6; ++i) c.joint_velocity_limits[static_cast<size_t>(i)] = p[i];
            })
        .def_prop_rw("max_pos_delta_per_cycle",
            [](const RtLoopConfig& c) {
                return nb::ndarray<nb::numpy, const double, nb::shape<6>>(
                    c.max_pos_delta_per_cycle.data(), {6});
            },
            [](RtLoopConfig& c, nb::ndarray<nb::numpy, const double, nb::shape<6>> arr) {
                const double* p = arr.data();
                for (int i = 0; i < 6; ++i) c.max_pos_delta_per_cycle[static_cast<size_t>(i)] = p[i];
            })
        .def_prop_rw("gravity_q_offset",
            [](const RtLoopConfig& c) {
                return nb::ndarray<nb::numpy, const double, nb::shape<6>>(
                    c.gravity_q_offset.data(), {6});
            },
            [](RtLoopConfig& c, nb::ndarray<nb::numpy, const double, nb::shape<6>> arr) {
                const double* p = arr.data();
                for (int i = 0; i < 6; ++i) c.gravity_q_offset[static_cast<size_t>(i)] = p[i];
            })
        .def_prop_rw("dither_amp_nm",
            [](const RtLoopConfig& c) {
                return nb::ndarray<nb::numpy, const double, nb::shape<6>>(
                    c.dither_amp_nm.data(), {6});
            },
            [](RtLoopConfig& c, nb::ndarray<nb::numpy, const double, nb::shape<6>> arr) {
                const double* p = arr.data();
                for (int i = 0; i < 6; ++i) c.dither_amp_nm[static_cast<size_t>(i)] = p[i];
            });

    // JointState
    nb::class_<JointState>(m, "JointState")
        .def_prop_ro("pos", [](const JointState& s) {
            return nb::ndarray<nb::numpy, const double, nb::shape<6>>(s.pos.data(), {6});
        })
        .def_prop_ro("vel", [](const JointState& s) {
            return nb::ndarray<nb::numpy, const double, nb::shape<6>>(s.vel.data(), {6});
        })
        .def_prop_ro("torque", [](const JointState& s) {
            return nb::ndarray<nb::numpy, const double, nb::shape<6>>(s.torque.data(), {6});
        })
        .def_prop_ro("t_mos", [](const JointState& s) {
            return nb::ndarray<nb::numpy, const uint8_t, nb::shape<6>>(s.t_mos.data(), {6});
        })
        .def_prop_ro("t_rotor", [](const JointState& s) {
            return nb::ndarray<nb::numpy, const uint8_t, nb::shape<6>>(s.t_rotor.data(), {6});
        })
        .def_prop_ro("sag_residual", [](const JointState& s) {
            return nb::ndarray<nb::numpy, const double, nb::shape<6>>(s.sag_residual.data(), {6});
        })
        .def_prop_ro("sag_bias", [](const JointState& s) {
            return nb::ndarray<nb::numpy, const double, nb::shape<6>>(s.sag_bias.data(), {6});
        });

    // GripperState
    nb::class_<GripperState>(m, "GripperState")
        .def_ro("pos", &GripperState::pos)
        .def_ro("torque", &GripperState::torque);

    // HealthState
    nb::class_<HealthState>(m, "HealthState")
        .def(nb::init<>())
        .def_ro("damping_mode", &HealthState::damping_mode)
        .def_ro("overcurrent_count", &HealthState::overcurrent_count)
        .def_ro("overspeed_count", &HealthState::overspeed_count)
        .def_ro("comm_loss", &HealthState::comm_loss)
        .def_ro("consecutive_empty_cycles", &HealthState::consecutive_empty_cycles)
        .def_ro("total_rx_bytes", &HealthState::total_rx_bytes)
        .def_ro("total_tx_frames", &HealthState::total_tx_frames)
        .def_ro("total_write_errors", &HealthState::total_write_errors)
        .def_ro("loop_count", &HealthState::loop_count)
        .def_prop_ro("motor_last_seen_cycle", [](const HealthState& h) {
            return nb::ndarray<nb::numpy, const uint64_t, nb::shape<7>>(
                h.motor_last_seen_cycle.data(), {7});
        })
        .def_prop_ro("motor_stale", [](const HealthState& h) {
            nb::list result;
            for (int i = 0; i < 7; ++i) result.append(h.motor_stale[i]);
            return result;
        });

    // RtControlLoop
    nb::class_<RtControlLoop>(m, "RtControlLoop")
        .def(nb::init<const RtLoopConfig&>())
        .def("start", &RtControlLoop::start)
        .def("stop", &RtControlLoop::stop)
        .def("command_joint_pos", [](RtControlLoop& loop,
                nb::ndarray<nb::numpy, const double, nb::shape<6>> q) {
            loop.command_joint_pos(q.data());
        })
        .def("command_gripper", &RtControlLoop::command_gripper)
        .def("get_joint_state", &RtControlLoop::get_joint_state)
        .def("get_gripper_state", &RtControlLoop::get_gripper_state)
        .def("get_health", &RtControlLoop::get_health)
        .def("reset_errors", &RtControlLoop::reset_errors, nb::arg("timeout_ms") = 100)
        .def("get_perf", &RtControlLoop::get_perf)
        .def("read_cycle_times", [](const RtControlLoop& loop, size_t max_count) {
            std::vector<float> buf(max_count);
            size_t n = loop.read_cycle_times(buf.data(), max_count);
            buf.resize(n);
            // Return as numpy array
            auto* data = new float[n];
            std::copy(buf.begin(), buf.end(), data);
            nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<float*>(p); });
            return nb::ndarray<nb::numpy, float>(data, {n}, owner);
        })
        .def("reset_perf", &RtControlLoop::reset_perf, nb::arg("timeout_ms") = 100)
        .def("is_running", &RtControlLoop::is_running)
        .def("is_rt_active", &RtControlLoop::is_rt_active);

    // Free functions
    m.def("detect_rt_kernel", &detect_rt_kernel,
          "Check if running on a PREEMPT_RT kernel");
}
