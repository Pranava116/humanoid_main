"""SimBackend: MuJoCo with the MG996R actuator model, same interface as the real robot."""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional, Tuple

import numpy as np

from robot.config import Config
from robot.hal.actuator_model import MG996R
from robot.hal.base import Backend, BackendInfo, ImuReading
from robot.hal.mapping import joint_range
from robot.rotations import quat_to_rpy

log = logging.getLogger(__name__)


def ensure_scene(cfg: Config) -> str:
    """Path of models/scene.xml, generating it from the URDF when it is missing or older."""
    scene = cfg.path(cfg.require("robot.sim.scene"))
    urdf = cfg.path(cfg.require("robot.sim.urdf"))
    if not os.path.exists(scene) or (os.path.exists(urdf) and os.path.getmtime(urdf) > os.path.getmtime(scene)):
        from robot.sim.scene import SceneOptions, write_scene
        log.info("generating %s from %s", scene, urdf)
        write_scene(urdf, scene, SceneOptions(actuator=cfg.actuator,
                                              physics_dt=float(cfg.get("robot.sim.physics_dt", 0.002))))
    return scene


class SimBackend(Backend):
    """MuJoCo physics. ``step(dt)`` runs dt / physics_dt substeps through the servo model."""

    name = "sim"

    def __init__(self, cfg: Config, viewer: bool = False, scene: Optional[str] = None,
                 seed: Optional[int] = None, camera_size: Optional[Tuple[int, int]] = None) -> None:
        import mujoco

        super().__init__(cfg)
        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(scene or ensure_scene(cfg))
        self.data = mujoco.MjData(self.model)
        self.physics_dt = float(self.model.opt.timestep)
        self.rng = np.random.default_rng(seed)

        names = self.joints.names
        self.act_ids = np.array([self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in names])
        jnt_ids = [self._id(mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
        self.qpos_adr = np.array([self.model.jnt_qposadr[j] for j in jnt_ids])
        self.qvel_adr = np.array([self.model.jnt_dofadr[j] for j in jnt_ids])
        rng = [joint_range(self.joints[n]) for n in names]
        self.lo = np.array([r[0] for r in rng])
        self.hi = np.array([r[1] for r in rng])
        self.apply_actuator_params(cfg.actuator)

        self.servo = MG996R(cfg.actuator, len(names), self.physics_dt)
        self.cmd = np.array([self.joints[n].default_rad for n in names])
        self.released = False
        self._sensor = {s: self._sensor_slice(s) for s in ("imu_quat", "imu_gyro", "imu_accel")}
        noise = cfg.get("robot.sim.imu_noise", {}) or {}
        self.gyro_std = float(noise.get("gyro_std", 0.0))
        self.accel_std = float(noise.get("accel_std", 0.0))

        cam = cfg.get("robot.sim.camera_name", "head_cam")
        self.cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam)
        w = int(cfg.get("robot.camera.width", 320))
        h = int(cfg.get("robot.camera.height", 240))
        self.camera_size = camera_size or (h, w)
        self._renderer = None
        self._person_mocap = self._mocap_index("person")

        self.viewer = None
        if viewer:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.reset()

    # ---------------------------------------------------------------- setup
    def _id(self, objtype, name: str) -> int:
        i = self.mj.mj_name2id(self.model, objtype, name)
        if i < 0:
            raise ValueError(f"'{name}' not found in the MuJoCo model (regenerate models/scene.xml?)")
        return i

    def _sensor_slice(self, name: str) -> slice:
        i = self._id(self.mj.mjtObj.mjOBJ_SENSOR, name)
        adr, dim = self.model.sensor_adr[i], self.model.sensor_dim[i]
        return slice(adr, adr + dim)

    def _mocap_index(self, body: str) -> int:
        b = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_BODY, body)
        return int(self.model.body_mocapid[b]) if b >= 0 else -1

    def reset(self, pose: Optional[Dict[str, float]] = None) -> None:
        """Back to the 'stand' keyframe (or ``pose``), servos holding it."""
        key = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_KEY, "stand")
        if key >= 0:
            self.mj.mj_resetDataKeyframe(self.model, self.data, key)
        else:
            self.mj.mj_resetData(self.model, self.data)
        if pose:
            for n, q in pose.items():
                self.cmd[self.joints.names.index(n)] = q
        q0 = np.clip(self.cmd, self.lo, self.hi)
        self.data.qpos[self.qpos_adr] = q0
        self.data.ctrl[self.act_ids] = q0
        self.servo.reset(q0)
        self._set_gains(1.0)
        self.released = False
        self.mj.mj_forward(self.model, self.data)
        self.time = 0.0

    def apply_actuator_params(self, act) -> None:
        """Push MG996R params (kp, kv, torque, joint damping/armature/friction) into the model.

        The scene file has the same values baked in, but applying them here means config
        changes (and RL domain randomisation) take effect without regenerating scene.xml.
        """
        m = self.model
        self.kp0 = np.full(len(self.act_ids), act.kp)
        self.kv0 = np.full(len(self.act_ids), act.kv)
        m.actuator_forcerange[self.act_ids] = (-act.torque_limit, act.torque_limit)
        m.actuator_forcelimited[self.act_ids] = 1
        m.dof_damping[self.qvel_adr] = act.damping
        m.dof_armature[self.qvel_adr] = act.armature
        m.dof_frictionloss[self.qvel_adr] = act.frictionloss
        self._set_gains(1.0)

    def _set_gains(self, scale: float) -> None:
        self.model.actuator_gainprm[self.act_ids, 0] = self.kp0 * scale
        self.model.actuator_biasprm[self.act_ids, 1] = -self.kp0 * scale
        self.model.actuator_biasprm[self.act_ids, 2] = -self.kv0 * scale

    # ---------------------------------------------------------------- interface
    def set_joint_targets(self, targets: Dict[str, float]) -> None:
        names = self.joints.names
        for n, q in targets.items():
            self.cmd[names.index(n)] = q
        np.clip(self.cmd, self.lo, self.hi, out=self.cmd)
        if self.released:
            self._set_gains(1.0)
            self.released = False

    def step(self, dt: float) -> None:
        n = max(1, int(round(dt / self.physics_dt)))
        for _ in range(n):
            q = self.data.qpos[self.qpos_adr]
            self.data.ctrl[self.act_ids] = self.servo.step(self.cmd, q)
            self.mj.mj_step(self.model, self.data)
        self.time = float(self.data.time)
        if self.viewer is not None:
            if self.viewer.is_running():
                self.viewer.sync()

    def get_imu(self) -> ImuReading:
        s = self.data.sensordata
        quat = tuple(float(x) for x in s[self._sensor["imu_quat"]])
        gyro = s[self._sensor["imu_gyro"]].copy()
        accel = s[self._sensor["imu_accel"]].copy()
        if self.gyro_std > 0:
            gyro += self.rng.normal(0, self.gyro_std, 3)
        if self.accel_std > 0:
            accel += self.rng.normal(0, self.accel_std, 3)
        return ImuReading(quat=quat, rpy=quat_to_rpy(quat), gyro=tuple(gyro), accel=tuple(accel), t=self.time)

    def get_camera_frame(self) -> Optional[np.ndarray]:
        if self.cam_id < 0:
            return None
        if self._renderer is None:
            try:
                h, w = self.camera_size
                self._renderer = self.mj.Renderer(self.model, height=h, width=w)
            except Exception as e:  # no OpenGL context (headless CI)
                log.warning("sim camera unavailable: %s", e)
                self.cam_id = -1
                return None
        self._renderer.update_scene(self.data, camera=self.cam_id)
        return self._renderer.render()

    def release(self) -> None:
        self._set_gains(0.0)
        self.released = True

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def info(self) -> BackendInfo:
        return BackendInfo(name=self.name, joints=self.joints.names, has_camera=self.cam_id >= 0)

    # ---------------------------------------------------------------- sim-only helpers
    def joint_positions(self) -> Dict[str, float]:
        """Measured joint angles (sim only; the real robot has no encoders)."""
        q = self.data.qpos[self.qpos_adr]
        return {n: float(v) for n, v in zip(self.joints.names, q)}

    def torso_position(self) -> np.ndarray:
        return self.data.qpos[0:3].copy()

    def torso_velocity(self) -> np.ndarray:
        """World-frame linear velocity of the torso."""
        return self.data.qvel[0:3].copy()

    def push(self, force: Tuple[float, float, float], duration: float = 0.1) -> None:
        """Apply a world-frame force (N) to the torso for ``duration`` seconds of sim time."""
        torso = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_BODY, "torso")
        self.data.xfrc_applied[torso, :3] = force
        self.step(duration)
        self.data.xfrc_applied[torso, :3] = 0.0

    def set_person(self, x: float, y: float) -> None:
        """Move the mocap person (no-op if the scene has none)."""
        if self._person_mocap >= 0:
            self.data.mocap_pos[self._person_mocap, :2] = (x, y)

    def person_position(self) -> Optional[np.ndarray]:
        if self._person_mocap < 0:
            return None
        return self.data.mocap_pos[self._person_mocap].copy()
