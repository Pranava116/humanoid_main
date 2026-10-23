"""One servo: logical angle -> hardware angle -> pulse width."""

import warnings


class UncalibratedError(RuntimeError):
    """Raised when you try to move a servo whose offset is still null."""


def _clamp(value, lo, hi):
    return max(lo, min(hi, value))


class Servo:
    def __init__(self, name, pca, channel, offset, invert=False,
                 min_angle=-60, max_angle=60, home=0,
                 center=90, min_pulse=500, max_pulse=2500):
        self.name = name
        self.pca = pca
        self.channel = channel
        self.offset = offset            # None = not calibrated
        self.invert = invert
        self.min_angle = min_angle      # logical limits
        self.max_angle = max_angle
        self.home = home
        self.center = center
        self.min_pulse = min_pulse
        self.max_pulse = max_pulse
        self.angle = None               # last commanded logical angle

    @classmethod
    def from_config(cls, name, cfg, pca, defaults):
        """Build from a servos.yaml entry, falling back to servo_defaults."""
        def pick(key, fallback):
            return cfg.get(key, defaults.get(key, fallback))
        return cls(
            name=name,
            pca=pca,
            channel=cfg["channel"],
            offset=cfg.get("offset"),
            invert=cfg.get("invert", False),
            min_angle=cfg.get("min_angle", -60),
            max_angle=cfg.get("max_angle", 60),
            home=cfg.get("home", 0),
            center=pick("center", 90),
            min_pulse=pick("min_pulse", 500),
            max_pulse=pick("max_pulse", 2500),
        )

    @property
    def calibrated(self):
        return self.offset is not None

    def clamp_logical(self, logical):
        return _clamp(logical, self.min_angle, self.max_angle)

    def to_hardware_angle(self, logical):
        """hardware = center + (+/-logical) + offset, kept inside 0-180."""
        if not self.calibrated:
            raise UncalibratedError(f"{self.name} has no offset in servos.yaml")
        if self.invert:
            logical = -logical
        hw = self.center + logical + self.offset
        if not 0 <= hw <= 180:
            warnings.warn(f"{self.name}: hardware angle {hw:.1f} clamped to 0-180")
        return _clamp(hw, 0, 180)

    def to_pulse_us(self, hardware_angle):
        span = self.max_pulse - self.min_pulse
        return self.min_pulse + span * hardware_angle / 180

    def set(self, logical):
        """Move to a logical angle (clamped to limits). Returns the angle used."""
        logical = self.clamp_logical(logical)
        hw = self.to_hardware_angle(logical)
        self.pca.set_pulse_us(self.channel, self.to_pulse_us(hw))
        self.angle = logical
        return logical

    def release(self):
        self.pca.release_channel(self.channel)
        self.angle = None
