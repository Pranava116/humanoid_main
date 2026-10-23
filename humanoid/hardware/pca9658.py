"""Driver for the PCA9685 16-channel PWM board. Knows nothing about servos."""

import time

# Registers
MODE1 = 0x00
MODE2 = 0x01
PRESCALE = 0xFE
LED0_ON_L = 0x06
ALL_LED_OFF_H = 0xFD

# MODE1 bits
RESTART = 0x80
AUTO_INCREMENT = 0x20
SLEEP = 0x10

OSC_HZ = 25_000_000  # internal oscillator
FULL_OFF = 0x10      # bit 4 of LEDn_OFF_H forces the output off


class PCA9685:
    def __init__(self, bus, address, frequency=50, channels=16):
        """bus: an open smbus2.SMBus shared by all boards."""
        self.bus = bus
        self.address = address
        self.channels = channels
        self.frequency = frequency
        self.period_us = 1_000_000 / frequency

        self.bus.write_byte_data(address, MODE2, 0x04)          # totem-pole outputs
        self.bus.write_byte_data(address, MODE1, AUTO_INCREMENT)
        time.sleep(0.005)
        self.set_frequency(frequency)

    def set_frequency(self, frequency):
        prescale = round(OSC_HZ / (4096 * frequency)) - 1
        prescale = max(3, min(255, prescale))
        old = self.bus.read_byte_data(self.address, MODE1)
        # Prescale can only be written while the chip is asleep
        self.bus.write_byte_data(self.address, MODE1, (old & 0x7F) | SLEEP)
        self.bus.write_byte_data(self.address, PRESCALE, prescale)
        self.bus.write_byte_data(self.address, MODE1, old)
        time.sleep(0.005)
        self.bus.write_byte_data(self.address, MODE1, old | RESTART | AUTO_INCREMENT)
        self.frequency = frequency
        self.period_us = 1_000_000 / frequency

    def _check(self, channel):
        if not 0 <= channel < self.channels:
            raise ValueError(f"channel {channel} out of range (0-{self.channels - 1})")

    def set_pwm(self, channel, on, off):
        """Raw 12-bit PWM: output goes high at tick `on`, low at tick `off`."""
        self._check(channel)
        reg = LED0_ON_L + 4 * channel
        self.bus.write_i2c_block_data(
            self.address, reg, [on & 0xFF, on >> 8, off & 0xFF, off >> 8]
        )

    def set_pulse_us(self, channel, pulse_us):
        """Send a pulse of the given width in microseconds."""
        ticks = round(pulse_us / self.period_us * 4096)
        ticks = max(0, min(4095, ticks))
        self.set_pwm(channel, 0, ticks)

    def release_channel(self, channel):
        """Stop sending pulses on one channel (servo goes limp)."""
        self._check(channel)
        reg = LED0_ON_L + 4 * channel
        self.bus.write_i2c_block_data(self.address, reg, [0, 0, 0, FULL_OFF])

    def release_all(self):
        self.bus.write_byte_data(self.address, ALL_LED_OFF_H, FULL_OFF)
