"""
    Author: J. Scott Thompson (j.scott.thompson.12@gmail.com)
    Date: 06/22/2024
    Description: Six degree-of-freedom motion of an F-16 derived from Stevens and Lewis, ported from AeroBenchVVPython

"""

from enum import Enum, IntEnum, auto
from math import sqrt, cos, sin, acos

class Stevens():
    class Status(Enum):
        READY = auto()
        FLYING = auto()
        FINISHED = auto()

    class Maneuver(Enum):
        STRAIGHT = auto()
        TURNING = auto()

    class StateComponents(IntEnum):
        VT =        0   # air speed (ft/sec)
        alpha =     1   # angle of attack (rad)
        beta =      2   # angle of sideslip (rad)
        phi =       3   # roll angle (rad)
        theta =     4   # pitch angle (rad)
        psi =       5   # yaw angle (rad)
        P =         6   # roll rate (rad/sec)
        Q =         7   # pitch rate (rad/sec)
        R =         8   # yaw rate (rad/sec)
        pn =        9   # northward horizontal displacement (feet)
        pe =        10  # eastward horizontal displacement (feet)
        h =         11  # altitude (feet)
        pow =       12  # engine thrust dynamics lag state

    class ManeuverComponents(IntEnum):
        throttle =  0   # Throttle command, 0.0 < u(1) < 1.0
        elevator =  1   # Elevator command in degrees
        aileron =   2   # Aileron command in degrees
        rudder =    3   # Rudder command in degrees

    def __init__(self):
        # Placeholder: DomainBehavior init

        self.state = {'status': self.Status.READY,
                      'manuever': self.Maneuver.STRAIGHT,
                      'sigma': 0.1,
                      'maneuver_commands': [0.0] * 4,
                      'physical_state': [0.0] * 18
                      }

    def setInitialPhysicalState(self, msg):
        # Initial state message format:
        # t0, x0, y0, z0, vx0, vy0, vz0, phi0, theta0, psi0
        pass

    def processManueverCommand(selfself, msg):
        # Manuver command message format:
        # t, throttle, elevator, aileron, rudder
        pass

    def intTransition(self):
        # Propagate state
        pass

    def extTransition(self):
        # Process external event (start, manuver, stop)
        pass

    def __str__(self):
        msg = str(self.state['physical_state'])
        return msg

    def outputFnc(self):
        pass

    def timeAdvance(self):
        return self.state['sigma']
