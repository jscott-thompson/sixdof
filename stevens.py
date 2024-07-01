"""
    Author: J. Scott Thompson (j.scott.thompson.12@gmail.com)
    Date: 06/22/2024
    Description: Six degree-of-freedom motion of an F-16 derived from Stevens and Lewis, ported from AeroBenchVVPython

"""

from DomainInterface.DomainBehavior import DomainBehavior
from Domain.Basic.Object import Message
from enum import Enum, IntEnum, auto
import numpy as np
from math import sqrt, cos, sin, acos


class stevens(DomainBehavior):
    class Status(Enum):
        READY = auto()
        FLYING = auto()
        FINISHED = auto()

    class Maneuver(Enum):
        STRAIGHT = auto()
        TURNING = auto()

    class StateComponents(IntEnum):
        t = auto()  # time (sec)
        VT = auto()  # air speed (ft/sec)
        alpha = auto()  # angle of attack (rad)
        beta = auto()  # angle of sideslip (rad)
        phi = auto()  # roll angle (rad)
        theta = auto()  # pitch angle (rad)
        psi = auto()  # yaw angle (rad)
        P = auto()  # roll rate (rad/sec)
        Q = auto()  # pitch rate (rad/sec)
        R = auto()  # yaw rate (rad/sec)
        pn = auto()  # northward horizontal displacement (feet)
        pe = auto()  # eastward horizontal displacement (feet)
        h = auto()  # altitude (feet)
        pow = auto()  # engine thrust dynamics lag state

    class ManeuverComponents(IntEnum):
        throttle = auto()  # Throttle command, 0.0 < u(1) < 1.0
        elevator = auto()  # Elevator command in degrees
        aileron = auto()  # Aileron command in degrees
        rudder = auto()  # Rudder command in degrees

    def __init__(self):
        DomainBehavior.__init__(self)

        self.state = {'status': self.Status.READY,
                      'maneuver': self.Maneuver.STRAIGHT,
                      'adjust_cy': True,
                      'sigma': 0.1,
                      'maneuver_commands': np.zeros(len(self.ManeuverComponents)+1),
                      'physical_state': np.zeros(len(self.StateComponents)+1)
                      }
        self.msg = Message(None, None)

    def setInitialPhysicalState(self, msg):
        # Initial state message format:
        # t0, vt, alpha, beta, phi, theta, psi, alt, power
        if not isinstance(msg.value, str):
            temp = ''
            for i in range(8):
                temp += str(msg.value[i]) + ', '
            temp += str(msg.value[8])
            msg.value = temp
        [t, vt, alpha, beta, phi, theta, psi, alt, power] = [float(x) for x in msg.value.split(',')]

        comp = self.StateComponents
        self.state['physical_state'][comp.t] = t
        self.state['physical_state'][comp.VT] = vt
        self.state['physical_state'][comp.alpha] = alpha
        self.state['physical_state'][comp.beta] = beta
        self.state['physical_state'][comp.phi] = phi
        self.state['physical_state'][comp.theta] = theta
        self.state['physical_state'][comp.psi] = psi
        self.state['physical_state'][comp.h] = alt
        self.state['physical_state'][comp.pow] = power

        return

    def processManueverCommand(self, msg):
        # Maneuver command message format:
        # t, throttle, elevator, aileron, rudder
        if not isinstance(msg.value, str):
            temp = str(msg.value[0]) + ', ' + str(msg.value[1]) + ', ' + str(msg.value[2]) + ', ' + str(msg.value[3]) + ', ' + str(1.0)
            msg = temp
        [t, throttle, elevator, aileron, rudder] = [float(x) for x in msg.split(',')]
        comp = self.ManeuverComponents
        self.Maneuver[comp.throttle] = throttle
        self.Maneuver[comp.elevator] = elevator
        self.Maneuver[comp.aileron] = aileron
        self.Maneuver[comp.rudder] = rudder

    def intTransition(self):
        # Propagate state
        from math import sin, cos, pi
        from enum import IntEnum

        class F16Model:
            class StateComponents(IntEnum):
                VT = 0
                ALPHA = 1
                BETA = 2
                PHI = 3
                THETA = 4
                PSI = 5
                P = 6
                Q = 7
                R = 8
                PN = 9
                PE = 10
                H = 11
                POWER = 12

            def __init__(self, model='stevens', adjust_cy=True):
                self.state = np.zeros(13)
                self.control_input = np.zeros(4)
                self.model = model
                self.adjust_cy = adjust_cy

            def intTransition(self):
                def adc(vt, alt):
                    '''converts velocity (vt) and altitude (alt) to mach number (amach) and dynamic pressure (qbar)

                    See pages 63-65 of Stevens & Lewis, "Aircraft Control and Simulation", 2nd edition
                    '''

                    # vt = freestream air speed

                    ro = 2.377e-3
                    tfac = 1 - .703e-5 * alt

                    if alt >= 35000:  # in stratosphere
                        t = 390
                    else:
                        t = 519 * tfac  # 3 rankine per atmosphere (3 rankine per 1000 ft)

                    # rho = freestream mass density
                    rho = ro * tfac ** 4.14

                    # a = speed of sound at the ambient conditions
                    # speed of sound in a fluid is the sqrt of the quotient of the modulus of elasticity over the mass density
                    a = sqrt(1.4 * 1716.3 * t)

                    # amach = mach number
                    amach = vt / a

                    # qbar = dynamic pressure
                    qbar = .5 * rho * vt * vt

                    return amach, qbar

                def tgear(thtl):
                    'tgear function'

                    if thtl <= .77:
                        tg = 64.94 * thtl
                    else:
                        tg = 217.38 * thtl - 117.38

                    return tg

                from aerobench.lowlevel.rtau import rtau

                def pdot(p3, p1):
                    'pdot function'

                    if p1 >= 50:
                        if p3 >= 50:
                            t = 5
                            p2 = p1
                        else:
                            p2 = 60
                            t = rtau(p2 - p3)
                    else:
                        if p3 >= 50:
                            t = 5
                            p2 = 40
                        else:
                            p2 = p1
                            t = rtau(p2 - p3)

                    pd = t * (p2 - p3)

                    return pd

                from aerobench.util import fix

                def thrust(power, alt, rmach):
                    'thrust lookup-table version'

                    a = np.array([[1060, 670, 880, 1140, 1500, 1860], \
                                  [635, 425, 690, 1010, 1330, 1700], \
                                  [60, 25, 345, 755, 1130, 1525], \
                                  [-1020, -170, -300, 350, 910, 1360], \
                                  [-2700, -1900, -1300, -247, 600, 1100], \
                                  [-3600, -1400, -595, -342, -200, 700]], dtype=float).T

                    b = np.array([[12680, 9150, 6200, 3950, 2450, 1400], \
                                  [12680, 9150, 6313, 4040, 2470, 1400], \
                                  [12610, 9312, 6610, 4290, 2600, 1560], \
                                  [12640, 9839, 7090, 4660, 2840, 1660], \
                                  [12390, 10176, 7750, 5320, 3250, 1930], \
                                  [11680, 9848, 8050, 6100, 3800, 2310]], dtype=float).T

                    c = np.array([[20000, 15000, 10800, 7000, 4000, 2500], \
                                  [21420, 15700, 11225, 7323, 4435, 2600], \
                                  [22700, 16860, 12250, 8154, 5000, 2835], \
                                  [24240, 18910, 13760, 9285, 5700, 3215], \
                                  [26070, 21075, 15975, 11115, 6860, 3950], \
                                  [28886, 23319, 18300, 13484, 8642, 5057]], dtype=float).T

                    if alt < 0:
                        alt = 0.01  # uh, why not 0?

                    h = .0001 * alt

                    i = fix(h)

                    if i >= 5:
                        i = 4

                    dh = h - i
                    rm = 5 * rmach
                    m = fix(rm)

                    if m >= 5:
                        m = 4
                    elif m <= 0:
                        m = 0

                    dm = rm - m
                    cdh = 1 - dh

                    # do not increment these, since python is 0-indexed while matlab is 1-indexed
                    # i = i + 1
                    # m = m + 1

                    s = b[i, m] * cdh + b[i + 1, m] * dh
                    t = b[i, m + 1] * cdh + b[i + 1, m + 1] * dh
                    tmil = s + (t - s) * dm

                    if power < 50:
                        s = a[i, m] * cdh + a[i + 1, m] * dh
                        t = a[i, m + 1] * cdh + a[i + 1, m + 1] * dh
                        tidl = s + (t - s) * dm
                        thrst = tidl + (tmil - tidl) * power * .02
                    else:
                        s = c[i, m] * cdh + c[i + 1, m] * dh
                        t = c[i, m + 1] * cdh + c[i + 1, m + 1] * dh
                        tmax = s + (t - s) * dm
                        thrst = tmil + (tmax - tmil) * (power - 50) * .02

                    return thrst

                from aerobench.util import fix, sign

                def cx(alpha, el):
                    'cx definition'

                    a = np.array([[-.099, -.081, -.081, -.063, -.025, .044, .097, .113, .145, .167, .174, .166], \
                                  [-.048, -.038, -.040, -.021, .016, .083, .127, .137, .162, .177, .179, .167], \
                                  [-.022, -.020, -.021, -.004, .032, .094, .128, .130, .154, .161, .155, .138], \
                                  [-.040, -.038, -.039, -.025, .006, .062, .087, .085, .100, .110, .104, .091], \
                                  [-.083, -.073, -.076, -.072, -.046, .012, .024, .025, .043, .053, .047, .040]],
                                 dtype=float).T

                    s = .2 * alpha
                    k = fix(s)
                    if k <= -2:
                        k = -1

                    if k >= 9:
                        k = 8

                    da = s - k
                    l = k + fix(1.1 * sign(da))
                    s = el / 12
                    m = fix(s)
                    if m <= -2:
                        m = -1

                    if m >= 2:
                        m = 1

                    de = s - m
                    n = m + fix(1.1 * sign(de))
                    k = k + 3
                    l = l + 3
                    m = m + 3
                    n = n + 3
                    t = a[k - 1, m - 1]
                    u = a[k - 1, n - 1]
                    v = t + abs(da) * (a[l - 1, m - 1] - t)
                    w = u + abs(da) * (a[l - 1, n - 1] - u)
                    cxx = v + (w - v) * abs(de)

                    return cxx

                def cy(beta, ail, rdr):
                    'cy function'

                    return -.02 * beta + .021 * (ail / 20) + .086 * (rdr / 30)

                def cz(alpha, beta, el):
                    'cz function'

                    a = np.array(
                        [.770, .241, -.100, -.415, -.731, -1.053, -1.355, -1.646, -1.917, -2.120, -2.248, -2.229], \
                        dtype=float).T

                    s = .2 * alpha
                    k = fix(s)

                    if k <= -2:
                        k = -1

                    if k >= 9:
                        k = 8

                    da = s - k
                    l = k + fix(1.1 * sign(da))
                    l = l + 3
                    k = k + 3
                    s = a[k - 1] + abs(da) * (a[l - 1] - a[k - 1])

                    return s * (1 - (beta / 57.3) ** 2) - .19 * (el / 25)

                def cl(alpha, beta):
                    'cl function'

                    a = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
                                  [-.001, -.004, -.008, -.012, -.016, -.022, -.022, -.021, -.015, -.008, -.013, -.015], \
                                  [-.003, -.009, -.017, -.024, -.030, -.041, -.045, -.040, -.016, -.002, -.010, -.019], \
                                  [-.001, -.010, -.020, -.030, -.039, -.054, -.057, -.054, -.023, -.006, -.014, -.027], \
                                  [.000, -.010, -.022, -.034, -.047, -.060, -.069, -.067, -.033, -.036, -.035, -.035], \
                                  [.007, -.010, -.023, -.034, -.049, -.063, -.081, -.079, -.060, -.058, -.062, -.059], \
                                  [.009, -.011, -.023, -.037, -.050, -.068, -.089, -.088, -.091, -.076, -.077, -.076]],
                                 dtype=float).T

                    s = .2 * alpha
                    k = fix(s)

                    if k <= -2:
                        k = -1

                    if k >= 9:
                        k = 8

                    da = s - k
                    l = k + fix(1.1 * sign(da))
                    s = .2 * abs(beta)
                    m = fix(s)
                    if m == 0:
                        m = 1

                    if m >= 6:
                        m = 5

                    db = s - m
                    n = m + fix(1.1 * sign(db))
                    l = l + 3
                    k = k + 3
                    m = m + 1
                    n = n + 1
                    t = a[k - 1, m - 1]
                    u = a[k - 1, n - 1]
                    v = t + abs(da) * (a[l - 1, m - 1] - t)
                    w = u + abs(da) * (a[l - 1, n - 1] - u)
                    dum = v + (w - v) * abs(db)

                    return dum * sign(beta)

                def cm(alpha, el):
                    'cm function'

                    a = np.array([[.205, .168, .186, .196, .213, .251, .245, .238, .252, .231, .198, .192], \
                                  [.081, .077, .107, .110, .110, .141, .127, .119, .133, .108, .081, .093], \
                                  [-.046, -.020, -.009, -.005, -.006, .010, .006, -.001, .014, .000, -.013, .032], \
                                  [-.174, -.145, -.121, -.127, -.129, -.102, -.097, -.113, -.087, -.084, -.069, -.006], \
                                  [-.259, -.202, -.184, -.193, -.199, -.150, -.160, -.167, -.104, -.076, -.041, -.005]],
                                 dtype=float).T

                    s = .2 * alpha
                    k = fix(s)

                    if k <= -2:
                        k = -1

                    if k >= 9:
                        k = 8

                    da = s - k
                    l = k + fix(1.1 * sign(da))
                    s = el / 12
                    m = fix(s)

                    if m <= -2:
                        m = -1

                    if m >= 2:
                        m = 1

                    de = s - m
                    n = m + fix(1.1 * sign(de))
                    k = k + 3
                    l = l + 3
                    m = m + 3
                    n = n + 3
                    t = a[k - 1, m - 1]
                    u = a[k - 1, n - 1]
                    v = t + abs(da) * (a[l - 1, m - 1] - t)
                    w = u + abs(da) * (a[l - 1, n - 1] - u)

                    return v + (w - v) * abs(de)

                def dlda(alpha, beta):
                    'dlda function'

                    a = np.array([[-.041, -.052, -.053, -.056, -.050, -.056, -.082, -.059, -.042, -.038, -.027, -.017], \
                                  [-.041, -.053, -.053, -.053, -.050, -.051, -.066, -.043, -.038, -.027, -.023, -.016], \
                                  [-.042, -.053, -.052, -.051, -.049, -.049, -.043, -.035, -.026, -.016, -.018, -.014], \
                                  [-.040, -.052, -.051, -.052, -.048, -.048, -.042, -.037, -.031, -.026, -.017, -.012], \
                                  [-.043, -.049, -.048, -.049, -.043, -.042, -.042, -.036, -.025, -.021, -.016, -.011], \
                                  [-.044, -.048, -.048, -.047, -.042, -.041, -.020, -.028, -.013, -.014, -.011, -.010], \
                                  [-.043, -.049, -.047, -.045, -.042, -.037, -.003, -.013, -.010, -.003, -.007, -.008]],
                                 dtype=float).T

                    s = .2 * alpha
                    k = fix(s)
                    if k <= -2:
                        k = -1

                    if k >= 9:
                        k = 8

                    da = s - k
                    l = k + fix(1.1 * sign(da))
                    s = .1 * beta
                    m = fix(s)
                    if m <= -3:
                        m = -2

                    if m >= 3:
                        m = 2

                    db = s - m
                    n = m + fix(1.1 * sign(db))
                    l = l + 3
                    k = k + 3
                    m = m + 4
                    n = n + 4
                    t = a[k - 1, m - 1]
                    u = a[k - 1, n - 1]
                    v = t + abs(da) * (a[l - 1, m - 1] - t)
                    w = u + abs(da) * (a[l - 1, n - 1] - u)

                    return v + (w - v) * abs(db)

                def dldr(alpha, beta):
                    'dldr function'

                    a = np.array([[.005, .017, .014, .010, -.005, .009, .019, .005, -.000, -.005, -.011, .008], \
                                  [.007, .016, .014, .014, .013, .009, .012, .005, .000, .004, .009, .007], \
                                  [.013, .013, .011, .012, .011, .009, .008, .005, -.002, .005, .003, .005], \
                                  [.018, .015, .015, .014, .014, .014, .014, .015, .013, .011, .006, .001], \
                                  [.015, .014, .013, .013, .012, .011, .011, .010, .008, .008, .007, .003], \
                                  [.021, .011, .010, .011, .010, .009, .008, .010, .006, .005, .000, .001], \
                                  [.023, .010, .011, .011, .011, .010, .008, .010, .006, .014, .020, .000]],
                                 dtype=float).T

                    s = .2 * alpha
                    k = fix(s)
                    if k <= -2:
                        k = -1

                    if k >= 9:
                        k = 8

                    da = s - k
                    l = k + fix(1.1 * sign(da))
                    s = .1 * beta
                    m = fix(s)

                    if m <= -3:
                        m = -2

                    if m >= 3:
                        m = 2

                    db = s - m
                    n = m + fix(1.1 * sign(db))
                    l = l + 3
                    k = k + 3
                    m = m + 4
                    n = n + 4
                    t = a[k - 1, m - 1]
                    u = a[k - 1, n - 1]

                    v = t + abs(da) * (a[l - 1, m - 1] - t)
                    w = u + abs(da) * (a[l - 1, n - 1] - u)

                    return v + (w - v) * abs(db)

                def cn(alpha, beta):
                    'cn function'

                    a = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
                                  [.018, .019, .018, .019, .019, .018, .013, .007, .004, -.014, -.017, -.033], \
                                  [.038, .042, .042, .042, .043, .039, .030, .017, .004, -.035, -.047, -.057], \
                                  [.056, .057, .059, .058, .058, .053, .032, .012, .002, -.046, -.071, -.073], \
                                  [.064, .077, .076, .074, .073, .057, .029, .007, .012, -.034, -.065, -.041], \
                                  [.074, .086, .093, .089, .080, .062, .049, .022, .028, -.012, -.002, -.013], \
                                  [.079, .090, .106, .106, .096, .080, .068, .030, .064, .015, .011, -.001]],
                                 dtype=float).T

                    s = .2 * alpha
                    k = fix(s)

                    if k <= -2:
                        k = -1

                    if k >= 9:
                        k = 8

                    da = s - k
                    l = k + fix(1.1 * sign(da))
                    s = .2 * abs(beta)
                    m = fix(s)

                    if m == 0:
                        m = 1

                    if m >= 6:
                        m = 5

                    db = s - m
                    n = m + fix(1.1 * sign(db))
                    l = l + 3
                    k = k + 3
                    m = m + 1
                    n = n + 1
                    t = a[k - 1, m - 1]
                    u = a[k - 1, n - 1]

                    v = t + abs(da) * (a[l - 1, m - 1] - t)
                    w = u + abs(da) * (a[l - 1, n - 1] - u)
                    dum = v + (w - v) * abs(db)

                    return dum * sign(beta)

                def dnda(alpha, beta):
                    'dnda function'

                    a = np.array([[.001, -.027, -.017, -.013, -.012, -.016, .001, .017, .011, .017, .008, .016], \
                                  [.002, -.014, -.016, -.016, -.014, -.019, -.021, .002, .012, .016, .015, .011], \
                                  [-.006, -.008, -.006, -.006, -.005, -.008, -.005, .007, .004, .007, .006, .006], \
                                  [-.011, -.011, -.010, -.009, -.008, -.006, .000, .004, .007, .010, .004, .010], \
                                  [-.015, -.015, -.014, -.012, -.011, -.008, -.002, .002, .006, .012, .011, .011], \
                                  [-.024, -.010, -.004, -.002, -.001, .003, .014, .006, -.001, .004, .004, .006], \
                                  [-.022, .002, -.003, -.005, -.003, -.001, -.009, -.009, -.001, .003, -.002, .001]],
                                 dtype=float).T

                    s = .2 * alpha
                    k = fix(s)

                    if k <= -2:
                        k = -1

                    if k >= 9:
                        k = 8

                    da = s - k
                    l = k + fix(1.1 * sign(da))
                    s = .1 * beta
                    m = fix(s)
                    if m <= -3:
                        m = -2

                    if m >= 3:
                        m = 2

                    db = s - m
                    n = m + fix(1.1 * sign(db))
                    l = l + 3
                    k = k + 3
                    m = m + 4
                    n = n + 4
                    t = a[k - 1, m - 1]
                    u = a[k - 1, n - 1]
                    v = t + abs(da) * (a[l - 1, m - 1] - t)
                    w = u + abs(da) * (a[l - 1, n - 1] - u)

                    return v + (w - v) * abs(db)

                def dndr(alpha, beta):
                    'dndr function'

                    a = np.array([[-.018, -.052, -.052, -.052, -.054, -.049, -.059, -.051, -.030, -.037, -.026, -.013], \
                                  [-.028, -.051, -.043, -.046, -.045, -.049, -.057, -.052, -.030, -.033, -.030, -.008], \
                                  [-.037, -.041, -.038, -.040, -.040, -.038, -.037, -.030, -.027, -.024, -.019, -.013], \
                                  [-.048, -.045, -.045, -.045, -.044, -.045, -.047, -.048, -.049, -.045, -.033, -.016], \
                                  [-.043, -.044, -.041, -.041, -.040, -.038, -.034, -.035, -.035, -.029, -.022, -.009], \
                                  [-.052, -.034, -.036, -.036, -.035, -.028, -.024, -.023, -.020, -.016, -.010, -.014], \
                                  [-.062, -.034, -.027, -.028, -.027, -.027, -.023, -.023, -.019, -.009, -.025, -.010]],
                                 dtype=float).T

                    s = .2 * alpha
                    k = fix(s)
                    if k <= -2:
                        k = -1

                    if k >= 9:
                        k = 8

                    da = s - k
                    l = k + fix(1.1 * sign(da))
                    s = .1 * beta
                    m = fix(s)
                    if m <= -3:
                        m = -2

                    if m >= 3:
                        m = 2

                    db = s - m
                    n = m + fix(1.1 * sign(db))
                    l = l + 3
                    k = k + 3
                    m = m + 4
                    n = n + 4
                    t = a[k - 1, m - 1]
                    u = a[k - 1, n - 1]
                    v = t + abs(da) * (a[l - 1, m - 1] - t)
                    w = u + abs(da) * (a[l - 1, n - 1] - u)
                    return v + (w - v) * abs(db)

                def dampp(alpha):
                    'dampp functon'

                    a = np.array([[-.267, -.110, .308, 1.34, 2.08, 2.91, 2.76, 2.05, 1.50, 1.49, 1.83, 1.21], \
                                  [.882, .852, .876, .958, .962, .974, .819, .483, .590, 1.21, -.493, -1.04], \
                                  [-.108, -.108, -.188, .110, .258, .226, .344, .362, .611, .529, .298, -2.27], \
                                  [-8.80, -25.8, -28.9, -31.4, -31.2, -30.7, -27.7, -28.2, -29.0, -29.8, -38.3, -35.3], \
                                  [-.126, -.026, .063, .113, .208, .230, .319, .437, .680, .100, .447, -.330], \
                                  [-.360, -.359, -.443, -.420, -.383, -.375, -.329, -.294, -.230, -.210, -.120, -.100], \
                                  [-7.21, -.540, -5.23, -5.26, -6.11, -6.64, -5.69, -6.00, -6.20, -6.40, -6.60, -6.00], \
                                  [-.380, -.363, -.378, -.386, -.370, -.453, -.550, -.582, -.595, -.637, -1.02, -.840], \
                                  [.061, .052, .052, -.012, -.013, -.024, .050, .150, .130, .158, .240, .150]],
                                 dtype=float).T

                    s = .2 * alpha
                    k = fix(s)

                    if k <= -2:
                        k = -1

                    if k >= 9:
                        k = 8

                    da = s - k
                    l = k + fix(1.1 * sign(da))
                    k = k + 3
                    l = l + 3

                    d = np.zeros((9,))

                    for i in range(9):
                        d[i] = a[k - 1, i] + abs(da) * (a[l - 1, i] - a[k - 1, i])

                    return d


                comp = self.StateComponents
                x = self.state
                u = self.control_input
                dt = self.state['sigma']

                xcg = 0.35
                thtlc, el, ail, rdr = u

                # Constants
                s = 300
                b = 30
                cbar = 11.32
                rm = 1.57e-3
                xcgr = .35
                he = 160.0
                c1 = -.770
                c2 = .02755
                c3 = 1.055e-4
                c4 = 1.642e-6
                c5 = .9604
                c6 = 1.759e-2
                c7 = 1.792e-5
                c8 = -.7336
                c9 = 1.587e-5
                rtod = 57.29578
                g = 32.17

                vt = x[comp.VT]
                alpha = x[comp.ALPHA] * rtod
                beta = x[comp.BETA] * rtod
                phi = x[comp.PHI]
                theta = x[comp.THETA]
                psi = x[comp.PSI]
                p = x[comp.P]
                q = x[comp.Q]
                r = x[comp.R]
                alt = x[comp.H]
                power = x[comp.POWER]

                # air data computer and engine model
                amach, qbar = adc(vt, alt)
                cpow = tgear(thtlc)

                power_dot = pdot(power, cpow)
                x[comp.POWER] += power_dot * dt

                t = thrust(power, alt, amach)
                dail = ail / 20
                drdr = rdr / 30

                # component build up
                cxt = cx(alpha, el)
                cyt = cy(beta, ail, rdr)
                czt = cz(alpha, beta, el)

                clt = cl(alpha, beta) + dlda(alpha, beta) * dail + dldr(alpha, beta) * drdr
                cmt = cm(alpha, el)
                cnt = cn(alpha, beta) + dnda(alpha, beta) * dail + dndr(alpha, beta) * drdr

                # add damping derivatives
                tvt = .5 / vt
                b2v = b * tvt
                cq = cbar * q * tvt

                # get ready for state equations
                d = dampp(alpha)
                cxt = cxt + cq * d[0]
                cyt = cyt + b2v * (d[1] * r + d[2] * p)
                czt = czt + cq * d[3]
                clt = clt + b2v * (d[4] * r + d[5] * p)
                cmt = cmt + cq * d[6] + czt * (xcgr - xcg)
                cnt = cnt + b2v * (d[7] * r + d[8] * p) - cyt * (xcgr - xcg) * cbar / b
                cbta = cos(x[comp.BETA])
                u = vt * cos(x[comp.ALPHA]) * cbta
                v = vt * sin(x[comp.BETA])
                w = vt * sin(x[comp.ALPHA]) * cbta
                sth = sin(theta)
                cth = cos(theta)
                sph = sin(phi)
                cph = cos(phi)
                spsi = sin(psi)
                cpsi = cos(psi)
                qs = qbar * s
                qsb = qs * b
                rmqs = rm * qs
                gcth = g * cth
                qsph = q * sph
                ay = rmqs * cyt
                az = rmqs * czt

                # force equations
                udot = r * v - q * w - g * sth + rm * (qs * cxt + t)
                vdot = p * w - r * u + gcth * sph + ay
                wdot = q * u - p * v + gcth * cph + az
                dum = (u * u + w * w)

                x[comp.VT] += ((u * udot + v * vdot + w * wdot) / vt) * dt
                x[comp.ALPHA] += ((u * wdot - w * udot) / dum) * dt
                x[comp.BETA] += ((vt * vdot - v * x[comp.VT]) * cbta / dum) * dt

                # kinematics
                x[comp.PHI] += (p + (sth / cth) * (qsph + r * cph)) * dt
                x[comp.THETA] += (q * cph - r * sph) * dt
                x[comp.PSI] += ((qsph + r * cph) / cth) * dt

                # moments
                x[comp.P] += ((c2 * p + c1 * r + c4 * he) * q + qsb * (c3 * clt + c4 * cnt)) * dt
                x[comp.Q] += ((c5 * p - c7 * he) * r + c6 * (r * r - p * p) + qs * cbar * c7 * cmt) * dt
                x[comp.R] += ((c8 * p - c2 * r + c9 * he) * q + qsb * (c4 * clt + c9 * cnt)) * dt

                # navigation
                t1 = sph * cpsi
                t2 = cph * sth
                t3 = sph * spsi
                s1 = cth * cpsi
                s2 = cth * spsi
                s3 = t1 * sth - cph * spsi
                s4 = t3 * sth + cph * cpsi
                s5 = sph * cth
                s6 = t2 * cpsi + t3
                s7 = t2 * spsi - t1
                s8 = cph * cth
                x[comp.PN] += (u * s1 + v * s3 + w * s6) * dt  # north speed
                x[comp.PE] += (u * s2 + v * s4 + w * s7) * dt  # east speed
                x[comp.H] += (u * sth - v * s5 - w * s8) * dt  # vertical speed

                # outputs
                xa = 15.0  # sets distance normal accel is in front of the c.g. (xa = 15.0 at pilot)
                az = az - xa * x[comp.Q]  # moves normal accel in front of c.g.

                if self.adjust_cy:
                    ay = ay + xa * x[comp.R]  # moves side accel in front of c.g.

                # For extraction of Nz
                Nz = (-az / g) - 1  # zeroed at 1 g, positive g = pulling up
                Ny = ay / g

                self.state['physical_state'] = x

    def extTransition(self):
        # Process external event (start, maneuver, stop)
        # Pull data off input port
        msg = self.peek(self.IPorts[0])

        if self.state['status'] == self.Status.READY:
            # Set initial state from first input
            self.setInitialPhysicalState(msg)
            self.state['status'] = self.Status.FLYING
            self.state['sigma'] = 0.1  # constant time step, hard-coded for now

        elif self.state['status'] == self.Status.FLYING:
            # Check if this is a shutdown message
            if msg in ['', '\n', '\r\n']:
                self.state['status'] = self.Status.FINISHED
            else:
                # Maneuver command message format:
                # t, throttle, elevator, aileron, rudder
                if not isinstance(msg.value, str):
                    temp = str(msg.value[0]) + ', ' + str(msg.value[1]) + ', ' + str(msg.value[2]) + ', ' + str(msg.value[3]) + ', ' + str(1.0)
                    msg = temp
                [t, throttle, elevator, aileron, rudder] = [float(x) for x in msg.split(',')]
                comp = self.ManeuverComponents
                self.state['maneuver_commands'][comp.throttle] = throttle
                self.state['maneuver_commands'][comp.elevator] = elevator
                self.state['maneuver_commands'][comp.aileron] = aileron
                self.state['maneuver_commands'][comp.rudder] = rudder
        return
    def __str__(self):
        msg = str(self.state['physical_state'])
        return msg

    def outputFnc(self):
        try:
            self.msg.time = self.state['physical_state'][self.StateComponents.t]
            self.msg.value = self.__str__()
            self.poke(self.OPorts[0], self.msg)
        except:
            pass

    def timeAdvance(self):
        return self.state['sigma']
