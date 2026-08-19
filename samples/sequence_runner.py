"""Run an explicit Pick / Put / MoveCarriage / SetOrientation / RunTask sequence.

★★ WARNING: MOVES THE ROBOT ARM, THE HAND AND THE TRAVEL CARRIAGE. ★★

Written during the 2026-08-14 real-machine session (HANDOVER.md §4.7) as the workhorse
for ad-hoc transport plans. Unlike ``carousel_demo.py`` (fixed Base2→3→4→5→2 route) this
takes an arbitrary step list, so any plan can be expressed by editing ``STEPS``.

Edit STEPS, then:

    uv run --directory ardea-sila2 python samples/sequence_runner.py

Design notes (each one exists because of a real failure that session):

- **No retries, no automatic recovery.** Stops at the first failure and prints the
  physical state, so the operator decides the recovery. Matches carousel_demo's policy.
- **Carriage settle wait.** Station lookup is an *exact* match on the carriage position
  (``MotionConfig.station_at``), and right after a move the position can still wobble.
  Waiting for N consecutive exact samples eliminated ``NoStationAtPosition`` entirely.
- **Pose reads are retried.** A b-CAP ``controller_connect`` issued right after RunTask
  can fail with ``E_CAO_COLLECTION_REGISTERED`` (0x80000205) even though the motion
  succeeded; a few seconds later it reads fine. Never treat it as a motion failure.
- **Per-step timing** is printed, which is how the session spotted that arm task
  durations changed between runs (controller-side trajectory edits, not this script).

Preconditions the server enforces (so plan the step order accordingly):
  Pick  needs the direction's *base or retract* pose + hand open. From the base pose it
        runs BasePosition->retract itself first, so the approach always starts at retract.
  Put   needs the direction's *retract* pose (forward: retract / reverse: inverse retract)
  Move  needs any of the four known poses
  SetOrientation preserves the pose family and only flips the facing

Both Pick and Put END at the retract pose (Put no longer returns home), so a pick/put
chain never passes through the base pose. Park at base explicitly with ("task",
"BasePosition") / ("task", "InverseBasePosition"), or with RobotOrientationService's
ReturnHome, which parks from any of the four known poses.

Since 2026-08-19 the server can also do a whole station-to-station route in one call --
``LabwareService.Transfer(source, destination)`` -- and drive the gripper and the machine
light directly (MoveHand / ActivateHand / ToggleLight). This runner deliberately keeps the
steps separate: its point is to time and observe each one. Use Transfer when you want the
route performed, this when you want to watch it happen.
"""
import sys, time
from sila2.client import SilaClient
from sila2.framework.errors.defined_execution_error import DefinedExecutionError

# The whole plan. ("pick"|"put", None) / ("move", "<mm>") / ("orient", "forward"|"reverse")
# / ("task", "<PacScript>"). Stations (motion.toml): Base2=2600 forward, Base3=2400,
# Base4=1600, Base5=2000 (all reverse).
STEPS = [
    # leg 1: Base3 -> Base4 (both reverse, no turn needed)
    ("pick",   None),        # Base3
    ("move",   "1600"),
    ("put",    None),        # Base4
    # leg 2: Base4 -> Base3
    ("pick",   None),        # Base4
    ("move",   "2400"),
    ("put",    None),        # Base3 -- back where it started
]

HOST, PORT = "127.0.0.1", 50053
SETTLE_TIMEOUT_S = 60.0
STABLE_SAMPLES = 3           # consecutive exact position samples before trusting a move
POST_STEP_SETTLE_S = 2.0     # let the machine settle before the next command's pose gate
RACE_RETRY_S = 3.0           # wait before re-reading after a 0x80000205 connect race

c = SilaClient(HOST, PORT, insecure=True)
DM = 18
rw = lambda no: c.DeviceService.ReadDevice(DM, str(no)).Value & 0xFFFF


def poses(retries=3):
    """(base, retract, inverse_base, inverse_retract), or None if unreadable."""
    for attempt in range(retries):
        try:
            return (c.RobotPoseService.IsAtBasePose().IsAtBasePose,
                    c.RobotPoseService.IsAtRetractPose().IsAtRetractPose,
                    c.RobotPoseService.IsAtInverseBasePose().IsAtInverseBasePose,
                    c.RobotPoseService.IsAtInverseRetractPose().IsAtInverseRetractPose)
        except DefinedExecutionError as e:
            print("       (pose read failed: %s; retry %d/%d)" % (e.identifier, attempt + 1, retries))
            time.sleep(RACE_RETRY_S)
    return None


def state(tag):
    p = poses()
    print("  [%s] carriage=%s mm  hand=%d  D6002=0x%04X  poses(b/r/ib/ir)=%s"
          % (tag, c.CarriageService.CarriagePosition.get(), rw(6060), rw(6002),
             "/".join("T" if x else "F" for x in p) if p else "unreadable"))


def await_observable(inst, attr="Phase"):
    """Poll an observable command, printing each new intermediate value."""
    seen = None
    while not inst.done:
        try:
            inter = inst.get_intermediate_response()
            v = getattr(inter, attr, None) if inter is not None else None
            if v is not None and v != seen:
                print("      %s: %s" % (attr.lower(), v))
                seen = v
        except Exception:
            pass
        time.sleep(0.3)
    return inst.get_responses()


def settle(target_mm):
    """Wait until the carriage reads exactly target_mm on STABLE_SAMPLES consecutive polls."""
    t0, stable = time.time(), 0
    while time.time() - t0 <= SETTLE_TIMEOUT_S:
        if int(c.CarriageService.CarriagePosition.get()) == target_mm:
            stable += 1
            if stable >= STABLE_SAMPLES:
                return True
        else:
            stable = 0
        time.sleep(0.5)
    return False


print("=== initial state ===")
state("start")
times = []

try:
    for i, (action, arg) in enumerate(STEPS, 1):
        label = "%s%s" % (action.upper(), "" if arg is None else
                          "(%s)" % (arg if action in ("task", "orient") else "%s mm" % arg))
        print("\n=== step %d/%d: %s ===" % (i, len(STEPS), label))
        t0 = time.time()
        if action == "pick":
            print("  ->", await_observable(c.LabwareService.PickLabware()))
        elif action == "put":
            print("  ->", await_observable(c.LabwareService.PutLabware()))
        elif action == "orient":
            print("  ->", await_observable(c.RobotOrientationService.SetOrientation(arg)))
        elif action == "task":
            c.TaskService.RunTask(arg)
            time.sleep(RACE_RETRY_S)     # a b-CAP connect right after RunTask can fail
            print("  -> RunTask(%s) returned" % arg)
        elif action == "move":
            print("  ->", await_observable(c.CarriageService.MoveCarriage(arg),
                                           attr="CurrentPosition"))
            if not settle(int(arg)):
                sys.exit("ABORT: carriage did not settle on %s mm; stopping." % arg)
            print("      settled at %s mm" % arg)
        else:
            sys.exit("ABORT: unknown action %r" % action)
        dt = time.time() - t0
        times.append((label, dt))
        print("  step time: %.1f s" % dt)
        time.sleep(POST_STEP_SETTLE_S)
        state("after step %d" % i)

    print("\n=== SEQUENCE COMPLETE (%d steps) ===" % len(STEPS))
    print("timing per step:")
    for label, dt in times:
        print("  %-26s %6.1f s" % (label, dt))
    print("  %-26s %6.1f s" % ("TOTAL", sum(dt for _, dt in times)))

except DefinedExecutionError as e:
    print("\n!!! FAILED at step %d [%s]: %s" % (len(times) + 1, e.identifier, e.message))
    time.sleep(RACE_RETRY_S)
    state("failure")
    print("\nRecovery hints: GraspFailed/HandError leave the arm inside the station with the"
          "\nhand closed -> open the hand, RunTask(<station>'s depart script), then the"
          "\ndirection's home task (BasePosition / InverseBasePosition). A RobotAccessError"
          "\nreading -2147483131 (0x80000205) is the connect race, NOT a motion failure --"
          "\nre-read the pose after a few seconds before deciding anything.")
    sys.exit(1)
