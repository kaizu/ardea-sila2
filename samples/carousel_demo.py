"""Sample SiLA client: carry one plate around the labware stations and back.

Starting with the plate on Base2, the demo walks it through every station in turn and
returns it to where it started:

    Base2 -> Base3 -> Base4 -> Base5 -> Base2

Each leg is PickLabware -> MoveCarriage -> (SetOrientation) -> PutLabware. The arm only
turns when the next station faces the other way, and it does so while holding the plate,
which is supported. PutLabware leaves the arm at the retract pose (it does not return
home), which is exactly where the next leg's PickLabware starts, so the tour stays in the
retract family throughout and ends there. Stations, their positions and their facing all
come from the server's motion config, so nothing about the route is hard-coded here beyond
the order of station names -- a station commented out in motion.toml is reported as missing
rather than guessed at.

PHYSICAL MOTION. The demo asks for confirmation before it moves anything; pass --yes to
skip that, or --dry-run to print the plan and the current state without moving.

The carriage position and the arm's facing are normalised before the tour starts. Two
things are deliberately NOT fixed automatically:

* A hand that is not fully open. There is no Feature for moving the hand on its own
  (LabwareService only offers PickLabware/PutLabware), and driving the gripper from a
  sample would mean copying the server's PLC sequence out here. The demo stops and says
  what to do instead.
* An arm at none of the four known poses -- it could be anywhere, including inside a
  station, so no way out is invented.

!! The demo cannot tell whether a plate is actually present. On this machine the grasp
!! check does not work for long-edge grips (commanding 110 stops at 118 with the grip bit
!! clear whether or not a plate is there), so PickLabware reports success on an empty
!! gripper and the demo will happily "carry" nothing. Check visually.

Start the server first, e.g.:

    uv run python -m ardea_sila2 --config config.toml --motion-config motion.toml --insecure

then run this sample:

    uv run python samples/carousel_demo.py --host 127.0.0.1 --port 50053
"""

from __future__ import annotations

import argparse
import sys
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path

from sila2.client import SilaClient
from sila2.framework.errors.defined_execution_error import DefinedExecutionError

# The tour, by station name. Ends where it starts so the plate is left as it was found.
ROUTE = ["Base2", "Base3", "Base4", "Base5", "Base2"]

DM = 18                  # device type for DM/D devices
ADDR_HAND_POS = "6060"   # D6060: hand current position
HAND_OPEN_TOLERANCE = 3  # matches the server's _HAND_OPEN_TOL
COMMAND_TIMEOUT_S = 300.0
POLL_S = 0.3


class DemoError(Exception):
    """Something the demo will not drive through; carries a human-facing message."""


@dataclass(frozen=True)
class Station:
    name: str
    position_mm: int
    facing: str          # "forward" | "reverse"


def await_command(instance, label: str):
    """Wait for an observable command, polling only the server-side status.

    Deliberately does not touch the robot while it runs: a concurrent robot read
    (GetJointAngles and friends) during a task raises RobotAccessError.
    """
    started = time.monotonic()
    while not instance.done:
        if time.monotonic() - started > COMMAND_TIMEOUT_S:
            raise DemoError(f"{label} did not finish within {COMMAND_TIMEOUT_S:g}s")
        time.sleep(POLL_S)
    return instance.get_responses()


def find_motion_config(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise DemoError(f"--motion-config {explicit} does not exist")
        return path
    for candidate in (Path("motion.toml"), Path(__file__).resolve().parent.parent / "motion.toml"):
        if candidate.is_file():
            return candidate
    raise DemoError(
        "motion.toml not found. Run this from the project root, or pass "
        "--motion-config with the same file the server was started with."
    )


def load_route(client: SilaClient, motion_path: Path) -> list[Station]:
    """Resolve ROUTE into stations, using the config the server was started with.

    The server publishes station names but not their positions or facing, so those come
    from the motion config. Names are cross-checked against the server so a stale file
    cannot send the carriage somewhere the server does not consider a station.
    """
    with motion_path.open("rb") as fh:
        table = tomllib.load(fh).get("stations", {})
    published = set(client.CarriageService.StationNames.get())

    missing = sorted({name for name in ROUTE if name not in table or name not in published})
    if missing:
        raise DemoError(
            f"the route needs {missing}, which the server does not offer (it has "
            f"{sorted(published)}). A station commented out in {motion_path.name}, or one "
            "whose PacScripts are not on the controller, looks like this."
        )
    return [
        Station(name, int(table[name]["position_mm"]), str(table[name].get("direction", "forward")))
        for name in ROUTE
    ]


class Machine:
    """Thin read/act wrapper so the demo body reads as the sequence it performs."""

    def __init__(self, client: SilaClient) -> None:
        self.c = client

    # ---- reads ----
    def carriage_mm(self) -> int:
        return int(self.c.CarriageService.CarriagePosition.get())

    def hand_position(self) -> int:
        return self.c.DeviceService.ReadDevice(DM, ADDR_HAND_POS).Value

    def pose(self) -> str | None:
        """Name of the pose the arm is at, or None if it matches none of them."""
        p = self.c.RobotPoseService
        for name, at in (
            ("base", p.IsAtBasePose().IsAtBasePose),
            ("retract", p.IsAtRetractPose().IsAtRetractPose),
            ("inverse base", p.IsAtInverseBasePose().IsAtInverseBasePose),
            ("inverse retract", p.IsAtInverseRetractPose().IsAtInverseRetractPose),
        ):
            if at:
                return name
        return None

    def facing(self) -> str | None:
        """"forward"/"reverse" for the pose the arm is at, else None."""
        pose = self.pose()
        return None if pose is None else ("reverse" if pose.startswith("inverse") else "forward")

    def describe(self) -> str:
        return (f"carriage {self.carriage_mm()} mm, arm at {self.pose() or 'NO known pose'}, "
                f"hand {self.hand_position()}")

    # ---- actions ----
    def move_carriage(self, mm: int):
        return await_command(self.c.CarriageService.MoveCarriage(str(mm)), f"MoveCarriage({mm})")

    def set_orientation(self, direction: str):
        return await_command(self.c.RobotOrientationService.SetOrientation(direction),
                             f"SetOrientation({direction})")

    def pick(self):
        return await_command(self.c.LabwareService.PickLabware(), "PickLabware")

    def put(self):
        return await_command(self.c.LabwareService.PutLabware(), "PutLabware")


def plan_lines(route: list[Station], hand_open_at: int) -> list[str]:
    first = route[0]
    lines = [f"start: carriage to {first.position_mm} mm ({first.name}), "
             f"arm facing {first.facing}, hand open ({hand_open_at})"]
    for n, (here, there) in enumerate(zip(route, route[1:]), start=1):
        turn = "" if here.facing == there.facing else f", turn to {there.facing}"
        lines.append(f"leg {n}: pick at {here.name} -> move to {there.name} "
                     f"({there.position_mm} mm){turn} -> put at {there.name}")
    return lines


def normalise(m: Machine, first: Station, hand_open_at: int) -> None:
    """Bring the machine to "at the first station, facing it, hand open"."""
    pose = m.pose()
    if pose is None:
        raise DemoError(
            "the arm is at none of the base/retract/inverse-base/inverse-retract poses, so the "
            "demo will not move it -- it could be anywhere, including inside a station.\n"
            "  Recover by hand: open the hand fully, run this station's depart task, then the "
            "base task, e.g. RunTask(\"xxxDepartPick2\") then RunTask(\"BasePosition\")."
        )

    hand = m.hand_position()
    if abs(hand - hand_open_at) > HAND_OPEN_TOLERANCE:
        raise DemoError(
            f"the hand is at {hand}, not open ({hand_open_at}). The demo does not drive the "
            "gripper: there is no Feature for it, and duplicating the server's PLC sequence in "
            "a sample would be a second source of truth.\n"
            "  Open it first (a completed PutLabware leaves it open), then re-run."
        )

    if m.carriage_mm() != first.position_mm:
        print(f"  carriage {m.carriage_mm()} mm -> {first.position_mm} mm")
        print(f"    -> {m.move_carriage(first.position_mm)}")

    if m.facing() != first.facing:
        print(f"  turning to face {first.facing}")
        print(f"    -> {m.set_orientation(first.facing)}")

    # Either pose family will do: PickLabware starts from the base or the retract pose of
    # the station's facing, and moves itself from base to retract when it has to. Since
    # SetOrientation preserves the family, the facing set above is all this has to get right.


def run_leg(m: Machine, n: int, here: Station, there: Station) -> None:
    print(f"\n--- leg {n}: {here.name} -> {there.name} ---")
    print(f"  pick at {here.name}")
    print(f"    -> {m.pick()}")
    print(f"  move to {there.name} ({there.position_mm} mm)")
    print(f"    -> {m.move_carriage(there.position_mm)}")
    if here.facing != there.facing:
        print(f"  turn to {there.facing} (holding the plate)")
        print(f"    -> {m.set_orientation(there.facing)}")
    print(f"  put at {there.name}")
    print(f"    -> {m.put()}")
    print(f"  now: {m.describe()}")


def report_stop(m: Machine, client: SilaClient, error: Exception) -> None:
    print(f"\nSTOPPED: {type(error).__name__}: {error}", file=sys.stderr)
    try:
        print(f"state: {m.describe()}", file=sys.stderr)
        angles = client.RobotService.GetJointAngles().JointAngles
        print(f"CurJnt: {[round(v, 3) for v in angles]}", file=sys.stderr)
        if m.pose() is None:
            print("the arm is at no known pose: open the hand fully, run this station's depart "
                  "task, then BasePosition / InverseBasePosition to recover.", file=sys.stderr)
    except Exception as read_error:  # noqa: BLE001 - diagnostics must not mask the real cause
        print(f"(could not read the state either: {read_error})", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1", help="server host")
    parser.add_argument("--port", type=int, default=50053, help="server port")
    parser.add_argument("--motion-config", default=None,
                        help="motion.toml the server was started with (default: ./motion.toml)")
    parser.add_argument("--yes", action="store_true",
                        help="do not ask for confirmation before moving")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and the current state, then exit")
    args = parser.parse_args()

    client = SilaClient(args.host, args.port, insecure=True)
    m = Machine(client)

    try:
        motion_path = find_motion_config(args.motion_config)
        with motion_path.open("rb") as fh:
            hand_open_at = int(tomllib.load(fh).get("hand", {}).get("open_position", 140))
        route = load_route(client, motion_path)

        print(f"stations from {motion_path}")
        print("plan:")
        for line in plan_lines(route, hand_open_at):
            print(f"  {line}")
        print(f"\nnow: {m.describe()}")
        print(f"\nnote: a missing plate cannot be detected on this machine, so the demo reports "
              f"success either way. Check that a plate is on {route[0].name} first.")

        if args.dry_run:
            print("\ndry run; nothing moved.")
            return
        if not args.yes:
            if input("\nmove the robot and carriage as planned? [y/N] ").strip().lower() not in (
                    "y", "yes"):
                print("cancelled; nothing moved.")
                return

        print(f"\n--- start: to {route[0].name} ---")
        normalise(m, route[0], hand_open_at)
        print(f"  ready: {m.describe()}")

        for n, (here, there) in enumerate(zip(route, route[1:]), start=1):
            run_leg(m, n, here, there)

        print(f"\ndone: the plate is back on {route[-1].name}.")
        print(f"final: {m.describe()}")

    except (DemoError, DefinedExecutionError) as e:
        # Stop where we are and describe it: driving on from an unexpected state is how a
        # plate gets dropped or a station gets hit.
        report_stop(m, client, e)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
