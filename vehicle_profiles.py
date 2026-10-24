"""Vehicle profiles for heavy-vehicle adoption.

Bigger vehicles have bigger blind spots, so the camera set and the risk weighting
change with the vehicle the system is fitted to:

* Car / SUV: four cameras (front, front-left, front-right, rear). This is the original
  VisionPriority configuration, and its scores are unchanged.
* Van, bus, coach and truck: six cameras. The two rear-side cameras (rear-left,
  rear-right) cover the long flanks of the vehicle, where cyclists, motorcyclists and
  cars sit unseen alongside. These are nuScenes' CAM_BACK_LEFT / CAM_BACK_RIGHT.
* Blind-spot gain: the zone-criticality factor Z of each camera is multiplied by a
  per-view gain (capped at 1.0), so an object at the same image position counts as
  more critical around a 16 m truck than around a 4.5 m car. The longer the vehicle,
  the larger the gain on the flank and rear cameras. The risk formula and its weights
  (0.25C + 0.20Z + 0.30A + 0.15D + 0.10P) and the priority thresholds are unchanged.
* India drives on the left, so the left side is the near side: a left-turning truck
  sweeps across the cycle and two-wheeler lane, and its left flank gets the
  highest gain.

The gains are design choices for this prototype, informed by the larger mirror blind
spots of buses and trucks. They have not been calibrated against measured blind-spot
maps of specific vehicle models.
"""
from __future__ import annotations

from dataclasses import dataclass, field

CAMS_4 = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK")
CAMS_6 = CAMS_4 + ("CAM_BACK_LEFT", "CAM_BACK_RIGHT")


@dataclass(frozen=True)
class VehicleProfile:
    key: str
    name: str
    wheels: str              # shown to the user, e.g. "6"
    length_m: float
    cameras: tuple
    gains: dict = field(default_factory=dict)   # view -> multiplier on the zone factor Z
    heavy: bool = False
    shape: str = "car"       # radar silhouette: car | van | bus | truck
    note: str = ""

    def gain(self, view: str) -> float:
        return float(self.gains.get(view, 1.0))

    def to_dict(self) -> dict:
        return {"key": self.key, "name": self.name, "wheels": self.wheels, "length_m": self.length_m,
                "cameras": list(self.cameras), "gains": {v: self.gain(v) for v in self.cameras},
                "heavy": self.heavy, "shape": self.shape, "note": self.note}


PROFILES: dict[str, VehicleProfile] = {p.key: p for p in [
    VehicleProfile("car", "Car / SUV", "4", 4.5, CAMS_4, {}, False, "car",
                   "Four cameras. Original configuration."),
    VehicleProfile("van", "Van / mini-truck", "4-6", 6.5, CAMS_6,
                   {"CAM_FRONT": 1.05, "CAM_FRONT_LEFT": 1.10, "CAM_FRONT_RIGHT": 1.05, "CAM_BACK": 1.15,
                    "CAM_BACK_LEFT": 1.20, "CAM_BACK_RIGHT": 1.15}, True, "van",
                   "Six cameras. No rear window, so the rear and both rear flanks are covered by camera."),
    VehicleProfile("bus", "City bus", "6", 12.0, CAMS_6,
                   {"CAM_FRONT": 1.15, "CAM_FRONT_LEFT": 1.20, "CAM_FRONT_RIGHT": 1.10, "CAM_BACK": 1.20,
                    "CAM_BACK_LEFT": 1.30, "CAM_BACK_RIGHT": 1.25}, True, "bus",
                   "Six cameras. Front blind zone below the windscreen and long left flank at bus stops."),
    VehicleProfile("coach", "Coach bus", "6-8", 13.5, CAMS_6,
                   {"CAM_FRONT": 1.15, "CAM_FRONT_LEFT": 1.20, "CAM_FRONT_RIGHT": 1.10, "CAM_BACK": 1.25,
                    "CAM_BACK_LEFT": 1.35, "CAM_BACK_RIGHT": 1.30}, True, "bus",
                   "Six cameras. Longest flanks of the passenger vehicles; highway lane changes."),
    VehicleProfile("truck", "Heavy truck / trailer", "10-18", 16.5, CAMS_6,
                   {"CAM_FRONT": 1.20, "CAM_FRONT_LEFT": 1.25, "CAM_FRONT_RIGHT": 1.15, "CAM_BACK": 1.30,
                    "CAM_BACK_LEFT": 1.40, "CAM_BACK_RIGHT": 1.35}, True, "truck",
                   "Six cameras. Largest blind spots: under the cab, along the trailer and directly behind."),
]}

# wheel count -> profile (for choosing the vehicle by its number of wheels)
WHEEL_OPTIONS = [4, 6, 8, 10, 12, 14, 16, 18]


def profile_for_wheels(wheels: int, passenger: bool = False) -> VehicleProfile:
    """4 wheels -> car; 6 -> city bus or mini-truck; 8 -> coach; 10+ -> heavy truck / trailer."""
    if wheels <= 4:
        return PROFILES["car"]
    if wheels <= 6:
        return PROFILES["bus"] if passenger else PROFILES["van"]
    if wheels <= 8:
        return PROFILES["coach"] if passenger else PROFILES["truck"]
    return PROFILES["truck"]


def get(key: str | None) -> VehicleProfile:
    return PROFILES.get(key or "car", PROFILES["car"])
