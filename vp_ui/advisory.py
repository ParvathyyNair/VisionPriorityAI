"""Driver-facing advisory text derived from the risk layer's output.

The wording is rule based and uses the same factors that feed the score
(view, class, approach rate A, priority). It is guidance for the prototype,
not a calibrated collision warning.
"""
from __future__ import annotations

VULNERABLE = {"person": "Pedestrian", "bicycle": "Cyclist", "motorcycle": "Motorcyclist"}

# priority string in risk_layer -> numeric level used by the UI
LEVEL = {"LOW": 0, "MEDIUM": 1, "HIGH / CRITICAL": 2}


def actor(cls: str) -> str:
    cls = str(cls).lower()
    return VULNERABLE.get(cls, cls.capitalize() or "Object")


def advise_heavy(t: dict, level: int, who: str, view: str, closing: bool, vehicle: str) -> tuple[str, str] | None:
    """Advice specific to long vehicles (bus, coach, truck, van): blind spots, turning sweep,
    reversing and longer stopping distance. Returns None to fall back to the standard advice."""
    vulnerable = str(t.get("class", "")).lower() in VULNERABLE
    in_blind = t.get("zone") == "blind-spot"
    if view == "CAM_BACK_LEFT":
        if level == 2:
            return ("BLIND SPOT LEFT", f"{who} in your left blind spot. Do not turn left or move left.")
        return "WATCH LEFT FLANK", f"{who} alongside on the left. Check the left mirror before turning."
    if view == "CAM_BACK_RIGHT":
        if level == 2:
            return ("BLIND SPOT RIGHT", f"{who} in your right blind spot. Do not change lane right.")
        return "WATCH RIGHT FLANK", f"{who} alongside on the right. Check the right mirror."
    if view == "CAM_FRONT":
        if level == 2 and vulnerable and t.get("zone") in ("critical", "blind-spot"):
            return "DO NOT MOVE OFF", f"{who} in the blind zone below the windscreen. Wait until clear."
        if level == 2:
            return ("BRAKE EARLY", f"{who} closing ahead. A loaded {vehicle} needs longer to stop." if closing
                    else f"{who} in your forward path. Reduce speed early.")
        return None
    if view == "CAM_FRONT_LEFT" and level == 2:
        return ("NO LEFT TURN", f"{who} on the front-left. The trailer cuts in on a left turn. Wait."
                if vehicle == "truck" else f"{who} on the front-left. The long body cuts in on a left turn. Wait.")
    if view == "CAM_FRONT_RIGHT" and level == 2:
        return "CHECK RIGHT", f"{who} on the front-right. Allow for the wide front swing when turning right."
    if view == "CAM_BACK":
        if level == 2:
            return "DO NOT REVERSE", f"{who} close behind{' in the rear blind spot' if in_blind else ''}. Do not reverse."
        return "CHECK REAR", f"{who} behind. Use the rear camera before reversing."
    return None


def advise(t: dict, profile=None) -> tuple[str, str]:
    """Return (short action tag, one-line message) for a scored track.
    profile: optional vehicle_profiles.VehicleProfile; heavy vehicles get blind-spot advice."""
    level = LEVEL.get(t.get("priority", "LOW"), 0)
    who = actor(t.get("class", "object"))
    view = t.get("view", "")
    closing = float(t.get("A", 0.0)) >= 0.45

    if level == 0:
        return "MONITOR", f"{who} in {t.get('view_display', 'view')} zone"
    if profile is not None and getattr(profile, "heavy", False):
        vehicle = {"truck": "truck", "bus": "bus", "van": "van"}.get(profile.shape, "vehicle")
        out = advise_heavy(t, level, who, view, closing, vehicle)
        if out:
            return out

    if view == "CAM_FRONT":
        if level == 2:
            return ("BRAKE READY", f"{who} closing ahead. Prepare to brake." if closing
                    else f"{who} in your forward path. Reduce speed.")
        return "EASE OFF", f"{who} ahead. Keep a safe gap."
    if view == "CAM_BACK":
        if level == 2:
            return ("HOLD LANE", f"{who} closing from behind. Avoid hard braking." if closing
                    else f"{who} close behind. Hold your lane.")
        return "CHECK MIRROR", f"{who} behind. Keep your speed steady."
    if view == "CAM_FRONT_LEFT":
        if level == 2:
            return "CHECK LEFT", f"{who} closing on the left. Do not steer left."
        return "WATCH LEFT", f"{who} on the front-left. Stay alert."
    if view == "CAM_FRONT_RIGHT":
        if level == 2:
            return "CHECK RIGHT", f"{who} closing on the right. Do not steer right."
        return "WATCH RIGHT", f"{who} on the front-right. Stay alert."
    return "ALERT", f"{who} detected. Stay alert."
