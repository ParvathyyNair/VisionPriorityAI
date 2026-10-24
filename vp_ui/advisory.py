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


def advise(t: dict) -> tuple[str, str]:
    """Return (short action tag, one-line message) for a scored track."""
    level = LEVEL.get(t.get("priority", "LOW"), 0)
    who = actor(t.get("class", "object"))
    view = t.get("view", "")
    closing = float(t.get("A", 0.0)) >= 0.45

    if level == 0:
        return "MONITOR", f"{who} in {t.get('view_display', 'view')} zone"

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
