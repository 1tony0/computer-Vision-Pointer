"""Arming: recognition only runs after a deliberate two-handed thumbs-up, so casual pointing does
nothing (a single thumbs-up is too easily confused with pointing seen from behind).

Two modes, each started by its own gesture held for `confirm_s`:
- two-handed thumbs-up -> POINT mode: name what the right index finger points at.
- peace sign (one hand, MediaPipe "Victory") -> LOOK mode: name what stays in the middle of the view.
Making the other mode's gesture switches modes. In look mode it stays armed until a thumbs-down.

In point mode: It then stays armed while it is being used:
every frame with a pointing hand pushes the deadline to `window_s` seconds later, so it disarms
once the hands have been down (no pointing) for `window_s` seconds. A thumbs-down held for
`confirm_s` disarms it straight away. Brief gaps in a gesture (tracking dropouts) shorter than
`grace_s` don't reset its confirmation timer.
"""


class _HeldGesture:
    """Fires once when a gesture has been visible for `confirm_s` (gaps up to `grace_s` allowed)."""

    def __init__(self, confirm_s: float, grace_s: float):
        self.confirm_s, self.grace_s = confirm_s, grace_s
        self._since: float | None = None
        self._last = float("-inf")
        self._consumed = False  # one trigger per continuous gesture

    def update(self, visible: bool, now_s: float) -> bool:
        if visible:
            if self._since is None or now_s - self._last > self.grace_s:
                self._since, self._consumed = now_s, False
            self._last = now_s
            if not self._consumed and now_s - self._since >= self.confirm_s:
                self._consumed = True
                return True
        elif now_s - self._last > self.grace_s:
            self._since = None
        return False


class ArmController:
    MODES = ("point", "look")

    def __init__(self, window_s: float = 5.0, confirm_s: float = 0.4, grace_s: float = 0.2,
                 always_armed: bool = False, mode: str = "point", lock_mode: bool = False):
        self.window_s = window_s
        self.confirm_s = confirm_s
        self.grace_s = grace_s
        self.always_armed = always_armed
        self.mode = mode
        self.lock_mode = lock_mode   # keep `mode`; mode gestures are ignored (clips recorded before the modes)
        self.armed_until = float("-inf")
        self._up = _HeldGesture(confirm_s, grace_s)
        self._peace = _HeldGesture(confirm_s, grace_s)
        self._down = _HeldGesture(confirm_s, grace_s)
        self.just_disarmed = False
        self.just_switched = False

    def keep_alive(self, now_s: float):
        """Called while the system is in use: an armed system stays armed during use."""
        if self.is_armed(now_s):
            self.armed_until = max(self.armed_until, now_s + self.window_s)

    def arm(self, now_s: float, mode: str | None = None):
        if mode is not None:
            self.mode = mode
        self.armed_until = now_s + self.window_s

    def disarm(self):
        self.armed_until = float("-inf")

    def update(self, thumbs_up: bool, now_s: float, thumbs_down: bool = False, peace: bool = False) -> bool:
        """Feed the gestures seen this frame; returns True on the frame the system becomes armed or
        changes mode. `just_disarmed` is True on the frame a thumbs-down switches it off and
        `just_switched` on the frame the mode changes."""
        was_armed = self.is_armed(now_s)
        self.just_disarmed = self.just_switched = False
        down = self._down.update(thumbs_down, now_s)
        up = self._up.update(thumbs_up, now_s)
        peace = self._peace.update(peace, now_s)
        if down and was_armed and not self.always_armed:
            self.disarm()
            self.just_disarmed = True
            return False
        for fired, mode in ((up, "point"), (peace, "look")):
            if fired and not (self.lock_mode and mode != self.mode):
                switched = mode != self.mode
                self.arm(now_s, mode)
                self.just_switched = was_armed and switched
                return not was_armed or switched
        return False

    def is_armed(self, now_s: float) -> bool:
        return self.always_armed or now_s < self.armed_until

    def remaining(self, now_s: float) -> float:
        return max(0.0, self.armed_until - now_s)
