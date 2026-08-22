"""
test_behaviour.py
-----------------
Tests for BehaviourAnalyzer.update() temporal persistence logic.

FIX (2026-08-22): update() returns a 2-tuple (newly_confirmed, is_active).
The previous tests asserted `result == False` against a tuple, which always
fails. Fixed to unpack the tuple correctly before asserting.
"""

import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from behaviour import BehaviourAnalyzer


def test_behaviour_persistence():
    """
    A behaviour that appears and persists for >= threshold seconds must
    fire exactly once (newly_confirmed=True), then enter cooldown.
    """
    analyzer = BehaviourAnalyzer()

    # Frame 1: behaviour just appeared — not yet confirmed
    newly_confirmed, is_active = analyzer.update("MOBILE_PHONE", True, 2.0)
    assert newly_confirmed is False, "should not confirm on first frame"
    assert is_active is False

    # Frame 2: ~1 second later — still below threshold
    time.sleep(1.05)
    newly_confirmed, is_active = analyzer.update("MOBILE_PHONE", True, 2.0)
    assert newly_confirmed is False, "should not confirm after only 1 second"

    # Frame 3: ~2.1 seconds total — crosses threshold, should confirm
    time.sleep(1.15)
    newly_confirmed, is_active = analyzer.update("MOBILE_PHONE", True, 2.0)
    assert newly_confirmed is True, "should confirm after persistence threshold"
    assert is_active is True

    # Frame 4: immediately after — now in cooldown, must NOT re-confirm
    newly_confirmed, is_active = analyzer.update("MOBILE_PHONE", True, 2.0)
    assert newly_confirmed is False, "should not re-confirm during cooldown"


def test_brief_behaviour_does_not_confirm():
    """
    A behaviour that disappears before the threshold must never confirm.
    """
    analyzer = BehaviourAnalyzer()

    newly_confirmed, _ = analyzer.update("SIDEWARD_GLANCE", True, 3.0)
    assert newly_confirmed is False

    time.sleep(0.5)
    newly_confirmed, _ = analyzer.update("SIDEWARD_GLANCE", True, 3.0)
    assert newly_confirmed is False

    # Now it disappears — reset
    newly_confirmed, _ = analyzer.update("SIDEWARD_GLANCE", False, 3.0)
    assert newly_confirmed is False

    # Appears again for 0.5s — must not confirm (timer reset from zero)
    time.sleep(0.5)
    newly_confirmed, _ = analyzer.update("SIDEWARD_GLANCE", True, 3.0)
    assert newly_confirmed is False
