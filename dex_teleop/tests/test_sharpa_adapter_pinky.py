"""--pinky-follows-ring: pinky flexion targets blended toward the ring finger's."""

import numpy as np
import pytest

from dex_teleop.hands import SHARPA_ACTION_JOINTS
from dex_teleop.omnigibson.sharpa_adapter import PINKY_RING_FLEXION, SharpaAdapterConfig, couple_pinky_to_ring


def test_pinky_flexion_blends_toward_ring_and_nothing_else_moves():
    hand = np.arange(len(SHARPA_ACTION_JOINTS), dtype=np.float64) / 10.0
    before = hand.copy()
    couple_pinky_to_ring(hand, 0.25)
    for pinky, ring in PINKY_RING_FLEXION:
        assert hand[pinky] == pytest.approx(0.75 * before[pinky] + 0.25 * before[ring])
    untouched = [i for i in range(len(hand)) if i not in {p for p, _ in PINKY_RING_FLEXION}]
    assert np.array_equal(hand[untouched], before[untouched])
    names = {SHARPA_ACTION_JOINTS[p] for p, _ in PINKY_RING_FLEXION}
    assert names == {"right_pinky_MCP_FE", "right_pinky_PIP", "right_pinky_DIP"}  # not CMC or abduction
    hand = before.copy()
    couple_pinky_to_ring(hand, 0.0)
    assert np.array_equal(hand, before)


def test_weight_is_validated_and_off_by_default():
    assert SharpaAdapterConfig().pinky_follows_ring == 0.0


def test_launcher_option_reaches_the_adapter():
    from dex_teleop.arat import AratTaskCatalog
    from dex_teleop.omnigibson.launch_config import parse_launch_arguments
    from dex_teleop.omnigibson.launcher import _parser

    args, _ = parse_launch_arguments(_parser(AratTaskCatalog()), ["--task", "arat_grasp_block_10cm", "--pinky-follows-ring", "0.25"], default_config=None)
    assert args.pinky_follows_ring == 0.25
