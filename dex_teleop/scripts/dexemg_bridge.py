#!/usr/bin/env python3
"""Repository-local bootstrap for ``dex_teleop.emg.dexemg_bridge`` (run it with the emg2pose environment).

    conda activate emg2pose
    python dex_teleop/scripts/dexemg_bridge.py --checkpoint ~/claude_work/emg_teleop/deploy_wrist_s0.pt --adapter hci0
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dex_teleop.emg.dexemg_bridge import main  # noqa: E402


if __name__ == "__main__":
    sys.exit(main())
