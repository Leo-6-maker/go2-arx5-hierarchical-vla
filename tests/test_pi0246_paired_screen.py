from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import pi0246_paired_screen as screen


def test_p_arm_adds_only_the_authorized_checkpoint_profiles():
    assert screen.PROFILES["S0"] == (None, 0)
    assert screen.PROFILES["C2000"] == ("C", 2000)
    assert screen.PROFILES["T2000"] == ("T", 2000)
    assert screen.PROFILES["P1000"] == ("P", 1000)
    assert screen.PROFILES["P2000"] == ("P", 2000)

