from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import pi0239_autonomous_engineering_to_science_gate as pi0239


class CallableFixture:
    def __call__(self) -> None:
        pass


def test_source_identity_accepts_callable_instances() -> None:
    identity = pi0239.source_identity(CallableFixture())
    assert identity["qualified"].endswith("CallableFixture")
    assert identity["source"].endswith("test_pi0239_runtime_identity.py")
    assert len(identity["sha256"]) == 64
