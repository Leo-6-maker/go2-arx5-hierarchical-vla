"""Viewer source seam tests, never qualification of a new native binary."""
import pytest
from go2arx5_vla.world_model import native_ui_rate_v1 as ui

SOURCE='''// original startup and controls remain outside the viewer
void OriginalControl() { unchanged_control(); }
void Simulate::RenderLoop() {
  original_initialization();
'''+ui.LOOP+'''    {
      const MutexLock lock(this->mtx);
      original_Sync();
    }
    original_Render();
  }
}
// add state to history buffer
void OriginalHistory() { unchanged_history(); }
'''


def test_only_viewer_is_paced_and_sleep_precedes_simulation_mutex():
    changed=ui.patch(SOURCE)
    assert changed[len(ui.INCLUDES):].replace(ui.THROTTLED,ui.LOOP)==SOURCE
    assert changed.index('sleep_until')<changed.index('const MutexLock')
    assert 'milliseconds(200)'in changed


def test_changed_duplicate_or_already_patched_viewer_is_rejected():
    for source in(SOURCE.replace(ui.LOOP,'different loop'),SOURCE+ui.LOOP,ui.patch(SOURCE)):
        with pytest.raises(PermissionError):ui.patch(source)
