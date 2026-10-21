"""Thin VAP1 dispatch into the unchanged CR2 physical and reader engines."""
from scripts.pi0294.cr2 import export, runtime


def static(root):
    return runtime.static(root, profile='VAP1')


def capture(root):
    return runtime.capture(root, profile='VAP1')


def run(root):
    from pathlib import Path
    from scripts.pi0292.durable import read
    if read(Path(root)/'plans/WORK_PLAN.private.json').get('operation') == 'VAP1_SAVED_TRAIN_TEACHER_EVENT_AUDIT':
        from .teacher_audit import run as audit
        return audit(root)
    return export.run(root, profile='VAP1')
