"""Thin VAP1 dispatch into the unchanged CR2 physical and reader engines."""
from scripts.pi0294.cr2 import export, runtime


def static(root):
    return runtime.static(root, profile='VAP1')


def capture(root):
    return runtime.capture(root, profile='VAP1')


def run(root):
    return export.run(root, profile='VAP1')
