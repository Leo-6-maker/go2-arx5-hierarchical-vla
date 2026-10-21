"""Use the proven READY-last delivery for the two explicitly authorized banks."""
def run(root):
    from pathlib import Path
    from scripts.pi0292.durable import read
    from scripts.pi0294.cr2.delivery import run as deliver
    work=read(Path(root)/'plans/WORK_PLAN.private.json')
    profile=work.get('bank_profile')
    if profile not in ('VAP1_D24','VAP1_D48'):
        raise PermissionError('VAP1_ONLY_D24_OR_D48_FIXED_PREFIX_DELIVERY')
    return deliver(root,profile=profile)
