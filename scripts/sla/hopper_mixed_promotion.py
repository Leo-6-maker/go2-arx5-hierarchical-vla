"""Publish CPU-certified mixed inputs into the normal course directory using atomic local hardlinks."""
from contextlib import closing
import os
from pathlib import Path

from scripts.pi0292.durable import OwnerLock, read
from scripts.sla.hopper_parent_custody import PLAN, digest, write_once
from scripts.sla.hopper_site import pinned
from scripts.sla.hopper_snapshot_relay import receive, verified
from hopper_reader_compat import READER, compare


def promote(view, original, reader, population, report):
    proof = pinned(report)
    dest = proof['destination']
    data = proof['actual']
    seal = dest['seal_sha256']
    n = dest['count']
    name = f'TRAIN-{n:03d}'
    stage = view.resolve().parent/'mixed-verified'/name
    # The repaired foundation view and its train-ready link share the original custody root.
    ready = (view/'train-ready').resolve(strict=True)
    if (stage.parent.parent != ready.parent or Path(dest['folder'])!=stage or proof['plan']!=PLAN
            or proof['status']!='MIXED_TRAIN_VANDA_READ_AND_HOPPER_STAGED' or proof['qualified_reader']!=READER
            or (proof['original_prefix_windows'],data['mixed'],data['seal'],data['manifest']) !=
               (1317,True,seal,dest['manifest_sha256']) or n!=data['episodes'] or data['windows']!=dest['windows']
            or any(proof[k] for k in ('optimizer_updates','model_generations','native_requests','course_switches'))):
        raise ValueError('Expected a real VANDA CPU readback and a matching isolated Hopper transfer')
    certificates = reader/'certificates'
    certificates.mkdir(exist_ok=True)
    with closing(OwnerLock(reader/'promotion.lock')):
        cert_file = certificates/(seal+'.json')
        final = ready/name
        if cert_file.is_file() and final.is_dir():
            cert=read(cert_file)
            if cert['vanda_readback']!=report or digest(final/'SEALED.private.json')!=seal:
                raise ValueError('Conflicting previous promotion')
            return cert
        sources=compare(original/'source',reader/'source')
        header,manifest,inventory=verified(stage,population,seal)
        original_prefix=read(ready/'TRAIN-016/SEALED.private.json')
        if (header['episode_sha256']!=data['member_sha256'] or header['episode_sha256'][:16]!=original_prefix['episode_sha256']
                or len(inventory)!=dest['members'] or sum(inventory.values())!=dest['bytes']):
            raise ValueError('Staged bytes or original membership differs')
        cases = read(stage/'ACTUAL.private.json')['cases']
        if (set(data['cases'])!={c['id'] for c in cases} or any(
                (data['cases'][c['id']]['native_result'],data['cases'][c['id']]['producer_spec'],data['cases'][c['id']]['windows'])
                !=(c['native_result_sha256'],c['producer_spec'],c['windows']) for c in cases)
                or data['supervised_atoms']!=40*manifest['windows'] or data['consumed_atoms']!=10*manifest['windows']):
            raise ValueError('The decoded windows belong to a different admission or producer')
        result = dict(plan=PLAN,status='HOPPER_MIXED_TRAIN_APPROVED',seal=seal,manifest=header['manifest_sha256'],
                      windows=manifest['windows'],episodes=n,sources=sources,vanda_readback=report)
        # The certificate must be durable before a boundary can see READY.
        write_once(cert_file,result)
        if not final.exists():
            incoming=ready/(name+'.incoming.'+seal)
            incoming.mkdir(mode=0o700,exist_ok=True)
            for file in inventory:
                src,dst=stage/file,incoming/file
                dst.parent.mkdir(mode=0o700,exist_ok=True)
                if dst.is_symlink() or (dst.exists() and not src.samefile(dst)):
                    raise ValueError('Conflicting in-flight TRAIN transfer member')
                if not dst.exists():
                    os.link(src,dst)
        receive(ready,population,n,seal)
        return result
