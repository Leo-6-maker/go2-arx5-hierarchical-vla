"""Reconcile saved per-query camera/debug sidecars omitted by the first inventory."""
from pathlib import Path
import hashlib
import numpy as np
from support import ROOT, read, sha, write


def main():
    source=ROOT/'design/VISITED_STATE_CARD.json';card=read(source);rows=[]
    assert len(card['rows'])==64
    for prior in card['rows']:
        row=dict(prior)
        if prior['status']=='MISSING_AFTER_STOP':rows.append(row);continue
        obs=Path(prior['request']['input']);assert sha(obs)==prior['observation_sha256']
        camera_path=obs.parent/'camera.json';debug_path=obs.parent/'debug.npz'
        if not camera_path.exists() or not debug_path.exists():
            row.update(camera_metadata_status='MISSING_SIDECAR_KEEP_PROPOSAL',teacher_label_status='UNAVAILABLE_CAMERA_COHERENCE')
            rows.append(row);continue
        camera=read(camera_path)
        with np.load(obs,allow_pickle=False) as z:
            assert float(z['timestamp'])==camera['timestamp']==prior['designated_timestamp_s']
            assert np.array_equal(camera['TCP_position_world'],z['position_world'])
            assert np.array_equal(camera['TCP_rotation_world'],z['rotation_world'])
        assert camera['physics_steps']==0 and camera['RNG_unchanged']
        for group in ('camera_view','camera_projection'):
            assert set(camera[group])=={'body','wrist'}
            for value in camera[group].values():assert np.asarray(value).shape==(4,4) and np.isfinite(value).all()
        with np.load(debug_path,allow_pickle=False) as z:
            assert set(z.files)=={'body_segmentation','wrist_segmentation','body_depth','wrist_depth'}
            fields={}
            for k in z.files:
                assert z[k].shape==(96,96)
                assert z[k].dtype==(np.uint32 if 'segmentation' in k else np.float32)
                # Depth backgrounds may be infinite; retain the actual bytes and denominator.
                fields[k]=dict(shape=list(z[k].shape),dtype=str(z[k].dtype),sha256=hashlib.sha256(z[k].tobytes()).hexdigest(),
                               nonfinite_count=int((~np.isfinite(z[k])).sum()))
        row.update(camera_per_frame_matrix='SAVED_VERIFIED_CAMERA_JSON',camera_metadata_status='PASS_SAME_QUERY_TCP_TIME_AND_DEBUG',
                   camera_file=str(camera_path),camera_sha256=sha(camera_path),debug_file=str(debug_path),debug_sha256=sha(debug_path),
                   debug_fields=fields,teacher_label_status='NOT_CALLED_A0_REQUIRED')
        rows.append(row)
    revised=dict(card,rows=rows,schema='PI0271_FIXED_64_SAVED_AGG_STATE_PROPOSALS_V2',
                 superseded_inventory_sha256=sha(source),camera_inventory_correction_source_sha256=sha(__file__))
    destination=ROOT/'design/VISITED_STATE_CARD_V2.json';write(destination,revised)
    write(ROOT/'evidence/VISITED_CAMERA_INVENTORY_ERRATUM.json',dict(status='PASS_SAVED_CAMERA_DEBUG_SIDECAR_RECONCILIATION',
          original_card=str(source),original_card_sha256=sha(source),corrected_card=str(destination),corrected_card_sha256=sha(destination),
          explanation='Original observation NPZ has no camera matrices; existing camera.json and debug.npz carry them. The first inventory omitted those files.',
          same_fixed_64_proposals=True,unchanged_input_noise_prediction_hashes=True,
          available_camera_rows=sum(r.get('camera_metadata_status')=='PASS_SAME_QUERY_TCP_TIME_AND_DEBUG' for r in rows),
          missing_or_stopped_rows=sum(r.get('camera_metadata_status')!='PASS_SAME_QUERY_TCP_TIME_AND_DEBUG' for r in rows),
          model_queries=0,teacher_calls=0,new_episodes=0,renders=0,source_sha256=sha(__file__),A0_still_required=True))
    print('PASS_64_FIXED_PROPOSALS_SAVED_CAMERA_DEBUG_ONLY_NO_RENDER_OR_QUERY',flush=True)


if __name__=='__main__':main()
