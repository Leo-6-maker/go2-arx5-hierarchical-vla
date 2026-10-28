"""TRAIN-only physical-event exposure, not Teacher phases or sampling changes."""
import numpy as np


def _edges(times, flags, rising):
    before=np.r_[False,flags[:-1]]
    return times[(~before & flags) if rising else (before & ~flags)].tolist()


def _coverage(events, width):
    anchors=np.arange(601)/20
    values=np.asarray(events,dtype=float)
    # Closed endpoint convention is explicit; these are diagnostic exposures,
    # not claims that a discrete controller integrates its final H40 point.
    hit=np.any((values[None,:]>=anchors[:,None]-1e-8)&
        (values[None,:]<=anchors[:,None]+width+1e-8),axis=1) if len(values) else np.zeros(601,dtype=bool)
    return dict(windows=int(hit.sum()),denominator=601,fraction=float(hit.mean()))


def analyze(data, card):
    if card.get('split')!='TRAIN' or card.get('task') not in ('T1','T2'):
        raise PermissionError('PV1_EVENT_EXPOSURE_ONLY_FROZEN_TRAIN')
    t=np.asarray(data['time_s'],dtype=float);mt=np.asarray(data['contact_time_s'],dtype=float)
    micro=np.asarray(data['contact_audit'],dtype=float)
    obj=np.asarray(data['object_pose'],dtype=float);tcp=np.asarray(data['tcp_world'],dtype=float)
    if (t.shape!=(641,) or mt.shape!=(16000,) or micro.shape!=(16000,5)
            or obj.shape!=(641,4,4) or tcp.shape!=(641,4,4)
            or not all(np.isfinite(a).all() for a in (t,mt,micro,obj,tcp))
            or not np.all(np.abs(t-np.arange(641)/20)<=1e-8)
            or not np.all(np.abs(mt-np.arange(1,16001)*.002)<=1e-8)
            or not np.isin(micro[:,[0,1,4]],[0,1]).all() or micro[:,4].any()):
        raise PermissionError('PV1_EVENT_EXPOSURE_COMPLETE_ORIGINAL_SAFE_CLOCKED_PHYSICS_REQUIRED')
    contact=micro[:,0].astype(bool);grasp=micro[:,1].astype(bool)
    moved=np.maximum.accumulate(micro[:,2])>=.10
    xyz=obj[:,:3,3];goal=np.asarray(card['goal_xy'],dtype=float)
    import itertools
    corners=np.einsum('nij,kj->nki',obj[:,:3,:3],np.asarray(list(itertools.product((-.02,.02),repeat=3))))+xyz[:,None,:]
    zone=(np.abs(corners[:,:,:2]-goal)<=.07).all(axis=(1,2))
    clear=np.linalg.norm(tcp[:,:3,3]-xyz,axis=1)>=.08
    release=_edges(mt,grasp,False);separation=_edges(mt,contact,False)
    def after_transport(events):
        return [v for v in events if moved[min(int(round(v/.002))-1,15999)]]
    events=dict(contact_start=_edges(mt,contact,True),contact_end=separation,
        native_grasp_start=_edges(mt,grasp,True),native_grasp_end=release,
        native_grasp_end_after_10cm_transport=after_transport(release),
        separation_after_10cm_transport=after_transport(separation),
        zone_entry_20Hz=_edges(t,zone,True),TCP_clear_entry_20Hz=_edges(t,clear,True))
    # Frame exposure is separately reported, not conflated with transition windows.
    indices=np.maximum(np.arange(601)*25-1,0)
    masks=dict(robot_contact_500Hz_sample=contact[indices],native_grasp_500Hz_sample=grasp[indices],
        object_in_zone_20Hz=zone[:601],TCP_clear_20Hz=clear[:601],
        post_transport_separated_500Hz_sample=(moved&~contact&~grasp)[indices])
    return dict(schema='PI0294_PV1_TRAIN_PHYSICAL_EVENT_EXPOSURE_V1',id=card['id'],task=card['task'],
        scene_group=card['group_id'],legal_H40_anchors=601,
        events={key:dict(count=len(values),first_s=values[0] if values else None,
            last_s=values[-1] if values else None,
            prefix8_0_to_0p35s=_coverage(values,.35),H40_0_to_1p95s=_coverage(values,1.95))
            for key,values in events.items()},
        anchor_state_exposure={key:dict(anchors=int(values.sum()),denominator=601,fraction=float(values.mean()))
            for key,values in masks.items()},
        resolution=dict(contact_native_grasp_and_transport_s=.002,object_zone_and_TCP_clear_s=.05),
        zero_anchor_contact_uses_first_2ms_sample_not_unrecorded_t0=True,
        native_grasp_flag_is_not_invented_complete_release_or_task_success=True,
        event_categories_overlap_and_must_not_be_summed=True,Teacher_phase_labels_used=False,
        score_changed=False,sampler_changed=False,new_labels_for_training=False)


def summarize(rows):
    if not rows or len({r['id'] for r in rows})!=len(rows):raise PermissionError('UNIQUE_ACTUAL_TRAIN_PARENTS_REQUIRED')
    out={}
    for task in ('T1','T2'):
        own=[r for r in rows if r['task']==task]
        if not own:continue
        out[task]=dict(independent_trajectories=len(own),unique_scene_groups=len({r['scene_group'] for r in own}),
            events={key:dict(parents_with_event=sum(r['events'][key]['count']>0 for r in own),
                transition_count=sum(r['events'][key]['count'] for r in own),
                mean_parent_prefix8_window_fraction=float(np.mean([r['events'][key]['prefix8_0_to_0p35s']['fraction'] for r in own])),
                mean_parent_H40_window_fraction=float(np.mean([r['events'][key]['H40_0_to_1p95s']['fraction'] for r in own])))
                for key in own[0]['events']},
            mean_parent_anchor_state_fraction={key:float(np.mean([r['anchor_state_exposure'][key]['fraction'] for r in own]))
                for key in own[0]['anchor_state_exposure']})
    return dict(status='ACTUAL_TRAIN_EVENT_EXPOSURE_NOT_METHOD_SELECTION_OR_FAILURE_CAUSE',tasks=out,
        categories_overlap=True,sampling_or_loss_changed=False,
        P3_qualification_still_requires_matched_fit_failure_evidence=True)
