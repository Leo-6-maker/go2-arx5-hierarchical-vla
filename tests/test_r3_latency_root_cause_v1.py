import copy
import pytest
from scripts.wm02.r3_latency_root_cause_v1 import reconstruct,original_write_proxy
from go2arx5_vla.sla_adapter.compilation_cache_binding_v1 import bind_declared_cache


def fixture():
    rows=[];evidence=[]
    for i in range(32):
        capture=10+i*.5;available=capture+.01;created=available+.02;dispatch=created+.005
        response=dispatch+.1;terminal=response+.03
        rows.append(dict(index=i,family='F1'if i<16 else'F2',phase_score_only='CLOSE',noise=0,cold=i==0,
            scheduled_monotonic=capture,release_lateness_s=0.,file_delivery_s=.01,input_available_to_request_s=.02,
            dispatch_monotonic=dispatch,IPC_to_response_s=.1,input_available_to_terminal_s=terminal-available,
            model_branch_s=[.07],seal_enqueue_s=.001,ACK=True,emitted_mock_atoms=0,TTL_rejected=False,reservation_wait_capacity_s=0.))
        evidence.append(dict(index=i,documents={'REQUEST.private.json':dict(created_monotonic=created),'RESPONSE.json':dict(worker_wall_s=.08)},
            reservation=dict(created_epoch=1000+i*.5,completed_epoch=1000+i*.5+.02),reservation_stat=dict(ctime_ns=int((1000+i*.5+.03)*1e9)),source_file_metadata=[]))
    return rows,evidence


def test_exact_nested_and_cross_request_phase_conservation_without_invented_timers():
    rows,evidence=fixture();out=reconstruct(rows,evidence)
    for r in out:
        d=r['durations']
        assert sum(d[k]for k in ('available_to_request_s','request_to_dispatch_s','provider_call_s','worker_outside_provider_s',
            'transport_publication_poll_residual_s','response_to_install_return_or_rejection_s'))==pytest.approx(.155)
        assert r['events']['model_completed']is None
        assert r['internal_unmeasured']['mock_consume_and_finish_s']is None
        assert r['mock_write_proxy']['first_atom_publication_monotonic_s']is None
        assert r['mock_write_proxy']['first_atom_status']=='NOT_EMITTED'
    for r in out[:-1]:
        c=r['cycle_to_next']
        assert c['dispatch_interval_s']==pytest.approx(.5)
        assert c['dispatch_to_terminal_s']+c['seal_enqueue_s']+c['unresolved_postresponse_excluding_seal_s']+c['next_capture_to_dispatch_s']==pytest.approx(.5)
    assert out[-1]['cycle_to_next']is None


@pytest.mark.parametrize('fault',['clock','worker_duration','row_count','row_order'])
def test_inconsistent_receipts_fail_instead_of_becoming_a_false_breakdown(fault):
    rows,evidence=fixture()
    if fault=='clock':rows[2]['file_delivery_s']=.05
    elif fault=='worker_duration':evidence[2]['documents']['RESPONSE.json']['worker_wall_s']=.2
    elif fault=='row_count':rows.pop()
    else:rows[2],rows[3]=rows[3],rows[2]
    with pytest.raises(ValueError):reconstruct(rows,evidence)


def test_archive_copy_time_or_unverified_original_never_becomes_consumer_timing():
    item=dict(source_file_metadata=[dict(file='receiver/live.json',original_sha_matches=False,original_stat=dict(mtime_ns=10**12),archive_stat=dict(mtime_ns=20**12)),
        dict(file='receiver/install-0001.json',original_sha_matches=True,original_stat=dict(mtime_ns=1))])
    out=original_write_proxy(item,10)
    assert out['consume_only_conditional_enclosure_s']is None
    assert out['first_atom_publication_monotonic_s']is None


def test_original_write_enclosure_keeps_explicit_clock_assumptions_and_missing_exact_time():
    item=dict(source_file_metadata=[dict(file=name,original_sha_matches=True,original_stat=dict(mtime_ns=t))for name,t in
        (('receiver/install-0001.json',1000000000),('receiver/emitted-0001-00.json',1001000000),('receiver/live.json',1004000000))])
    result=original_write_proxy(item,10)
    assert result['consume_only_conditional_enclosure_s']==pytest.approx(.004)
    assert result['enclosure_excludes_finish_final_fsync']
    assert result['first_atom_publication_monotonic_s']is None and result['assumptions']


def test_declared_cache_binding_is_read_back_and_changes_no_numerical_setting(tmp_path):
    class Config:
        jax_compilation_cache_dir=None
        jax_enable_compilation_cache=False
        numerical=dict(precision=None,steps=10)
        def update(self,key,value):
            assert key=='jax_compilation_cache_dir'
            setattr(self,key,value)
    config=Config();before=copy.deepcopy(config.numerical)
    result=bind_declared_cache(dict(compilation_cache_dir=str(tmp_path)),config)
    assert result['effective_directory']==str(tmp_path.resolve())
    assert result['persistent_cache_enabled']is False  # Explicit disabling is respected.
    assert config.numerical==before
    assert not list(tmp_path.iterdir())


def test_cache_binding_rejects_missing_directory_and_ignored_config(tmp_path):
    with pytest.raises(FileNotFoundError):bind_declared_cache(dict(compilation_cache_dir=str(tmp_path/'missing')),object())
    class Ignored:
        jax_compilation_cache_dir=None
        def update(self,*args):pass
    with pytest.raises(RuntimeError):bind_declared_cache(dict(compilation_cache_dir=str(tmp_path)),Ignored())
