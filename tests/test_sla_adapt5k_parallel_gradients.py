"""Run with two logical CPU devices to exercise the real JAX collective."""
import numpy as np
import pytest


def test_pair_worker_binds_only_the_two_pbs_uuids_with_reference_first(tmp_path, monkeypatch):
    from datetime import datetime
    import json
    import os
    import time
    from zoneinfo import ZoneInfo

    from scripts.sla import benchmark_adapt5k_pair_vanda as pair

    home = tmp_path/'tmp'
    home.mkdir()
    monkeypatch.setenv('TMPDIR',str(home))
    monkeypatch.setenv('PBS_JOBID','19.site')
    start = datetime.now(ZoneInfo('Asia/Singapore')).strftime('%a %b %d %H:%M:%S %Y')
    job = dict(job_state='R',project='mine',Job_Owner='student@site',stime=start,
               Resource_List=dict(ngpus=2,walltime='03:00:00'))
    monkeypatch.setattr(pair,'read_pbs_with_auth_retry',lambda *args:job)
    monkeypatch.setattr(pair.adapt5k_pair_site,'visible_cuda_uuids',lambda:['second','primary'])
    host = os.uname().nodename
    config = dict(project='mine',user='student',uid=os.getuid(),parent_params=str(home),dataset=str(home),XLA_FLAGS='')
    reference = dict(device=dict(host=host,GPU_uuid='primary'))
    spec = dict(job_id='19.site',deadline_epoch=time.time()+8900,source_sha='a'*40)

    def validate_in_fresh_process(argv, *, input, env, **kwargs):
        received = json.loads(input)
        with open(received['pair_site_binding']['path']) as stream:
            body = json.load(stream)
        assert env['CUDA_VISIBLE_DEVICES']=='primary,second'
        assert body['GPU_uuids']==['primary','second'] and body['deadline_epoch']==received['deadline_epoch']
        return json.dumps(dict(topology='observed',devices=[dict(GPU_bytes=48*1024**3)]*2,GPU_name='NVIDIA A40'))

    monkeypatch.setattr(pair.subprocess,'check_output',validate_in_fresh_process)
    common,env,bound,observed = pair.site_inputs(spec,config,reference,tmp_path,dict(checkpoint=dict(sha256='b'*64)))
    assert env['CUDA_VISIBLE_DEVICES']=='primary,second'
    assert common['GPU_uuid']=='primary' and common['deadline_epoch']==bound['deadline_epoch']
    with open(observed['path']) as stream:
        assert json.load(stream)['allocated_GPU_uuids']==['primary','second']
    monkeypatch.setattr(pair.adapt5k_pair_site,'visible_cuda_uuids',lambda:['other','second'])
    with pytest.raises(PermissionError,match='MISSING_REFERENCE_CARD'):
        pair.site_inputs(spec,config,reference,tmp_path,dict(checkpoint=dict(sha256='b'*64)))


def test_two_device_local_accumulation_and_single_global_adam():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp

    from go2arx5_vla.sla_adapter import adapt5k_contract as contract
    from go2arx5_vla.sla_adapter.adapt5k_parallel_gradients import ParallelGradients
    from go2arx5_vla.sla_adapter.adapt5k_training import OPTIMIZER
    from go2arx5_vla.sla_adapter.scale20k_contract import make_adamw

    devices = jax.local_devices(backend='cpu')
    if len(devices)<2:
        pytest.skip('launch with two logical CPU devices')

    @jax.jit
    @jax.value_and_grad
    def loss(params, frozen, obs, action, noise, tau):
        x = obs['state']+frozen+action+noise+tau
        return jnp.mean((params['shared']*x-1.)**2+(params['private']*x+1.)**2)

    kernels = ParallelGradients(devices[:2],dict.fromkeys(contract.DOMAINS,loss),loss)
    parameters = {name:jnp.asarray(.25,dtype=jnp.float32) for name in ('shared',*contract.DOMAINS)}
    fm,reference = {},{}
    frozen = jnp.asarray(.125,dtype=jnp.float32)
    replica = kernels.replicate(frozen)

    def arguments(offset,size):
        values = jnp.arange(offset,offset+size,dtype=jnp.float32)/32
        return dict(state=values),values*.125,values*.25,values*.5

    for index,(name,size) in enumerate(contract.BATCH.items()):
        active = dict(shared=parameters['shared'],private=parameters[name])
        mirrored = kernels.replicate(active)
        total = kernels.zeros(mirrored)
        for offset in range(0,size//2,2):
            pos = index*64+offset
            prepared = kernels.shard(arguments(pos,2),arguments(pos+size//2,2))
            _,total = kernels.fm[name](total,2/size,mirrored,replica,*prepared)
        fm[name] = total
        _,reference[name] = loss(active,frozen,*arguments(index*64,size))

    active = dict(shared=parameters['shared'],private=parameters['target'])
    mirrored = kernels.replicate(active)
    aux = kernels.zeros(mirrored)
    for slot in range(2):
        prepared = kernels.shard(arguments(slot,1),arguments(slot+2,1))
        _,aux = kernels.auxiliary(aux,.25,mirrored,replica,*prepared)
    _,expected_aux = loss(active,frozen,*arguments(0,4))
    expected = contract.joint_gradient(reference,expected_aux,tree_map=jax.tree.map)
    summed = kernels.to_optimizer_device(kernels.reduce_joint(fm,aux))
    initialize,apply = make_adamw(**OPTIMIZER)
    observed_effect = apply(parameters,initialize(parameters),summed,3e-6)
    expected_effect = apply(parameters,initialize(parameters),expected,3e-6)
    for observed,truth in zip(jax.tree.leaves((summed,observed_effect)),
                              jax.tree.leaves((expected,expected_effect)),strict=True):
        np.testing.assert_allclose(observed,truth,rtol=3e-6,atol=1e-7)


def test_two_device_update_preserves_real_engine_visit_cursors_and_auxiliary_inputs(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace

    jax = pytest.importorskip('jax')
    import jax.numpy as jnp

    from go2arx5_vla.sla_adapter import adapt5k_contract as contract
    from go2arx5_vla.sla_adapter import scale20k_training as scale
    from go2arx5_vla.sla_adapter.adapt5k_parallel_gradients import ParallelAdaptUpdate
    from go2arx5_vla.sla_adapter.adapt5k_training import AdaptEngine, OPTIMIZER

    devices = jax.local_devices(backend='cpu')
    if len(devices)<2:
        pytest.skip('launch with two logical CPU devices')

    class Windows:
        def __init__(self, domain):
            self.domain = domain
            self.rows = [(dict(id='episode'),index) for index in range(3)]
            self.labels = dict(episode=dict(nominal_policy_H40=np.full((3,40,10),.5)))
            self.statistics = dict(actions=None)

        def __len__(self):
            return len(self.rows)

        def __getitem__(self,index):
            state = np.zeros(32,np.float32)
            state[0] = .25*(index+1)
            masks = dict.fromkeys(scale.CAMERAS,True)
            return dict(state=state,prompt=str(index),actions=np.zeros((scale.HORIZONS[self.domain],32),np.float32),
                image=dict.fromkeys(scale.CAMERAS,np.zeros((1,1,3),np.uint8)),image_mask=masks,image_padding_mask=masks)

        def reference(self,index):
            return dict(width_m=.040,source_native_s=1.,query_native_s=1.05)

    @jax.jit
    @jax.value_and_grad
    def fm(params,frozen,observation,actions,noise,tau):
        signal = observation['state'][:,0]+actions[:,0,0]+noise[:,0,0]*.05+tau*.01
        return jnp.mean((params['shared']*signal-.01)**2+(params['private']*signal+.01)**2)

    @jax.jit
    @jax.value_and_grad
    def aux(params,frozen,observation,noise,previous,target):
        signal = observation['state'][:,0]+noise[:,0,0]*.05+previous+target[:,0]
        return jnp.mean((params['shared']*signal-.01)**2+(params['private']*signal+.01)**2)

    def tokenize(prompt,state,*,use_discrete_state_format):
        return np.array([int(prompt),state is None]),np.ones(2,bool)

    monkeypatch.setattr(scale,'domain_loss',lambda graph:fm)
    monkeypatch.setattr(scale,'target_auxiliary',lambda graph,quantiles:aux)
    monkeypatch.setattr(scale.batch_v2,'device_batch',lambda row:(row['observation'],row['actions']))
    monkeypatch.setattr(ParallelAdaptUpdate,'_observation',staticmethod(lambda row:row))
    tokenizer = SimpleNamespace(delegate=SimpleNamespace(tokenize=tokenize))
    datasets = {d:Windows(d) for d in contract.DOMAINS}
    banks = {d:(np.arange(2000*n,dtype=np.int64)+i)%3 for i,(d,n) in enumerate(contract.BATCH.items())}
    params = {name:jnp.asarray(.125,dtype=jnp.float32) for name in ('shared',*contract.DOMAINS)}
    initialize,_ = scale.make_adamw(**OPTIMIZER)

    def engine(executor):
        instance = AdaptEngine(dict.fromkeys(contract.DOMAINS,'toy'),params,{},datasets,banks,tokenizer,
                    dict.fromkeys(contract.DOMAINS,2),restored_adam=initialize(params),restored_rng=jax.random.PRNGKey(29),
                    parent_step=500,phase_step=1000,stage_start=0,stage_stop=2000)
        instance.input_executor,instance.defer_fm_loss = executor,True
        return instance

    with ThreadPoolExecutor(max_workers=1) as executor:
        control,candidate = engine(executor),engine(executor)
        expected = control.update(lambda:None)
        actual = ParallelAdaptUpdate(candidate,devices[:2]).update(lambda:None)
    assert candidate.gradient_windows==control.gradient_windows==contract.BATCH
    assert candidate.completed_aux==control.completed_aux==4
    assert actual['lineage_cursors']==expected['lineage_cursors']
    assert actual['domains']['target']['auxiliaries_each_device']==[2,2]
    assert actual['domains']['target']['auxiliary_slots']==expected['domains']['target']['auxiliary_slots']
    for name,d in actual['domains'].items():
        assert d['window_count_each_device']==[contract.BATCH[name]//2]*2
        assert d['visit_slice_sha256']==expected['domains'][name]['visit_slice_sha256']
    for a,b in zip(jax.tree.leaves((candidate.parameters,candidate.states)),
                   jax.tree.leaves((control.parameters,control.states)),strict=True):
        np.testing.assert_allclose(a,b,rtol=3e-6,atol=1e-7)
