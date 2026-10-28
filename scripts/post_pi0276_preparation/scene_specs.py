"""Factorized StackCube proposal cards, never simulator instances."""
import argparse,hashlib,json,math,random
from pathlib import Path
from cpu_dataset import RELEASE,sha

UPSTREAM='62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3'
SPLITS={'DEV':range(50000,50008),'TRAIN':range(51000,51096),'VAL':range(52000,52016),'TEST':range(53000,53032)}
DOMAINS=('D0_LAYOUT','D1_APPEARANCE','D2_EXTERNAL_CAMERA')
EYE=[.3,0,.6];TARGET=[-.1,0,.1]
def namespace(group,kind):return int.from_bytes(hashlib.sha256(f'POST_PI0276/{group}/{kind}'.encode()).digest()[:8],'big')
def cross(a,b):return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
def normalized(v):
    norm=math.sqrt(sum(x*x for x in v))
    if norm<1e-12:raise ValueError('Degenerate static camera axis')
    return [x/norm for x in v]
def multiply(a,b):return [[sum(a[i][k]*b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
def camera(offset=(0,0,0),yaw=0,pitch=0,fov=90):
    eye=[a+b for a,b in zip(EYE,offset)];forward=normalized([t-e for t,e in zip(TARGET,eye)])
    left=normalized(cross([0,0,1],forward));up=cross(forward,left)
    rotation=[[forward[i],left[i],up[i]] for i in range(3)]
    y,p=map(math.radians,(yaw,pitch));rz=[[math.cos(y),-math.sin(y),0],[math.sin(y),math.cos(y),0],[0,0,1]]
    ry=[[math.cos(p),0,math.sin(p)],[0,1,0],[-math.sin(p),0,math.cos(p)]]
    rotation=multiply(rz,multiply(rotation,ry))
    return dict(eye_offset_m=list(offset),yaw_world_z_deg=yaw,pitch_local_y_deg=pitch,fov_deg=fov,
        world_from_sapien_camera=[rotation[i]+[eye[i]] for i in range(3)]+[[0,0,0,1]],
        convention='SAPIEN local +X forward, +Y left, +Z up; yaw worldZ then pitch localY',
        dimensions=[128,128],near=.01,far=100,binding_stage='CONSTRUCTION_OR_RECONFIGURATION',actual_resolved=None)

def cards():
    rows=[]
    for split,seeds in SPLITS.items():
        for seed in seeds:
            common=dict(parent_group=f'{split}-{seed}',split=split,native_reset_seed=seed,
                task='StackCube-v1',robot='panda_wristcam',source_commit=UPSTREAM,authority_release=RELEASE,
                rng={kind:namespace(seed,kind) for kind in ('placement','appearance','camera','source_planning','policy_noise')},
                original_geometry_physics_controls_and_wrist=True,sim_freq=100,control_freq=20,
                initial_state_hash=None,actual_resolved=None,status='METADATA_ONLY_RESERVED' if split=='TEST' else 'STATIC_PROPOSAL',
                instruction='Stack the red cube on the green cube and release it.',native_visibility='NOT_RUN')
            appearance=random.Random(common['rng']['appearance']);view=random.Random(common['rng']['camera'])
            for domain in DOMAINS:
                value=dict(common,domain=domain,scene_id=f'{split}-{seed}-{domain}',layout=dict(recipe='PINNED_NATIVE_RESET_PLACEMENT_AND_YAW',actual_state=None),appearance=None,camera=None)
                if split!='TEST':
                    value['camera']=camera()
                    if domain=='D1_APPEARANCE':
                        value['appearance']=dict(table_rgb=appearance.choice([[.35,.35,.35],[.55,.55,.55],[.70,.65,.55]]),roughness=appearance.choice([.5,.8]),
                            illumination_gain=appearance.choice([.8,1.,1.2]),task_object_colors_unchanged=True,shader='ORIGINAL_MINIMAL',pixel_effect='UNVERIFIED_REQUIRES_RENDERED_DEV',binding_stage='VISUAL_CONSTRUCTION_BEFORE_FIRST_RENDER')
                    elif domain=='D2_EXTERNAL_CAMERA':value['camera']=camera([view.uniform(-.03,.03) for _ in range(3)],view.uniform(-5,5),view.uniform(-5,5),view.choice([80,90,100]))
                rows.append(value)
    return rows

def static_check(row):
    if row['split']=='TEST':
        if row['appearance'] is not None or row['camera'] is not None:raise ValueError('TEST parameters must remain metadata-only')
        return dict(scene_id=row['scene_id'],status='TEST_METADATA_ONLY',rendered=False)
    cam=row['camera'];t=cam['world_from_sapien_camera'];r=[x[:3] for x in t[:3]]
    if t[3]!=[0,0,0,1] or not 0<cam['fov_deg']<180:raise ValueError('Invalid camera matrix/FOV')
    error=max(abs(sum(r[k][i]*r[k][j] for k in range(3))-(i==j)) for i in range(3) for j in range(3))
    determinant=sum(r[0][i]*cross(r[1],r[2])[i] for i in range(3))
    if error>1e-10 or abs(determinant-1)>1e-10:raise ValueError('Camera rotation is not right-handed SO3')
    # Conservative native-placement envelope; seed-resolved geometry remains unavailable.
    outside=0;tan=math.tan(math.radians(cam['fov_deg'])/2)
    for x in (-.2,.2):
        for y in (-.3,.3):
            delta=[x-t[0][3],y-t[1][3],.02-t[2][3]];local=[sum(r[k][j]*delta[k] for k in range(3)) for j in range(3)]
            outside+=not (cam['near']<local[0]<cam['far'] and abs(local[1])<=local[0]*tan and abs(local[2])<=local[0]*tan)
    return dict(scene_id=row['scene_id'],status='CPU_STATIC_TRANSFORM_PASS',orthogonality_max_abs=error,determinant=determinant,
        conservative_envelope_corners_outside_frustum=outside,seed_resolved_collision='NOT_OBSERVED',rendered_visibility='NOT_RUN',
        requires_future_DEV_visibility_check=outside>0,proposal_replaced=False)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);a=parser.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    rows=cards();assert len(rows)==456 and len({r['parent_group'] for r in rows})==152
    (a.output/'SCENE_CARDS.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
    manifest=dict(status='SCENE_SPEC_READY_NATIVE_NOT_RUN',release=RELEASE,generator_sha256=sha(__file__),source_commit=UPSTREAM,
        groups={split:[f'{split}-{seed}' for seed in seeds] for split,seeds in SPLITS.items()},counts=dict(base_groups=152,cards=456,TRAIN_independent_groups=96,TRAIN_correlated_cards=288),
        TEST_metadata_only=True,all_correlated_variants_same_split=True,original_PI0276_TEST_consulted=False,native_calls=0,render_calls=0,model_calls=0)
    (a.output/'SPLIT_GROUP_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (a.output/'STATIC_SCENE_CHECKS.json').write_text(json.dumps([static_check(r) for r in rows],indent=2)+'\n')
    print(json.dumps(manifest))

if __name__=='__main__':main()
