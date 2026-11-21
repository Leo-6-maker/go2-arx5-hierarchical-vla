"""Exercise a loopback-only DDS candidate in two unused project domains, without a robot."""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4

DOMAINS = tuple(range(192,198))
MESSAGES = 32
PROFILE = """<?xml version="1.0"?>
<profiles xmlns="http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles">
  <transport_descriptors>
    <transport_descriptor>
      <transport_id>core_loopback_udp</transport_id>
      <type>UDPv4</type>
      <interfaceWhiteList><address>127.0.0.1</address></interfaceWhiteList>
    </transport_descriptor>
  </transport_descriptors>
  <participant profile_name="core_loopback_test" is_default_profile="true">
    <rtps>
      <useBuiltinTransports>false</useBuiltinTransports>
      <userTransports><transport_id>core_loopback_udp</transport_id></userTransports>
      <builtin>
        <metatrafficUnicastLocatorList>
          <locator><udpv4><address>127.0.0.1</address></udpv4></locator>
        </metatrafficUnicastLocatorList>
        <initialPeersList>
          <locator><udpv4><address>127.0.0.1</address></udpv4></locator>
        </initialPeersList>
      </builtin>
    </rtps>
  </participant>
</profiles>
"""


def seal(path, value):
    with path.open('x') as out:
        json.dump(value,out,sort_keys=True)
        out.flush()
        os.fsync(out.fileno())


def mappings():
    maps = [line.split(maxsplit=5) for line in Path('/proc/self/maps').read_text().splitlines()]
    files = {row[5]:int(row[4]) for row in maps if len(row)==6}
    paths = sorted(p for p in files if p.startswith(('/dev/shm/', '/SYSV')))
    toolkit = sorted(p for p in files if 'libfastrtps.so.' in p or 'librmw_fastrtps_cpp.so' in p)
    realm = os.environ.get('SLA_MULTIWORLD_UST_REALM')
    local = {p:inode for p,inode in files.items() if realm and p.startswith(realm+'/') and p.endswith('.wait')}
    return dict(shm=paths,libs=toolkit,private_ust=local)


def ust_separated(maps):
    realm = os.environ.get('SLA_MULTIWORLD_UST_REALM')
    if not realm:
        return True
    expected = {str(p):p.stat().st_ino for p in (Path(realm)/'user.wait', Path(realm)/'global.wait')}
    return maps['private_ust']==expected and not any('/lttng-ust-' in f for f in maps['shm'])


def expected(domain):
    return [f'core-slot-{domain}:{ordinal:02d}' for ordinal in range(MESSAGES)]


def actor(root, domain, role, topic):
    import rclpy
    from rclpy.utilities import get_rmw_implementation_identifier
    from std_msgs.msg import String

    if domain not in DOMAINS or os.environ.get('ROS_DOMAIN_ID')!=str(domain):
        raise PermissionError('Unexpected DDS probe domain')
    rclpy.init(args=[])
    node = rclpy.create_node(f'core_probe_{role}_{topic.rsplit("_",1)[-1]}')
    received = []
    if role=='send':
        pub = node.create_publisher(String,topic,64)
    else:
        sub = node.create_subscription(String,topic,lambda msg: received.append(msg.data),64)
    before = mappings()
    until = time.monotonic()+2
    while not ust_separated(before) and time.monotonic()<until:
        rclpy.spin_once(node,timeout_sec=.02)
        before = mappings()
    seal(root/f'{domain}-{role}.MAPPINGS.json',before)
    if (not ust_separated(before) or any('/fastrtps_' in f for f in before['shm'])
            or get_rmw_implementation_identifier()!='rmw_fastrtps_cpp'):
        raise RuntimeError('The scoped XML did not select the intended transport')
    seal(root/f'{domain}-{role}.READY.json',dict(pid=os.getpid(),domain=domain,maps=before,epoch=time.time()))
    start = time.monotonic()
    deadline = start+12
    samples = []
    try:
        while time.monotonic()<deadline:
            rclpy.spin_once(node,timeout_sec=.04)
            if role=='send':
                matched = pub.get_subscription_count()
                if matched>1:
                    raise RuntimeError('Unexpected cross-domain subscription')
                if matched==1:
                    expected_messages = expected(domain)
                    for value in expected_messages:
                        pub.publish(String(data=value))
                    if not pub.wait_for_all_acked(timeout=rclpy.duration.Duration(seconds=3)):
                        raise TimeoutError('UDP fixture acknowledgments did not complete')
                    samples=expected_messages
                    break
            if role=='receive':
                if sub.get_publisher_count()>1:
                    raise RuntimeError('Unexpected cross-domain publisher')
                if received!=expected(domain)[:len(received)]:
                    raise ValueError('Unexpected sender or out-of-order transport message')
                if len(received)==MESSAGES:
                    until=time.monotonic()+.35
                    while time.monotonic()<until:
                        rclpy.spin_once(node,timeout_sec=.025)
                    samples=received[:]
                    break
        after=mappings()
        if samples!=expected(domain) or not ust_separated(after) or any('/fastrtps_' in f for f in after['shm']):
            raise TimeoutError('The registered fixture sequence or transport changed')
        seal(root/f'{domain}-{role}.json',dict(pid=os.getpid(),matched=len(samples),values_sha=hashlib.sha256(
             json.dumps(samples).encode()).hexdigest(),elapsed=time.monotonic()-start,maps=after))
    finally:
        node.destroy_node()
        rclpy.shutdown()


def probe(root, control, domains, profile_check, ust_library=None):
    if (os.getuid()!=2009 or os.uname().nodename!='corelab-4080s' or not root.is_dir()
            or root.stat().st_uid!=os.getuid() or len(set(domains))!=2):
        raise PermissionError('Use the existing Core project owner')
    profile=root/'LOOPBACK.xml'
    topic='/sla_core_dds_probe_'+uuid4().hex
    profile.write_text(PROFILE)
    profile_env=dict(os.environ,FASTDDS_DEFAULT_PROFILES_FILE=str(profile),
                     FASTRTPS_DEFAULT_PROFILES_FILE=str(profile))
    checked=subprocess.run([str(profile_check),str(profile)],env=profile_env,
                           text=True,capture_output=True,timeout=5)
    parsed=checked.stdout.strip()
    if checked.returncode or parsed!='VERIFIED_UDP4_LOOPBACK_ONLY':
        raise PermissionError(f'Fast DDS loopback profile preflight {checked.returncode}: {checked.stderr.strip()}')
    with ExitStack() as hold:
        for domain in domains:
            lock=control/f'ros-{domain}.lock'
            if not lock.is_file() or lock.is_symlink() or lock.stat().st_uid!=os.getuid():
                raise PermissionError('Expected an existing project ROS-domain lease')
            fd=hold.enter_context(lock.open('r+b'))
            fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        sockets=subprocess.check_output(['ss','-H','-lun'],text=True,timeout=4)
        ports={int(row.split()[3].rsplit(':',1)[-1]) for row in sockets.splitlines()}
        if any(7400+250*d<=p<7400+250*(d+1) for p in ports for d in domains):
            raise PermissionError('A reserved test domain already has a listening socket')
        children=[]
        statuses=[]
        try:
            for domain in domains:
                realm = root/f'ust-{domain}'
                if ust_library:
                    realm.mkdir(mode=0o700)
                    for page in ('user.wait', 'global.wait'):
                        with os.fdopen(os.open(realm/page,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),'wb') as f:
                            f.truncate(os.sysconf('SC_PAGE_SIZE'))
                for role in ('receive','send'):
                    log=hold.enter_context((root/f'{domain}-{role}.log').open('xb'))
                    # Jazzy adds a SHM transport under LOCALHOST even when the XML disables the defaults.
                    env=dict(os.environ,ROS_DOMAIN_ID=str(domain),ROS_AUTOMATIC_DISCOVERY_RANGE='SYSTEM_DEFAULT',
                        ROS_STATIC_PEERS='',FASTDDS_DEFAULT_PROFILES_FILE=str(profile),
                        RMW_IMPLEMENTATION='rmw_fastrtps_cpp',RMW_FASTRTPS_USE_QOS_FROM_XML='0',
                        FASTRTPS_DEFAULT_PROFILES_FILE=str(profile),CUDA_VISIBLE_DEVICES='')
                    env.pop('ROS_DISCOVERY_SERVER',None)
                    env.pop('ROS_LOCALHOST_ONLY',None)
                    if ust_library:
                        env.update(LD_PRELOAD=str(ust_library.resolve(strict=True)),LTTNG_HOME=str(realm),
                                   SLA_MULTIWORLD_UST_REALM=str(realm),LTTNG_UST_REGISTER_TIMEOUT='0')
                    command=[sys.executable,'-B',str(Path(__file__).resolve()),'actor',
                             str(root),str(domain),role,topic]
                    children.append(subprocess.Popen(command,env=env,cwd=root,stdout=log,stderr=subprocess.STDOUT))
            until = time.monotonic()+16
            for child in children:
                try:
                    child.wait(timeout=max(.01,until-time.monotonic()))
                except subprocess.TimeoutExpired:
                    continue
        finally:
            for p in children:
                if p.poll() is None:
                    p.terminate()
                    try:
                        p.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        p.kill()
                        p.wait(timeout=3)
                statuses.append(dict(pid=p.pid,exit=p.returncode))
        success = len(statuses)==4 and all(row['exit']==0 for row in statuses)
        rows={f.stem:json.loads(f.read_bytes()) for f in root.glob('[0-9]*.json') if len(f.name.split('.'))==2}
        observed={f.name:json.loads(f.read_bytes()) for f in root.glob('*.MAPPINGS.json')}
        if success and len(rows)==4:
            for d in domains:
                if rows[f'{d}-receive']['values_sha']!=rows[f'{d}-send']['values_sha']:
                    raise ValueError('Delivered bytes differ between the two endpoints')
            if ust_library:
                mapped = [{name:rows[f'{domain}-{name}']['maps']['private_ust'] for name in ('send','receive')}
                          for domain in domains]
                if (any(x['send']!=x['receive'] or len(x['send'])!=2 for x in mapped)
                        or set(mapped[0]['send'].values()) & set(mapped[1]['send'].values())):
                    raise ValueError('LTTng wait pages are shared across the two fixture domains')
        result=dict(status='LOOPBACK_FIXTURE_PASS' if success and len(rows)==4 else 'LOOPBACK_FIXTURE_FAILED',
                    scopes=domains,expected=MESSAGES*len(domains),results=rows,topic=topic,
                    parsed_profile=parsed,mappings=observed,child_exits=statuses,
                    ust_probe_sha256=hashlib.sha256(ust_library.read_bytes()).hexdigest() if ust_library else None,
                    profile_sha256=hashlib.sha256(profile.read_bytes()).hexdigest(),new_native_requests=0,
                    simulation_integrations=0,optimizer_updates=0,model_generations=0,epoch=time.time())
        seal(root/'RESULT.json',result)
        print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_subparsers(dest='mode',required=True)
    p=modes.add_parser('probe')
    p.add_argument('root',type=Path)
    p.add_argument('control',type=Path)
    p.add_argument('domains',type=int,choices=DOMAINS,nargs=2)
    p.add_argument('--profile-check',type=Path,required=True)
    p.add_argument('--ust-library',type=Path)
    a=modes.add_parser('actor')
    a.add_argument('root',type=Path)
    a.add_argument('domain',type=int,choices=DOMAINS)
    a.add_argument('role',choices=('receive','send'))
    a.add_argument('topic')
    args=parser.parse_args()
    if args.mode=='probe':
        probe(args.root,args.control,args.domains,args.profile_check,args.ust_library)
    else:
        actor(args.root,args.domain,args.role,args.topic)
