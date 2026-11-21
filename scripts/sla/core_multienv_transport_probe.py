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
    paths = [line.split(maxsplit=5)[-1] for line in Path('/proc/self/maps').read_text().splitlines()
             if ' /dev/shm/' in line or ' /SYSV' in line]
    toolkit = [line.split(maxsplit=5)[-1] for line in Path('/proc/self/maps').read_text().splitlines()
               if 'libfastrtps.so.' in line or 'librmw_fastrtps_cpp.so' in line]
    return dict(shm=sorted(set(paths)),libs=sorted(set(toolkit)))


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
    seal(root/f'{domain}-{role}.MAPPINGS.json',before)
    if any('/fastrtps_' in f for f in before['shm']) or get_rmw_implementation_identifier()!='rmw_fastrtps_cpp':
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
        if samples!=expected(domain) or any('/fastrtps_' in f for f in after['shm']):
            raise TimeoutError('The registered fixture sequence or transport changed')
        seal(root/f'{domain}-{role}.json',dict(pid=os.getpid(),matched=len(samples),values_sha=hashlib.sha256(
             json.dumps(samples).encode()).hexdigest(),elapsed=time.monotonic()-start,maps=after))
    finally:
        node.destroy_node()
        rclpy.shutdown()


def probe(root, control, domains, profile_check):
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
        try:
            for domain in domains:
                for role in ('receive','send'):
                    log=hold.enter_context((root/f'{domain}-{role}.log').open('xb'))
                    # Jazzy adds a SHM transport under LOCALHOST even when the XML disables the defaults.
                    env=dict(os.environ,ROS_DOMAIN_ID=str(domain),ROS_AUTOMATIC_DISCOVERY_RANGE='SYSTEM_DEFAULT',
                        ROS_STATIC_PEERS='',FASTDDS_DEFAULT_PROFILES_FILE=str(profile),
                        RMW_IMPLEMENTATION='rmw_fastrtps_cpp',RMW_FASTRTPS_USE_QOS_FROM_XML='0',
                        FASTRTPS_DEFAULT_PROFILES_FILE=str(profile),CUDA_VISIBLE_DEVICES='')
                    env.pop('ROS_DISCOVERY_SERVER',None)
                    env.pop('ROS_LOCALHOST_ONLY',None)
                    command=[sys.executable,'-B',str(Path(__file__).resolve()),'actor',
                             str(root),str(domain),role,topic]
                    children.append(subprocess.Popen(command,env=env,cwd=root,stdout=log,stderr=subprocess.STDOUT))
            success=all(p.wait(timeout=16)==0 for p in children)
        finally:
            for p in children:
                if p.poll() is None:
                    p.terminate()
                    p.wait(timeout=3)
        rows={f.stem:json.loads(f.read_bytes()) for f in root.glob('[0-9]*.json') if len(f.name.split('.'))==2}
        observed={f.name:json.loads(f.read_bytes()) for f in root.glob('*.MAPPINGS.json')}
        if success and len(rows)==4:
            for d in domains:
                if rows[f'{d}-receive']['values_sha']!=rows[f'{d}-send']['values_sha']:
                    raise ValueError('Delivered bytes differ between the two endpoints')
        result=dict(status='LOOPBACK_FIXTURE_PASS' if success and len(rows)==4 else 'LOOPBACK_FIXTURE_FAILED',
                    scopes=domains,expected=MESSAGES*len(domains),results=rows,topic=topic,
                    parsed_profile=parsed,mappings=observed,
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
    a=modes.add_parser('actor')
    a.add_argument('root',type=Path)
    a.add_argument('domain',type=int,choices=DOMAINS)
    a.add_argument('role',choices=('receive','send'))
    a.add_argument('topic')
    args=parser.parse_args()
    if args.mode=='probe':
        probe(args.root,args.control,args.domains,args.profile_check)
    else:
        actor(args.root,args.domain,args.role,args.topic)
