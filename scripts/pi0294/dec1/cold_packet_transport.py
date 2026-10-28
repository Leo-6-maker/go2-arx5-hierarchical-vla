"""Frame cold control stdin data; no network, retry, admission or time override."""
import gzip
import hashlib

MAX_PACKET_BYTES=128*1024**2
MAX_WIRE_BYTES=16*1024**2
MAX_PROGRAM_BYTES=96*1024

def frame(program,packet_bytes,*,action):
    if type(program) is not str or type(packet_bytes) is not bytes:
        raise TypeError('EXACT_CONTROL_PROGRAM_AND_PACKET_BYTES')
    if action not in ('deploy','launch'):
        raise ValueError('KNOWN_COLD_CONTROL_ACTION_ONLY')
    if not 0<len(packet_bytes)<=MAX_PACKET_BYTES or len(program.encode())>MAX_PROGRAM_BYTES:
        raise ValueError('BOUNDED_PACKET_AND_CONTROL_PROGRAM')
    compile(program,'sealed-cold-control-program','exec')
    wire=gzip.compress(packet_bytes,mtime=0)
    if len(wire)>MAX_WIRE_BYTES:raise ValueError('BOUNDED_COLD_WIRE_BYTES')
    packet_sha=hashlib.sha256(packet_bytes).hexdigest();wire_sha=hashlib.sha256(wire).hexdigest()
    bootstrap='\n'.join([
        'import base64,gzip,hashlib,sys',
        '_cold_wire=sys.stdin.buffer.read('+str(len(wire)+1)+')',
        "assert len(_cold_wire)=="+str(len(wire))+",'EXACT_COLD_WIRE_LENGTH'",
        "assert hashlib.sha256(_cold_wire).hexdigest()=="+repr(wire_sha)+",'EXACT_COLD_WIRE_SHA256'",
        '_cold_packet=gzip.decompress(_cold_wire)',
        "assert len(_cold_packet)=="+str(len(packet_bytes))+",'EXACT_COLD_PACKET_LENGTH'",
        "assert hashlib.sha256(_cold_packet).hexdigest()=="+repr(packet_sha)+",'EXACT_COLD_PACKET_SHA256'",
        'P=dict(packet_base64=base64.b64encode(_cold_wire).decode(),packet_sha256='+repr(packet_sha)+',action='+repr(action)+')',
        'del _cold_wire,_cold_packet',program])
    compile(bootstrap,'sealed-cold-data-bootstrap','exec')
    return dict(program=bootstrap,stdin_bytes=wire,packet_sha256=packet_sha,wire_sha256=wire_sha,
                original_program_sha256=hashlib.sha256(program.encode()).hexdigest(),
                automatic_retry=False,requires_fresh_native_admission=True,deadline_override=False)
