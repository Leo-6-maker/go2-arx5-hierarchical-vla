"""Two independent paused worlds sharing one sequential inference consumer."""

from pathlib import Path

from scripts.pi0292.durable import write
from .hybrid_nonphysical_v3 import read, identity


def message_route(world, request):
    return dict(environment_id=world, episode_id=request['episode'],
                capture_id=request['input']['sha256'], native_step=request['native_step'],
                request_id=request['request_id'],
                package_sha256=request['bindings']['policy_package_sha256'])


class WorldQueue:
    """One bounded mailbox per world, round-robin selection of live requests."""

    def __init__(self, root, registration):
        self.root = Path(root).resolve(strict=True)
        self.worlds = tuple(registration)
        if self.worlds not in (('W1',),('W1','W2')):
            raise PermissionError('D2_QUEUE_REQUIRES_FIXED_ONE_OR_TWO_WORLDS')
        self.registration = {}
        endpoints = set()
        for world, cases in registration.items():
            paths = tuple(Path(path) for path in cases)
            if not paths or len(set(paths))!=len(paths) or set(paths)&endpoints:
                raise PermissionError('D2_QUEUE_CASE_BELONGS_TO_EXACTLY_ONE_WORLD')
            if any(p != p.resolve() or p.parent.name!='cases' for p in paths):
                raise PermissionError('D2_QUEUE_CANONICALLY_REGISTERED_CASE_PATH_REQUIRED')
            self.registration[world] = paths
            endpoints.update(paths)
        self.cursor = 0

    def path(self, world):
        if world not in self.worlds:
            raise PermissionError('D2_QUEUE_UNKNOWN_WORLD')
        return self.root/f'REQUEST.{world}.json'

    def validate(self, world, message):
        request = message['request']
        if (request['clock_mode'] != 'SIM_PAUSED_ONLY'
                or identity({k:v for k,v in request.items() if k!='request_id'}) != request['request_id']
                or message.get('d2_route') != message_route(world,request)
                or message.get('branch_barriers') is not False):
            raise ValueError('D2_QUEUE_WRONG_REQUEST_OR_PAUSED_ROUTE')
        folder = Path(message['response_root'])
        live = Path(message['live_path'])
        capture = Path(request['input']['path']).parent
        if (folder != capture/'model-response' or live != capture/'inference-control'/'live.json'
                or folder.resolve() != folder or live.resolve() != live):
            raise ValueError('D2_QUEUE_EXACT_CAPTURE_LOCAL_REPLY_AND_LIVENESS_PATHS_REQUIRED')
        candidates = [p for p in self.registration[world] if p in capture.parents
                      and request['episode'].startswith(p.name+'-q')]
        if len(candidates)!=1:
            raise ValueError('D2_QUEUE_CAPTURE_OUTSIDE_OWNED_EPISODE')
        return message

    def publish(self, world, message):
        path = self.path(world)
        message = dict(message,d2_route=message_route(world,message['request']))
        self.validate(world,message)
        try:
            previous = read(path)
        except FileNotFoundError:
            previous = None
        if previous is not None:
            if previous==message:
                return message['d2_route']
            state = read(previous['live_path'])
            if not (state['cancelled'] or state['stop']
                    or (Path(previous['response_root'])/'RESPONSE.json').is_file()):
                raise RuntimeError('D2_QUEUE_WORLD_ALREADY_HAS_A_PAUSED_REQUEST')
        write(path,message)
        return message['d2_route']

    def next(self, handled):
        for offset in range(len(self.worlds)):
            index = (self.cursor+offset)%len(self.worlds)
            world = self.worlds[index]
            try:
                message = read(self.path(world))
            except FileNotFoundError:
                continue
            self.validate(world,message)
            if message['request']['request_id'] in handled:
                continue
            self.cursor = (index+1)%len(self.worlds)
            return message
        return None
