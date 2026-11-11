"""Quote a remote argv across two OpenSSH command-joining shell boundaries."""
import shlex

OPTIONS=['-T','-o','BatchMode=yes','-o','ConnectTimeout=10',
         '-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2']


def via(relay,target,remote_argv):
    # OpenSSH joins its remaining argv before the destination shell parses it.
    # Preserve the entire inner command as one argument at the relay boundary.
    inner=shlex.join(remote_argv)
    outer=shlex.join(['ssh',*OPTIONS,target,inner])
    return ['ssh',*OPTIONS,relay,outer]
