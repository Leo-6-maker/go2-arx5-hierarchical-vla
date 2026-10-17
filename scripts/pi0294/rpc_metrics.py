"""RPC telemetry is not a science charge and must not serialize each reply."""
import math


class SessionMetrics:
    def __init__(self):
        self.operations=0
        self.settlement_started=False

    def reply(self,value,emit):
        if self.settlement_started or self.operations>=40000:
            raise PermissionError('FINITE_OPEN_RPC_METRICS_SESSION_ONLY')
        self.operations+=1
        emit(value)

    def settle(self,ledger,cpu):
        if self.settlement_started:raise PermissionError('RPC_METRICS_SETTLEMENT_NO_BLIND_RETRY')
        if not isinstance(cpu,(int,float)) or not math.isfinite(cpu) or cpu<0:
            raise ValueError('FINITE_POSITIVE_RPC_CPU_ACCOUNTING')
        self.settlement_started=True
        # All neural/physical/update charges remain in their original transactions.
        with ledger.transaction() as value:
            value['RPC_calls']=value.get('RPC_calls',0)+self.operations
            key='Core_CPU_s' if ledger.role=='core' else 'CPU_s'
            value[key]=value.get(key,0)+cpu
        return dict(RPC_calls=self.operations,CPU_s=cpu,
            telemetry_settled_once=True,scientific_resource_transactions_changed=False)
