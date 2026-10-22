"""Bind prefix evidence to the EXISTING Core ledger; no admission or physics.

The admitted worker owns one outer 16000-live reservation and the original
constructor/clone meter. Each shadow owns exactly one original ledger context.
Live prefix callbacks are evidence only, never a second debit. Unknown effects
retain the original reservation; known partial rejection closes actual counts.
"""
import threading


class SourceComponentUnknown(RuntimeError):
    pass


class SourceComponentMeters:
    def __init__(self,ledger,*,episode_id,outer_live_meter,record):
        if (not isinstance(episode_id,str) or not episode_id or not callable(record)
                or type(outer_live_meter) is not dict or outer_live_meter.get('actual')!=0
                or type(outer_live_meter.get('actual')) is not int):
            raise ValueError('ADMITTED_OUTER_LIVE_CONTEXT_AND_OWNER_EVIDENCE_REQUIRED')
        self.ledger=ledger;self.episode_id=episode_id;self.outer=outer_live_meter;self.record=record
        self.thread=threading.get_ident();self.counter=0;self.entries={};self.active={};self.unknown=False

    def _owner(self):
        if threading.get_ident()!=self.thread:
            self.unknown=True;self.outer['component_unknown']=True
            raise SourceComponentUnknown('SINGLE_COMPONENT_METER_OWNER_REQUIRED')

    def reserve(self,kind,ticks,clones):
        self._owner()
        if (self.unknown or kind not in ('shadow','live') or type(ticks) is not int or ticks!=200
                or type(clones) is not int or clones!=int(kind=='shadow') or self.active):
            raise PermissionError('ONE_ORDERED_200_TICK_PREFIX_CONTEXT_NO_NEW_WORK_AFTER_UNKNOWN')
        self.ledger.guard();self.counter+=1;token=self.episode_id+':'+kind+':'+str(self.counter)
        entry=dict(kind=kind,known=0,uncertain=False,attempted_finish=False,context=None,meter=None)
        # Record intent before entering a possibly persisted native reservation.
        # If either callback raises, its result is unknown and must not be retried.
        self.entries[token]=entry;self.active[kind]=token
        try:
            self.record(dict(kind='COMPONENT_PREFIX_RESERVE_INTENT',token=token,domain=kind,ticks=200,clones=clones))
            if kind=='shadow':
                context=self.ledger.steps('shadow_integrations',200,token)
                entry['context']=context
                entry['meter']=context.__enter__()
            self.record(dict(kind='COMPONENT_PREFIX_RESERVED',token=token,domain=kind,
                ledger_debit='SHADOW_CONTEXT' if kind=='shadow' else 'EXISTING_OUTER_LIVE_ONLY'))
        except BaseException:
            self.unknown=True;self.outer['component_unknown']=True;entry['uncertain']=True
            raise
        return token

    def charge(self,kind,count,unknown,context):
        self._owner()
        token=self.active.get(kind);entry=self.entries.get(token)
        if (entry is None or entry['attempted_finish'] or type(count) is not int or count<0
                or type(unknown) is not bool or type(context) is not dict or entry['known']+count>200):
            self.unknown=True;self.outer['component_unknown']=True
            raise SourceComponentUnknown('PREFIX_KNOWN_COUNTER_OR_CALLBACK_IDENTITY_UNKNOWN')
        entry['known']+=count;entry['uncertain']|=unknown
        if unknown:self.unknown=True;self.outer['component_unknown']=True
        try:self.record(dict(kind='COMPONENT_PREFIX_NATIVE_EFFECT',token=token,domain=kind,
            actual_increment=count,actual_lower_bound=entry['known'],unknown=unknown,context=dict(context)))
        except BaseException:
            entry['uncertain']=True;self.unknown=True;self.outer['component_unknown']=True
            raise

    def finish(self,kind,token,count,unknown,status):
        self._owner();entry=self.entries.get(token)
        if (entry is None or self.active.get(kind)!=token or entry['kind']!=kind or entry['attempted_finish']):
            raise SourceComponentUnknown('PREFIX_FINISH_UNKNOWN_OR_ALREADY_ATTEMPTED_NO_RETRY')
        entry['attempted_finish']=True
        valid=(type(count) is int and count==entry['known'] and 0<=count<=200
            and type(unknown) is bool and status in (
                'PASS','REFERENCE_REJECT','S1_REJECT','NATIVE_SAFETY_ERROR','ENGINEERING_MISSING')
            and (status!='PASS' or count==200 and unknown is False))
        uncertain=not valid or unknown or entry['uncertain']
        if uncertain:self.unknown=True;self.outer['component_unknown']=True
        try:
            if kind=='shadow':
                if entry['meter'] is None:
                    raise SourceComponentUnknown('SHADOW_RESERVE_OUTCOME_UNKNOWN_NO_CONTEXT_EXIT_RETRY')
                entry['meter']['actual']=entry['known']
                if uncertain:
                    error=SourceComponentUnknown('ACTUAL_SHADOW_LOWER_BOUND_WITH_ORIGINAL_RESERVATION_RETAINED')
                    entry['context'].__exit__(type(error),error,None)
                else:
                    # Settle real partial effects BEFORE scientific rejection is
                    # propagated by the caller; do not throw it through steps().
                    entry['context'].__exit__(None,None,None)
            self.record(dict(kind='COMPONENT_PREFIX_FINISHED',token=token,domain=kind,
                known_ticks=entry['known'],reservation_retained=uncertain,status=status,
                ledger_debit='ORIGINAL_SHADOW_CONTEXT_ONLY' if kind=='shadow' else 'EXISTING_OUTER_LIVE_ONLY'))
        except BaseException:
            self.unknown=True;self.outer['component_unknown']=True;entry['uncertain']=True
            raise
        finally:self.active.pop(kind,None)
        if not valid:raise SourceComponentUnknown('PREFIX_FINISH_COUNTER_MISMATCH_RETAINED')

    def reconcile_outer(self,native_steps):
        """Call inside the original live context, even when episode.run fails.

        A known partial safety/reference reject can return through its context.
        Unknown callback/native effects raise so Ledger.steps keeps its full bound.
        """
        self._owner()
        if type(native_steps) is int and 0<=native_steps<=16000:self.outer['actual']=native_steps
        else:self.unknown=True;self.outer['component_unknown']=True
        if self.active or self.unknown or self.outer.get('component_unknown'):
            raise SourceComponentUnknown('KEEP_OUTER_LIVE_RESERVATION_WITH_ACTUAL_LOWER_BOUND')
        return self.outer['actual']
