"""New task geometry pairs; all native pair decisions stay identical."""
FEET=frozenset(('FL','FR','RL','RR'))
TASK_PAIRS=frozenset((frozenset(('task_table','task_cube_geom')),
    frozenset(('task_left_pad','task_cube_geom')),frozenset(('task_right_pad','task_cube_geom'))))


def original_allowed(a,b):
    return a=='floor' and b in FEET or b=='floor' and a in FEET


def allowed(a,b):
    return original_allowed(a,b) or frozenset((a,b)) in TASK_PAIRS


def make_observer(source,output):
    """Generate isolated task observer; never edit or unload the original."""
    import hashlib
    from pathlib import Path
    raw=Path(source).read_bytes()
    if hashlib.sha256(raw).hexdigest()!='f4937ff6832bcb6ca06224900eccea9036664926a43b80861481b410efb99b92':
        raise ValueError('EXACT_ORIGINAL_OBSERVER_REQUIRED')
    text=raw.decode()
    old='    illegal_contact|=!allowed;'
    new='''    auto pair=[&](const char* x,const char* y) {
      return a && b && ((std::string(a)==x && std::string(b)==y) || (std::string(a)==y && std::string(b)==x));
    };
    bool task_pair=pair("task_table","task_cube_geom") || pair("task_left_pad","task_cube_geom") || pair("task_right_pad","task_cube_geom");
    allowed=allowed || task_pair;
    illegal_contact|=!allowed || (task_pair && d->contact[i].dist < -.004);'''
    if text.count(old)!=1:raise ValueError('OBSERVER_PATCH_SITE_CHANGED')
    with Path(output).open('x') as f:f.write(text.replace(old,new))
