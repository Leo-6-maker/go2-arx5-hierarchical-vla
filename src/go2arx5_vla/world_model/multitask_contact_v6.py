"""Explicit two-object roles over the original contact predicate and limits.

This candidate does not replace any loaded observer or grant native admission.
It retains feet/floor, object/table and selected-object/pad contact classes;
wrong-object/pad and every pad/table pair remain forbidden.
"""
OBJECT_GEOMS={'red':'task_cube_geom','blue':'task_blue_cube_geom'}
PADS=frozenset(('task_left_pad','task_right_pad'))
PENETRATION_LIMIT=.004


def allowed_pair(a,b,selected):
    from scripts.pi0294.task_contacts import original_allowed
    if selected not in OBJECT_GEOMS:raise ValueError('FROZEN_RED_OR_BLUE_TASK_SELECTION_REQUIRED')
    if original_allowed(a,b):return True
    pair=frozenset((a,b));target=OBJECT_GEOMS[selected]
    if any(pair==frozenset(('task_table',obj))for obj in OBJECT_GEOMS.values()):return True
    return any(pair==frozenset((pad,target))for pad in PADS)


def observer_permitted(a,b,distance,selected):
    permitted=allowed_pair(a,b,selected)
    task_pair=any(x in OBJECT_GEOMS.values()for x in(a,b))
    return bool(permitted and not(task_pair and distance<-PENETRATION_LIMIT))


def preview_permitted(a,b,distance,stage,selected):
    pair=frozenset((a,b));target=OBJECT_GEOMS[selected]
    if stage in('approach','align')and any(pair==frozenset((p,target))for p in PADS):return False
    return observer_permitted(a,b,distance,selected)


def observer_source(original_source):
    """Cold source candidate, same STOP latch and4mm task-contact threshold."""
    import hashlib
    if hashlib.sha256(original_source).hexdigest()!='99b6018167719dd9970517ab8cc11b826fbc137ac9823ff93367b93914ec6a9a':
        raise ValueError('EXACT_FROZEN_SERIALIZED_TRACE_OBSERVER_SOURCE_REQUIRED')
    source=original_source.decode()
    old='    bool task_pair=pair("task_table","task_cube_geom") || pair("task_left_pad","task_cube_geom") || pair("task_right_pad","task_cube_geom");'
    replacement='''    const char* selected=std::getenv("MFREE_V6_SELECTED_COLOR");
    if(!selected || (std::string(selected)!="red" && std::string(selected)!="blue"))std::abort();
    const char* target=std::string(selected)=="red"?"task_cube_geom":"task_blue_cube_geom";
    bool support=pair("task_table","task_cube_geom") || pair("task_table","task_blue_cube_geom");
    bool grasp=pair("task_left_pad",target) || pair("task_right_pad",target);
    bool task_pair=support || grasp;'''
    if source.count(old)!=1:raise ValueError('ORIGINAL_OBSERVER_CONTACT_PATCH_SITE_CHANGED')
    return source.replace(old,replacement).encode()
