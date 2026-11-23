"""Build a separate gripper-only variant using the pinned original source and site compiler."""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


def sha(f):
    return hashlib.sha256(f.read_bytes()).hexdigest()


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("frozen source does not match the registered hook")
    return text.replace(old, new)


def build(spec, out):
    lib = Path(spec["outer_library"])
    p = lib.with_suffix(".cpp")
    if (
        sha(lib) != "acb7627e986301dad5ffea933ff320ae474c0faeef09750b7ea22be5709dc187"
        or sha(p) != "490c41c0dea0e28580f2cdae06903d020333d00654e39d11fb5d573e55cd63c5"
    ):
        raise ValueError("unexpected receiver ABI")
    cmd = next(
        x["argv"]
        for x in json.loads((p.parent / "COMMANDS.private.json").read_bytes())
        if x["label"] == "task_actuation_r1_compile"
    )
    if cmd[0] != "g++" or cmd[-2:] != ["-o", str(lib)] or cmd.count(str(p)) != 1:
        raise ValueError("unexpected build contract")
    out.mkdir(parents=True, exist_ok=False)
    # Rebuild untouched source first to prove all headers, options and linked ABI match the original binary.
    control = out / "control"
    control.mkdir()

    def compile(src, dst):
        args = [
            str(src) if x == str(p) else str(dst) if x == str(lib) else x for x in cmd
        ]
        subprocess.run(args, check=True)

    compile(p, control / lib.name)
    if sha(control / lib.name) != sha(lib):
        raise ValueError("control build differs from pinned native receiver")
    text = p.read_text()
    text = replace_once(
        text,
        "namespace {",
        '#include "student_receiver.hpp"\nnamespace {\nstudent_grip::Session student;',
    )
    text = replace_once(
        text,
        " initialized=true;",
        """ const char* cap=std::getenv("V16_GRIP_STOP_TICK");
 if(!cap || std::strcmp(cap,"6503"))std::abort();
 student.reset(std::getenv("V16_GRIP_DIR"),std::getenv("V16_GRIP_EPOCH"),6503,target);
 initialized=true;""",
    )
    a = text.index("void read_command(")
    b = text.index("void record(", a)
    text = text[:a] + text[b:]
    text = replace_once(
        text,
        " read_command(d);",
        """ const double measured_before[2]={d->qpos[m->jnt_qposadr[joint[0]]],d->qpos[m->jnt_qposadr[joint[1]]]};
 if(!student.receive(steps,d->time,target,measured_before)){
   student.stop("RECEIVER_REJECT_STOP",steps,target,measured_before);
   fail("V16_GRIPPER_TERMINAL");return;
 }
 has_command=student.active; sequence=student.last.sequence; opening=student.last.opening; expiry=student.last.expiry;""",
    )
    text = replace_once(
        text,
        " ++steps;record(m,d);",
        """ ++steps;record(m,d);
 const double measured_after[2]={d->qpos[m->jnt_qposadr[joint[0]]],d->qpos[m->jnt_qposadr[joint[1]]]};
 bool contact_ok=true;
 for(int i=0;i<d->ncon;++i) {
   const char* a=mj_id2name(m,mjOBJ_GEOM,d->contact[i].geom1);
   const char* b=mj_id2name(m,mjOBJ_GEOM,d->contact[i].geom2);
   auto support=[](const char* x,const char* y){
     if(!x || !y)return false;
     if(!std::strcmp(x,"task_table"))return !std::strcmp(y,"task_cube_geom");
     return !std::strcmp(x,"floor") && (!std::strcmp(y,"FL") || !std::strcmp(y,"FR") || !std::strcmp(y,"RL") || !std::strcmp(y,"RR"));
   };
   if(!support(a,b) && !support(b,a))contact_ok=false;
 }
 if(!contact_ok)student.stop("PROHIBITED_CONTACT_STOP",steps,target,measured_after);
 student.observed(steps,target,measured_after);
 if(!contact_ok){fail("V16_NO_CONTACT_VIOLATION");return;}""",
    )
    h = Path(__file__).with_name("student_receiver.hpp")
    shutil.copyfile(h, out / h.name)
    generated = out / p.name
    generated.write_text(text)
    dst = out / "student_grip.so"
    compile(generated, dst)
    return dict(
        original_binary=sha(lib),
        original_source=sha(p),
        rebuilt_control=sha(control / lib.name),
        student_binary=sha(dst),
        student_source=sha(generated),
        student_header=sha(h),
        stop_tick=6503,
        new_integrations=0,
    )


if __name__ == "__main__":
    print(
        json.dumps(
            build(json.loads(Path(sys.argv[1]).read_bytes()), Path(sys.argv[2])),
            indent=2,
        )
    )
