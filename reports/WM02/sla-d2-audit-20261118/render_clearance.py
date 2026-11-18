from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['svg.hashsalt'] = 'sla-d2-clearance'
HERE=Path(__file__).resolve().parent
d=json.loads((HERE/"lower_clearance.json").read_text())
r=d["rows"]; t=[x["time_after_first_lower_query_s"] for x in r]
fig,ax=plt.subplots(figsize=(10.5,5.8),layout="constrained")
colors=["#075985","#ca8a04","#c026d3"]
series=[("measured_anchor_gap_mm","Geometry at measured state",colors[0]),
("nominal_P10_min_mm","Nominal P10 minimum",colors[1]),
("submitted_P10_min_mm","Submitted P10 minimum",colors[2])]
for key,label,col in series:
    ax.plot(t,[x[key] for x in r],label=label,color=col,lw=2,marker="o",ms=3)
stop_t=(d["stop_step"]-r[0]["step"])*.002
ax.scatter([stop_t],[d["stop_geometry_gap_mm"]],color="#dc2626",marker="x",s=70,zorder=6,label="Actual contact STOP (geometry recomputed)")
ax.axhline(0,color="#475569",lw=.8)
ax.set(xlabel="Native time since first LOWER query (s)",ylabel="Wrist / non-target object geometric clearance (mm)",xlim=(-.15,12.05),ylim=(-2,56))
ax.set_title("Episode 18: clearance contracts during LOWER",loc="left",fontsize=15,fontweight="bold",pad=14)
ax.text(0,1.01,"24 saved query states; 50 ms atoms / 500 ms query interval",transform=ax.transAxes,fontsize=10,color="#475569")
ax.grid(alpha=.18);ax.spines[["top","right"]].set_visible(False)
ax.legend(loc="upper right",frameon=True,fontsize=9)
ax.annotate("Last query: 0.0386 mm\nSTOP follows 110 ms later",xy=(11.5,.0386306),xytext=(1.5,8),
    arrowprops={"arrowstyle":"->","color":"#075985"},fontsize=10,color="#075985")
ax.annotate("Submitted minimum remains 9.805 mm",xy=(11.5,9.805123),xytext=(6.0,30),
    arrowprops={"arrowstyle":"->","color":"#c026d3"},fontsize=10,color="#86198f")
fig.text(.01,-.035,"Source: PR409 @ b29bd2ba, EMBODIMENT_D1_ACTUAL.json. Post-hoc geometry; selected controller reference is UNKNOWN.",
         fontsize=8.5,color="#475569")
fig.savefig(HERE/"lower_clearance.svg",bbox_inches="tight",metadata={"Date":None,"Creator":"SLA D2 audit; matplotlib; source data retained beside figure"})
fig.savefig(HERE/"lower_clearance.png",dpi=180,bbox_inches="tight")
print(json.dumps({"rows":len(r),"last_gap_mm":r[-1]["measured_anchor_gap_mm"],"stop_after_last_query_ms":(d["stop_step"]-r[-1]["step"])*2,"svg_bytes":(HERE/"lower_clearance.svg").stat().st_size}))

