#!/usr/bin/env python3
"""Generate the ten sourced, editable report figures from figures/data/."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

MPLCONFIG = Path(tempfile.gettempdir()) / "emotionalcompany-mpl-cache"
MPLCONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG))

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                            "axes.unicode_minus": False, "savefig.dpi": 300})
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.ticker import MultipleLocator
from fontTools.ttLib import TTCollection

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "figures" / "data"
FIG = ROOT / "figures"
SRC = FIG / "source"
MANIFEST = FIG / "FIGURE_MANIFEST.md"

C = {"primary":"#1C3B53", "secondary":"#35708D", "teal":"#548A86",
     "body":"#20272E", "muted":"#5E6974", "divider":"#CBD3DA",
     "light":"#F1F4F6", "pale":"#E8EFF3", "gray":"#AAB5BC",
     "palegray":"#E4E8EB", "white":"#FFFFFF", "orange":"#A66B32"}

META = {
"01-overall-pipeline": ("项目总体技术路线","figures/data/01-overall-pipeline.json","docs/context/E0.1_当前状态与后续交接.md；docs/report/全局实验方案与结果总览.md","训练与工程流程","Train 16,192；Dev 1,821","Qwen3-1.7B；A8 P4504；E03-B Replay-1152","研究阶段与模型路由","数据构建→基线→专项分支→共享 Base 双 Adapter→本地验收","流程图"),
"02-data-distribution": ("Train 情绪分布与画像稀疏性","figures/data/02-data-distribution.csv","docs/data/D0.1_数据质量审计报告.md；docs/experiments/E0.3-A0.1_画像标签与细粒度监督诊断报告.md","情绪：原始 Train；画像：SFT Train","18,657；16,192","不适用","情绪 Gold counts；画像空/非空样本数","anxiety 6,340；画像全空 6,033、非空 10,159","样本数"),
"03-sft-dataflow": ("历史展开与 Completion-only SFT 构造","figures/data/03-sft-dataflow.json","docs/data/D0.4.2_Token长度审计报告.md；docs/data/D0.4.3_监督数据构建与Batch验收报告.md","示意对话与训练预算","P≤1,280；S≤2,048","Qwen3 Tokenizer；P0/F2/T1","Prompt mask 与 Completion 监督","Prompt labels=-100；Completion+EOS 参与 Loss","Token 示意图"),
"04-qlora-training": ("QLoRA 训练结构与参数更新","figures/data/04-qlora-training.json","docs/experiments/E0.1-B_正式SFT与Dev对照验收报告.md；docs/workflow.md","E01-B Full","1 epoch；4,048 optimizer steps","Qwen3-1.7B；NF4；BF16","LoRA rank / alpha / modules","r=8；alpha=16；q/k/v/o 与 gate/up/down","配置图"),
"05-supervision-share": ("统一 SFT 监督 Token 组成","figures/data/05-supervision-share.csv","docs/experiments/E0.3-A0_监督结构诊断报告.md","正式 SFT Train","16,192 samples；2,851,614 supervised tokens","E01-B Completion","字段监督 Token share","回复 79.2621%；画像 8.8924%；情绪 1.0569%；其他 10.7885%","百分比"),
"06-a8-learning-curve": ("A8 分阶段画像 Micro-F1","figures/data/06-a8-learning-curve.csv","docs/experiments/E0.3-A8_V2_分阶段训练观察.md","Pilot160 与 Dev224","每节点 n=160 / 224","A8 P512/P1024/P2048/P4048","Profile 33-label Micro-F1","Pilot 0.04982/0.11184/0.28649/0.34679；Dev 0.02985/0.15755/0.38961/0.44051","Micro-F1"),
"07-emotion-replay": ("E03-B Replay 固定 209 点类别命中","figures/data/07-emotion-replay.csv","outputs/data-b/B-materials/e03-b/E03-B_replay_难例回放实验报告.md","固定 Val 切片","209 prediction points","checkpoint-4665 与 Replay-1152","各 Gold 类正确命中数","mixed 1→12；sadness 2→11；care 0→1；disgust 0→2；anxiety 58→49","命中数/Gold数"),
"08-dual-adapter-architecture": ("双 Adapter 串行推理与 JSON 组装","figures/data/08-dual-adapter-architecture.json","docs/experiments/E03_双Adapter最终集成与Docker验收.md；participant/README.md","最终 Docker Run 03","固定 target-free 100 条验收","单一 Base；A8 P4504；E03-B Replay-1152","Profile 后 Emotion/Response 串行生成","共享 Base、切换 Adapter、解析并组装 Schema","系统图"),
"09-final-profile-compare": ("E01-B 与 A8 画像 F1 对比","figures/data/09-final-profile-compare.csv","docs/experiments/E01-B_TrainDev_Val_最终对照与收尾.md；docs/experiments/E0.3-A8_DevFT至Val评测报告.md","同一冻结 Val ID","2,354","E01-B Train+Dev 4504 与 A8 P4504","三维与总体 Micro-F1","Personality .3770/.4479；Interests .5099/.6736；Style .3790/.3894；Overall .4014/.4678","Micro-F1"),
"10-deployment-performance": ("最终 Docker 双 Adapter 系统性能","figures/data/10-deployment-performance.csv","outputs/final_delivery/E03-DUAL-20261109-03/docker/stress100/performance_report.json；verify-output.json","最终 Docker Run 03","100 fixed target-free inputs","E03-DUAL-20261109-03；vLLM 0.11.1","端到端延迟、显存、Schema","P50 3.244649s；P90 10.206099s（最近秩派生）；P95 15.237110s；GPU 7,756/8,188 MiB；100/100","秒；MiB；行数"),
}

REG: Path
BOLD: Path
FONT_FAMILY: str


def configure() -> None:
    global REG, BOLD, FONT_FAMILY
    try:
        def font_path(pattern: str) -> Path:
            found = subprocess.run(["fc-match", "-f", "%{file}", pattern], check=True,
                                   capture_output=True, text=True).stdout.strip()
            path = Path(found)
            if not path.is_file():
                raise RuntimeError(f"Fontconfig returned a missing font file for {pattern!r}.")
            if path.suffix.lower() in (".ttc", ".otc"):
                collection = TTCollection(str(path))
                face = next((f for f in collection.fonts
                             if any(n.toUnicode() == "Noto Sans CJK SC" for n in f["name"].names if n.nameID == 1)), None)
                if face is None:
                    raise RuntimeError(f"Font collection has no Simplified Chinese face: {path}")
                static = Path(tempfile.gettempdir()) / f"emotioncompany-report-noto-sc-{pattern.rsplit(':', 1)[-1].lower()}.otf"
                face.flavor = None
                face.save(static)
                path = static
            fontManager.addfont(str(path))
            return path
        REG = font_path("Noto Sans CJK SC:style=Regular")
        BOLD = font_path("Noto Sans CJK SC:style=Bold")
    except (OSError, subprocess.CalledProcessError, RuntimeError) as exc:
        raise RuntimeError("Install Noto Sans CJK SC and make it visible to Matplotlib/Fontconfig.") from exc
    FONT_FAMILY = "Noto Sans CJK SC"
    matplotlib.rcParams.update({"font.family":FONT_FAMILY,"font.size":8.7,
        "text.color":C["body"],"axes.labelcolor":C["body"],"xtick.color":C["muted"],
        "ytick.color":C["body"],"axes.edgecolor":C["divider"],"axes.linewidth":0.7})


def fp(size: float = 8.7, bold: bool = False) -> FontProperties:
    return FontProperties(fname=str(BOLD if bold else REG), size=size)


def read_json(name: str) -> dict[str,Any]:
    return json.loads((DATA/name).read_text(encoding="utf-8"))


def read_csv(name: str) -> list[dict[str,str]]:
    with (DATA/name).open(encoding="utf-8",newline="") as f:
        return list(csv.DictReader(f))


def text(ax: Any,x: float,y: float,s: str,size: float=8.7,bold: bool=False,**kw: Any) -> Any:
    color=kw.pop("color",C["body"])
    return ax.text(x,y,s,fontproperties=fp(size,bold),color=color,**kw)


def box(ax: Any,xy: tuple[float,float],wh: tuple[float,float],label: str,kind: str="neutral",fs: float=8.5) -> None:
    x,y=xy; w,h=wh
    styles={"primary":(C["primary"],C["primary"],C["white"]),
            "secondary":(C["pale"],C["secondary"],C["primary"]),
            "teal":(C["pale"],C["teal"],C["primary"]),
            "neutral":(C["light"],C["divider"],C["body"])}
    face,edge,color=styles[kind]
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.008,rounding_size=0.014",
                               linewidth=0.8,edgecolor=edge,facecolor=face))
    text(ax,x+w/2,y+h/2,label,fs,color=color,ha="center",va="center",linespacing=1.2)


def arrow(ax: Any,a: tuple[float,float],b: tuple[float,float],color: str|None=None,rad: float=0,dash: bool=False) -> None:
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=9.2,linewidth=1.0,
        color=color or C["muted"],linestyle="--" if dash else "-",
        connectionstyle=f"arc3,rad={rad}",shrinkA=2,shrinkB=2,zorder=1))


def canvas(size: tuple[float,float]) -> tuple[Any,Any]:
    fig,ax=plt.subplots(figsize=size)
    ax.set(xlim=(0,1),ylim=(0,1))
    ax.axis("off")
    return fig,ax


def save(name: str,fig: Any) -> None:
    FIG.mkdir(parents=True,exist_ok=True); SRC.mkdir(parents=True,exist_ok=True)
    fig.savefig(FIG/f"{name}.pdf",format="pdf",metadata={"Title":META[name][0],"Creator":"report/scripts/generate_figures.py"})
    svg_path = SRC/f"{name}.svg"
    fig.savefig(svg_path,format="svg",metadata={"Title":META[name][0]})
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text("\n".join(line.rstrip() for line in svg_text.splitlines()).rstrip()+"\n",encoding="utf-8")
    plt.close(fig)


def draw_01() -> None:
    d=read_json("01-overall-pipeline.json"); fig,ax=canvas((6.4,2.6))
    ns={n["id"]:n for n in d["nodes"]}
    for s,t in d["edges"]:
        a,b=ns[s],ns[t]; x,y=a["xy"]; w,h=a["wh"]; xx,yy=b["xy"]; ww,hh=b["wh"]
        dx=(xx+ww/2)-(x+w/2); dy=(yy+hh/2)-(y+h/2)
        scale_a=min((w/2)/max(abs(dx),1e-6),(h/2)/max(abs(dy),1e-6))
        scale_b=min((ww/2)/max(abs(dx),1e-6),(hh/2)/max(abs(dy),1e-6))
        p=(x+w/2+dx*scale_a,y+h/2+dy*scale_a)
        q=(xx+ww/2-dx*scale_b,yy+hh/2-dy*scale_b)
        arrow(ax,p,q,C["secondary"] if t=="specialists" else None)
    for n in d["nodes"]: box(ax,tuple(n["xy"]),tuple(n["wh"]),n["label"],n["kind"],8.2)
    text(ax,.02,.97,"训练研发主线",8.8,True,color=C["primary"],va="top")
    text(ax,.98,.97,"专项分支汇入共享基座",8.0,color=C["muted"],ha="right",va="top")
    fig.subplots_adjust(left=.01,right=.99,top=.95,bottom=.01)
    save("01-overall-pipeline",fig)


def draw_02() -> None:
    rows=read_csv("02-data-distribution.csv")
    em=[r for r in rows if r["group"]=="emotion"]; pr=[r for r in rows if r["group"]=="profile"]
    labels=[r["label"] for r in em]; counts=[int(r["count"]) for r in em]; y=list(range(len(em)))
    fig,(a,b)=plt.subplots(1,2,figsize=(6.4,2.75),gridspec_kw={"width_ratios":[1.7,1]})
    colors=[C["primary"] if x=="anxiety" else C["secondary"] if x=="relaxed" else C["gray"] for x in labels]
    a.barh(y,counts,color=colors,height=.68); a.set_yticks(y,labels,fontproperties=fp(8.0)); a.invert_yaxis()
    a.set_xlim(0,7500); a.xaxis.set_major_locator(MultipleLocator(2500)); a.grid(axis="x",color=C["divider"],lw=.5); a.set_axisbelow(True)
    a.set_title("原始 Train 情绪分布（n=18,657）",fontproperties=fp(8.2,True),color=C["primary"],pad=6)
    a.set_xlabel("Gold 标签数",fontproperties=fp(7.6)); a.tick_params(axis="x",labelsize=7,length=2)
    for i,v in enumerate(counts): a.text(v+80,i,f"{v:,}",va="center",fontproperties=fp(7.3))
    for s in ("top","right","left"): a.spines[s].set_visible(False)
    empty=next(int(r["count"]) for r in pr if "empty" in r["label"])
    nonempty=next(int(r["count"]) for r in pr if "non-empty" in r["label"])
    total=empty+nonempty
    b.set(xlim=(0,103),ylim=(-.6,.7)); b.barh(0,100*empty/total,color="#9AA7AF",height=.42)
    b.barh(0,100*nonempty/total,left=100*empty/total,color=C["secondary"],height=.42)
    b.text(50*empty/total,0,f"全空\n{empty:,}\n37.26%",ha="center",va="center",fontproperties=fp(7.8),linespacing=1.1)
    b.text(100*empty/total+50*nonempty/total,0,f"非空\n{nonempty:,}\n62.74%",ha="center",va="center",fontproperties=fp(7.8),color=C["white"],linespacing=1.1)
    b.set_title("有效 SFT Train 画像（n=16,192）",fontproperties=fp(8.2,True),color=C["primary"],pad=6)
    b.set_xticks([0,50,100],["0%","50%","100%"],fontproperties=fp(7.7)); b.set_yticks([]); b.grid(axis="x",color=C["white"],lw=.5)
    for s in ("top","right","left"): b.spines[s].set_visible(False)
    fig.subplots_adjust(left=.14,right=.96,top=.91,bottom=.18,wspace=.4)
    save("02-data-distribution",fig)


def draw_03() -> None:
    d=read_json("03-sft-dataflow.json"); fig,ax=canvas((6.4,2.7))
    text(ax,.02,.97,"按预测时刻展开历史，再执行 Completion-only SFT",8.8,True,color=C["primary"],va="top")
    messages=[
        ((.02,.66),(.20,.20),"t1 用户预测点\n这周考试压力很大。","secondary"),
        ((.25,.66),(.20,.20),"助手历史\n听起来最近要忙不少。","neutral"),
        ((.48,.66),(.20,.20),"t2 当前用户\n我今晚想先复习数学。","secondary"),
        ((.71,.66),(.25,.20),"Gold 回复目标\n可以先定一个小目标。","neutral"),
    ]
    for pos,size,label,kind in messages: box(ax,pos,size,label,kind,7.9)
    text(ax,.71,.60,"不进入 t2 Prompt",7.5,True,color=C["orange"],ha="center",va="top")
    arrow(ax,(.58,.65),(.155,.52),C["secondary"],rad=-.08)
    box(ax,(.02,.28),(.27,.23),"T1 历史选择\n保留截至 t2 的完整消息后缀\n从较早消息开始裁剪","neutral",7.6)
    arrow(ax,(.30,.395),(.35,.395),C["secondary"])
    box(ax,(.36,.28),(.31,.23),f"Prompt\nsystem + selected history\n+ assistant_prefix\n≤{d['prompt_budget']:,} Token · labels={d['prompt_label']}","secondary",7.5)
    arrow(ax,(.68,.395),(.72,.395),C["secondary"])
    box(ax,(.73,.28),(.25,.23),"Gold Completion + EOS\n参与 causal CE Loss","teal",7.7)
    text(ax,.02,.13,f"完整训练序列上限：{d['sequence_budget']:,} Token。示例由人工编写，不含个人信息。",7.5,color=C["muted"],va="center")
    fig.subplots_adjust(left=.01,right=.99,top=.96,bottom=.02)
    save("03-sft-dataflow",fig)


def draw_04() -> None:
    d=read_json("04-qlora-training.json"); fig,ax=canvas((6.4,2.55))
    box(ax,(.02,.43),(.18,.23),"Prompt + Gold\nCompletion + EOS","neutral",8.2)
    arrow(ax,(.21,.545),(.29,.545))
    box(ax,(.30,.38),(.38,.36),f"{d['base_model']} Base\n4-bit NF4 · BF16 compute\n基座权重冻结\nAttention / FFN 投影层","primary",8.5)
    box(ax,(.38,.15),(.24,.16),f"LoRA A/B\nr={d['lora_rank']} · α={d['lora_alpha']} · dropout={d['lora_dropout']}","teal",8.1)
    arrow(ax,(.50,.32),(.50,.37),C["teal"])
    arrow(ax,(.69,.545),(.77,.545))
    box(ax,(.78,.43),(.20,.23),"Completion logits\nCausal CE\n含 EOS","secondary",8.1)
    arrow(ax,(.84,.42),(.60,.32),C["orange"],rad=-.22,dash=True)
    text(ax,.03,.055,"仅更新 LoRA A/B；Base 与 LM Head 冻结\n目标模块："+", ".join(d["target_modules"]),7.1,color=C["muted"],va="center")
    fig.subplots_adjust(left=.01,right=.99,top=.97,bottom=.13)
    save("04-qlora-training",fig)


def draw_05() -> None:
    rows=read_csv("05-supervision-share.csv")
    names={"response_text":"回复正文","user_profile":"用户画像","emotion_label":"情绪标签",
           "public_json_structure":"公共 JSON 结构","boundary_mixed":"边界混合","eos":"EOS"}
    colors={"response_text":C["primary"],"user_profile":C["secondary"],"emotion_label":C["teal"],
            "public_json_structure":"#AAB5BC","boundary_mixed":"#C8D0D5","eos":"#75838C"}
    order=list(names); idx={r["region"]:r for r in rows}
    fig,ax=plt.subplots(figsize=(6.4,1.9)); left=0
    plot_order=["response_text","user_profile","emotion_label"]
    other=sum(float(idx[k]["share_percent"]) for k in order if k not in plot_order)
    plotted={k:float(idx[k]["share_percent"]) for k in plot_order}
    plotted["other_supervised"]=other
    colors["other_supervised"]="#AAB5BC"
    names["other_supervised"]="其他监督 Token"
    for k in plot_order+["other_supervised"]:
        share=plotted[k]
        ax.barh(.56,share,left=left,color=colors[k],height=.34,edgecolor=C["white"],lw=.5)
        left+=share
    ax.set(xlim=(0,102),ylim=(0,1)); ax.set_yticks([])
    ax.set_xticks([0,20,40,60,80,100],[f"{x}%" for x in [0,20,40,60,80,100]],fontproperties=fp(7.8))
    ax.set_xlabel("明确监督 Token 占比（正式 Train 16,192 条；总数 2,851,614）",fontproperties=fp(8.1))
    ax.grid(axis="x",color=C["divider"],lw=.5); ax.set_axisbelow(True)
    for s in ("top","right","left"): ax.spines[s].set_visible(False)
    handles=[Rectangle((0,0),1,1,facecolor=colors[k],label=f"{names[k]} {plotted[k]:.2f}%") for k in plot_order+["other_supervised"]]
    ax.legend(handles=handles,ncol=2,loc="upper center",bbox_to_anchor=(.5,.98),
              bbox_transform=fig.transFigure,frameon=False,prop=fp(7.6),
              columnspacing=1.5,handlelength=1,handletextpad=.5)
    fig.subplots_adjust(left=.06,right=.95,top=.59,bottom=.32)
    save("05-supervision-share",fig)


def draw_06() -> None:
    rows=read_csv("06-a8-learning-curve.csv"); fig,ax=plt.subplots(figsize=(6.4,2.5))
    styles={"Pilot160":(C["primary"],"o","-"),"Dev224":(C["secondary"],"s","--")}
    for cohort,(color,marker,line) in styles.items():
        subset=sorted([r for r in rows if r["cohort"]==cohort],key=lambda r:int(r["optimizer_step"]))
        xs=[int(r["optimizer_step"]) for r in subset]; ys=[float(r["micro_f1"]) for r in subset]
        ax.plot(xs,ys,color=color,marker=marker,linestyle=line,lw=1.6,ms=4,label=cohort)
        for i,(x,y) in enumerate(zip(xs,ys)):
            if i==len(xs)-1: ax.annotate(f"{y:.5f}",(x,y),xytext=(-5,8 if cohort=="Dev224" else -14),
               textcoords="offset points",ha="right",fontproperties=fp(7.5),color=color)
    ax.set(xlim=(450,4200),ylim=(0,.5))
    ax.set_xticks([512,1024,2048,4048],["512","1,024","2,048","4,048"],fontproperties=fp(7.4))
    ax.set_yticks([0,.1,.2,.3,.4,.5],[f"{x:.1f}" for x in [0,.1,.2,.3,.4,.5]],fontproperties=fp(7.2))
    ax.set_xlabel("累计 Optimizer Step",fontproperties=fp(8)); ax.set_ylabel("画像 33-label Micro-F1",fontproperties=fp(8))
    ax.grid(color=C["divider"],lw=.5); ax.set_axisbelow(True); ax.legend(frameon=False,ncol=2,loc="upper left",prop=fp(7.3))
    for s in ("top","right"): ax.spines[s].set_visible(False)
    fig.subplots_adjust(left=.11,right=.98,top=.94,bottom=.22)
    save("06-a8-learning-curve",fig)


def draw_07() -> None:
    rows=read_csv("07-emotion-replay.csv"); fig,ax=plt.subplots(figsize=(6.4,3.1))
    anxiety=next(r for r in rows if r["emotion"]=="anxiety")
    rows=[r for r in rows if r["emotion"]!="anxiety"]
    labels=[r["emotion"] for r in rows]; gold=[int(r["gold_count"]) for r in rows]
    old=[int(r["checkpoint_4665_hits"]) for r in rows]; new=[int(r["replay_1152_hits"]) for r in rows]
    x=list(range(len(labels))); w=.31
    a=ax.bar([i-w/2 for i in x],old,w,color=C["primary"],label="checkpoint-4665")
    b=ax.bar([i+w/2 for i in x],new,w,color=C["secondary"],label="Replay-1152")
    ax.set_ylim(0,15); ax.set_xticks(x,[f"{n}\nGold={g}" for n,g in zip(labels,gold)],fontproperties=fp(7.3))
    ax.set_ylabel("正确识别数（命中）",fontproperties=fp(7.9))
    ax.set_yticks([0,3,6,9,12]); ax.set_yticklabels(["0","3","6","9","12"],fontproperties=fp(7.2))
    ax.grid(axis="y",color=C["divider"],lw=.5); ax.set_axisbelow(True)
    ax.legend(frameon=False,loc="upper right",prop=fp(7.2))
    for bars,vals in ((a,old),(b,new)):
        for bar,v,n in zip(bars,vals,gold):
            ax.annotate(f"{v}/{n}",(bar.get_x()+bar.get_width()/2,bar.get_height()),xytext=(0,3),
                        textcoords="offset points",ha="center",fontproperties=fp(6.7))
    ax.text(.5,-.33,f"高频类单独标注：anxiety（Gold={anxiety['gold_count']}）命中 {anxiety['checkpoint_4665_hits']}→{anxiety['replay_1152_hits']}",
            transform=ax.transAxes,ha="center",va="top",fontproperties=fp(7.5),color=C["muted"])
    for s in ("top","right"): ax.spines[s].set_visible(False)
    fig.subplots_adjust(left=.10,right=.99,top=.93,bottom=.38)
    save("07-emotion-replay",fig)


def draw_08() -> None:
    d=read_json("08-dual-adapter-architecture.json"); fig,ax=canvas((6.4,2.85))
    text(ax,.02,.97,d["title"],8.6,True,color=C["primary"],va="top")
    box(ax,(.02,.42),(.12,.18),d["input"],"neutral",8.2)
    box(ax,(.17,.65),(.20,.18),d["profile_prompt"],"secondary",7.9)
    box(ax,(.17,.20),(.20,.18),d["emotion_prompt"],"secondary",7.5)
    box(ax,(.41,.37),(.22,.34),d["base"],"primary",8.0)
    box(ax,(.67,.65),(.14,.18),d["profile_parser"],"neutral",7.0)
    box(ax,(.67,.20),(.14,.18),d["emotion_parser"],"neutral",6.8)
    box(ax,(.85,.33),(.14,.29),d["assembler"],"primary",6.8)
    box(ax,(.85,.08),(.14,.18),d["output"],"neutral",6.8)
    arrow(ax,(.14,.54),(.17,.74),C["secondary"]); arrow(ax,(.14,.47),(.17,.29),C["secondary"])
    arrow(ax,(.37,.74),(.41,.63),C["teal"]); arrow(ax,(.37,.29),(.41,.44),C["teal"])
    arrow(ax,(.63,.63),(.67,.74),C["secondary"]); arrow(ax,(.63,.44),(.67,.29),C["secondary"])
    arrow(ax,(.81,.74),(.85,.55),C["secondary"]); arrow(ax,(.81,.29),(.85,.45),C["secondary"])
    arrow(ax,(.92,.33),(.92,.27),C["primary"])
    text(ax,.02,.06,d["file_note"],7.3,color=C["muted"],va="bottom")
    fig.subplots_adjust(left=.01,right=.99,top=.95,bottom=.02)
    save("08-dual-adapter-architecture",fig)


def draw_09() -> None:
    rows=read_csv("09-final-profile-compare.csv"); dims=[r for r in rows if r["dimension"]!="Overall Micro-F1"]
    total=next(r for r in rows if r["dimension"]=="Overall Micro-F1")
    fig,(a,b)=plt.subplots(1,2,figsize=(6.4,2.65),gridspec_kw={"width_ratios":[2.15,1]})
    names=[r["dimension"] for r in dims]; x=list(range(3)); w=.32
    av=[float(r["e01b_train_dev_4504"]) for r in dims]; bv=[float(r["a8_p4504"]) for r in dims]
    ba=a.bar([i-w/2 for i in x],av,w,color=C["primary"],label="E01-B Train+Dev 4504")
    bb=a.bar([i+w/2 for i in x],bv,w,color=C["secondary"],label="A8 P4504")
    a.set_ylim(0,.78); a.set_xticks(x,names,fontproperties=fp(7.4)); a.set_ylabel("Micro-F1",fontproperties=fp(8))
    a.set_yticks([0,.2,.4,.6],["0.0","0.2","0.4","0.6"],fontproperties=fp(7.1))
    a.grid(axis="y",color=C["divider"],lw=.5); a.set_axisbelow(True); a.legend(frameon=False,loc="upper left",prop=fp(6.6),handlelength=1)
    for bars,vals in ((ba,av),(bb,bv)):
        for bar,v in zip(bars,vals): a.annotate(f"{v:.4f}",(bar.get_x()+bar.get_width()/2,bar.get_height()),xytext=(0,3),textcoords="offset points",ha="center",fontproperties=fp(6.6))
    vals=[float(total["e01b_train_dev_4504"]),float(total["a8_p4504"])]
    bars=b.bar([0,1],vals,color=[C["primary"],C["secondary"]],width=.56)
    b.set_ylim(0,.78); b.set_xticks([0,1],["E01-B","A8"],fontproperties=fp(7.4))
    b.set_title("总体 Micro-F1",fontproperties=fp(8.1,True),color=C["primary"],pad=7)
    b.set_yticks([0,.2,.4,.6],["0.0","0.2","0.4","0.6"],fontproperties=fp(7.1)); b.grid(axis="y",color=C["divider"],lw=.5); b.set_axisbelow(True)
    for bar,v in zip(bars,vals): b.annotate(f"{v:.4f}",(bar.get_x()+bar.get_width()/2,bar.get_height()),xytext=(0,4),textcoords="offset points",ha="center",fontproperties=fp(7))
    for ax in (a,b):
        for s in ("top","right"): ax.spines[s].set_visible(False)
    fig.subplots_adjust(left=.10,right=.99,top=.87,bottom=.20,wspace=.30)
    save("09-final-profile-compare",fig)


def draw_10() -> None:
    rows=read_csv("10-deployment-performance.csv")
    v={r["metric"]:float(r["value"]) for r in rows if r["metric"] in ("P50","P90","P95")}
    gpu=int(next(r["value"] for r in rows if r["metric"]=="GPU peak"))
    schema=int(next(r["value"] for r in rows if r["metric"]=="Schema valid"))
    errors=int(next(r["value"] for r in rows if r["metric"]=="Unresolved errors"))
    fig,(a,b)=plt.subplots(1,2,figsize=(6.4,2.55),gridspec_kw={"width_ratios":[1.55,1]})
    names=["P50","P90","P95"]; vals=[v["P50"],v["P90"],v["P95"]]; yy=[2,1,0]
    a.barh(yy,vals,color=[C["primary"],C["secondary"],C["teal"]],height=.46)
    a.set_yticks(yy,names,fontproperties=fp(7.8,True)); a.set_xlim(0,18)
    a.set_xticks([0,5,10,15],[0,5,10,15],fontproperties=fp(7.1)); a.set_xlabel("逐样本端到端延迟（秒）",fontproperties=fp(7.8))
    a.grid(axis="x",color=C["divider"],lw=.5); a.set_axisbelow(True)
    for y,x in zip(yy,vals): a.text(x+.22,y,f"{x:.3f} s",va="center",fontproperties=fp(7.2))
    for s in ("top","right","left"): a.spines[s].set_visible(False)
    b.axis("off")
    text(b,.5,.91,"固定输入 Schema 合法率",8,True,color=C["primary"],ha="center",va="top")
    text(b,.5,.72,f"{schema}/100",17,True,color=C["primary"],ha="center",va="center")
    text(b,.5,.57,f"未解决错误：{errors}",8,color=C["muted"],ha="center",va="center")
    text(b,.5,.36,f"GPU 峰值 {gpu:,} / 8,188 MiB",8,True,ha="center",va="center")
    b.add_patch(FancyBboxPatch((.10,.19),.80,.09,boxstyle="round,pad=.003,rounding_size=.02",
        facecolor=C["palegray"],edgecolor="none",transform=b.transAxes))
    b.add_patch(FancyBboxPatch((.10,.19),.80*gpu/8188,.09,boxstyle="round,pad=.003,rounding_size=.02",
        facecolor=C["secondary"],edgecolor="none",transform=b.transAxes))
    text(b,.5,.08,"Docker Run 03 · n=100 · 本地验收",7.1,color=C["muted"],ha="center",va="center")
    fig.subplots_adjust(left=.10,right=.99,top=.91,bottom=.20,wspace=.3)
    save("10-deployment-performance",fig)


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()


def write_manifest() -> None:
    out=["# Figure Manifest","","Generated by `scripts/generate_figures.py` from the aggregate inputs in `figures/data/`. No raw dialogue, model weights, or prediction JSONL are included.","",
         "| Figure | PDF | Data input | Source document / run | Cohort / n | Model / checkpoint | Metric | Value / unit | Script | PDF SHA-256 | Verification |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name,m in META.items():
        title,data,source,cohort,n,model,metric,value,unit=m
        out.append(f"| {name[:2]} {title} | `figures/{name}.pdf` | `{data}` | {source} | {cohort}；{n} | {model} | {metric} | {value}；{unit} | `scripts/generate_figures.py` | `{sha(FIG/f'{name}.pdf')}` | Generated; source checked |")
    out += ["","## Notes","",
      "- 02 uses raw Train emotion counts and SFT Train profile sparsity with separate denominators.",
      "- 06 draws the two fixed cohorts independently and connects only the four observed optimizer steps.",
      "- 07 shows correct hits with the Gold support for the same 209-point evaluation slice.",
      "- 09 compares the same 2,354 Val IDs; native prompts differ, so it is not a single-factor ablation.",
      "- 10 P90 is the nearest-rank 90th value from the 100 stored latencies; P50 and P95 are the recorded values.",
      "- The sample dialogue in 03 is authored for illustration and contains no personal data.",""]
    MANIFEST.write_text("\n".join(out),encoding="utf-8")


def main() -> None:
    configure()
    for fn in (draw_01,draw_02,draw_03,draw_04,draw_05,draw_06,draw_07,draw_08,draw_09,draw_10): fn()
    write_manifest()
    print(f"Generated 10 vector PDFs and SVG sources in {FIG}")
    print(f"Manifest: {MANIFEST}")


if __name__=="__main__":
    main()
