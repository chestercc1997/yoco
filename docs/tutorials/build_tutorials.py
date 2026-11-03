"""Build offline tutorial HTML and original SVG teaching diagrams."""
from pathlib import Path
import html,json,hashlib,re
import markdown
R=Path(__file__).resolve().parent
A=R/'assets';A.mkdir(exist_ok=True)

def diagram(name,title,nodes,edges,notes,height=490):
    s=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1100 {height}" font-family="Noto Sans CJK SC, sans-serif" role="img" aria-label="{html.escape(title)}">', '<defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#64748b"/></marker></defs>', '<rect width="1100" height="100%" rx="18" fill="#f4f7fb"/>',f'<text x="36" y="42" font-size="25" font-weight="700" fill="#132b46">{title}</text>']
    colors={'blue':('#e4efff','#285ea9'),'green':('#def5eb','#247653'),'orange':('#fff0d9','#9a6513'),'purple':('#eee6ff','#6943a0')}
    for i,j in edges:
        x,y,w,h,_,_=nodes[i];u,v,a,b,_,_=nodes[j]
        if abs(y-v)<40:
            p=f'M{x+w},{y+h/2} L{u-8},{v+b/2}'
        else:
            p=f'M{x+w/2},{y+h} L{u+a/2},{v-8}'
        s.append(f'<path d="{p}" stroke="#64748b" fill="none" stroke-width="2.5" marker-end="url(#arrow)"/>')
    for x,y,w,h,lines,color in nodes:
        bg,fg=colors[color]
        s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{bg}" stroke="{fg}"/>')
        for k,t in enumerate(lines):
            ty=y+h/2+(k-(len(lines)-1)/2)*26+7
            size=min(19,(w-24)/max(1,sum(1 if ord(c)>127 else .57 for c in t)))
            s.append(f'<text x="{x+w/2}" y="{ty}" text-anchor="middle" font-size="{size:.1f}" fill="{fg}">{html.escape(t)}</text>')
    for k,t in enumerate(notes):s.append(f'<text x="36" y="{height-50+k*25}" font-size="17" fill="#475569">{html.escape(t)}</text>')
    s.append('</svg>');(A/name).write_text('\n'.join(s))

diagram('memory_sources.svg','全局记忆从哪里来？（抽象依赖图）',[
(35,85,220,75,['普通 Transformer','本层隐藏状态 H'], 'blue'),(310,85,220,75,['本层独立 KV','每层一组历史'], 'green'),(595,85,450,75,['深层历史 KV','依赖前面所有层的历史计算'],'orange'),
(35,205,220,75,['YOCO','self-decoder → E'],'blue'),(310,205,220,75,['共享 K* / V*','一组全局记忆'],'green'),(595,205,450,75,['多个 cross-decoder 层','每层独立 Q，反复读取同一记忆'],'purple'),
(35,325,220,75,['CED','causal encoder → E'],'blue'),(310,325,220,75,['投影 / 压缩 / 分组','全局记忆'],'green'),(595,325,450,75,['decoder：全局读取 + 本层 SWA','局部状态仍有逐层依赖'],'purple')],[(0,1),(1,2),(3,4),(4,5),(6,7),(7,8)],['同一组 KV ≠ 同一组 attention 权重；不同 Q 可以得到不同输出。'],490)
diagram('phases.svg','YOCO：哪些位置需要经过后半部分？',[
(40,90,270,95,['训练','全部位置经过 self-decoder'],'blue'),(410,90,270,95,['全部位置经过 cross-decoder','所有位置参与 loss'],'purple'),
(40,225,270,95,['Prefill','全部 prompt 建立共享 KV'],'green'),(410,225,270,95,['末位置经过 cross-decoder','取得首个新 token 的 logits'],'purple'),
(40,360,270,95,['Decode','新 token 经过 self-decoder'],'blue'),(410,360,270,95,['新 token 经过 cross-decoder','更新缓存并预测下一 token'],'purple'),(790,210,260,125,['成立条件','后半部分历史交互','只依赖共享记忆'],'orange')],[(0,1),(2,3),(4,5)],['图示为典型 YOCO 的依赖分析；CED 的逐层局部状态需额外处理。'],540)
diagram('replay.svg','局部路径的依赖范围会随深度扩展',[
(35,90,270,90,['第 1 层：W = 4','位置 10 可依赖 7—10'],'blue'),(415,90,270,90,['第 2 层','位置 10 可间接依赖 4—10'],'blue'),(795,90,270,90,['第 3 层','位置 10 可间接依赖 1—10'],'blue'),
(125,265,360,100,['只重放最后 4 个位置','保留输入 7—10'],'orange'),(615,265,360,100,['左边界的历史被截断','恢复状态可能近似而非精确'],'purple')],[(0,1),(1,2),(3,4)],['纯 SWA 教学示例：理论跨度 1 + D(W−1)；不表示加入全局分支后的总感受野。'],450)
diagram('v41_overview.svg','V4.1：主路径与辅助模块的分工',[
(35,85,225,80,['文字 / 图片','embedding / ViT + projector'],'blue'),(320,85,210,80,['causal encoder','20 层'],'blue'),(590,85,210,80,['全局记忆','投影与共享'],'green'),(860,85,205,80,['decoder','20 层 → logits'],'purple'),
(35,240,225,110,['Engram','选定层条件查表','增强模式记忆'],'orange'),(320,240,210,110,['每层 MoE','稀疏专家前馈','独立于 KV 选择'],'orange'),(590,240,210,110,['CSA2 + SWA','全局稀疏读取','局部逐层加工'],'green'),(860,240,205,110,['DSpark','提出草稿','目标模型验证'],'purple')],[(0,1),(1,2),(2,3)],['mHC 负责层间残差流混合；FP4 与缓存管理负责状态存储和迁移。','下排是职责注释，不是串行执行顺序；全局记忆与 decoder 局部路径需结合阅读。'],450)
diagram('csa_modes.svg','CSA2：三个模式都要重新计算本层输出',[
(35,95,310,200,['Full','生成主 KV 与 indexer K','本层 indexer 选择位置','本层 Q + SWA → 输出'],'green'),(395,95,310,200,['Reindex','复用主 KV 与 indexer K','重新选择位置','本层 Q + SWA → 输出'],'blue'),(755,95,310,200,['Reuse','复用主 KV 与 indexer K','复用已选位置','本层 Q + SWA → 输出'],'purple')],[],['复用 KV：减少缓存；复用 indices：减少索引计算。','图中列的是模式职责，不表示发布模型每三层固定按此顺序排列。'],390)
diagram('hierarchy.svg','候选池不是最终 attention 集合',[
(35,100,270,100,['所有因果可见条目','第一次 indexer 扫描'],'blue'),(415,100,270,100,['共享候选池','2048 块 × 8 = 16384'],'green'),(795,100,270,100,['后续 Reindex','各自选择 Top-512'],'purple'),(415,280,270,90,['第一次层自己的 Top-512','直接用于本层 attention'],'orange')],[(0,1),(1,2),(0,3)],['后面的搜索范围缩小，但初次全范围扫描仍存在；各层最终选择可以不同。'],440)
diagram('cache_tiers.svg','状态的访问频率与保存期限决定放在哪里',[
(35,105,310,140,['GPU HBM','正在计算的全局 / 局部状态','高带宽，容量有限'],'blue'),(395,105,310,140,['主机内存','短期会话状态 / 中间层缓存','加载需要传输'],'green'),(755,105,310,140,['SSD / 持久前缀','长期复用的全局记忆','命中后仍需加载'],'purple'),(285,310,530,80,['局部状态缺失 → replay 恢复','恢复成本、数值差异和质量要单独评估'],'orange')],[(0,1),(1,2)],['这是缓存层次的教学示意，不表示每个请求都必须顺序经过全部层次。'],480)

css='''body{margin:0;background:#f6f8fb;color:#243448;font:17px/1.85 system-ui,"Noto Sans CJK SC",sans-serif}header{background:#132b46;color:white;padding:48px max(24px,calc((100vw - 1000px)/2))}header p{color:#cfdbed}main{max-width:1000px;margin:30px auto;padding:0 24px 80px}h1{font-size:34px;line-height:1.4}h2{margin-top:64px;border-top:1px solid #ccd7e5;padding-top:28px;color:#143f6b}h3{margin-top:36px}a{color:#1768aa}table{border-collapse:collapse;width:100%;font-size:15px;display:block;overflow:auto}th,td{border:1px solid #d1dbe7;padding:12px;min-width:110px}th{background:#e8eff8}pre{overflow:auto;background:#162a40;color:#e5effc;padding:22px;border-radius:12px;font-size:14px;line-height:1.65}code{font-family:ui-monospace,monospace}svg{width:100%;height:auto;margin:20px 0}blockquote,.lab{background:#eaf2fc;border-left:5px solid #3386bd;padding:18px 24px;margin:28px 0}.toc{background:white;border:1px solid #d4ddea;padding:25px;border-radius:12px}.toc ul{margin:4px 0}.lab input{font-size:16px;width:95px;margin:8px;padding:5px}.lab output{display:block;font-weight:bold;color:#164d77}.bar{height:26px;background:#367abb;margin:8px 0;border-radius:5px;min-width:2px}.shared{background:#2a9971}button{font-size:16px;padding:8px 18px} @media print{header{background:white;color:black;padding:0}.lab,nav{display:none}h2{break-before:auto}pre,svg{break-inside:avoid}}'''
lab='''<section class="lab"><h2>动手调整：缓存与 prefill 的教学算例</h2><p>普通双数组 GQA 模型；独立全局组用相同头维度。结果只计有效 KV 载荷，不含局部状态、索引器、权重和元数据，不能直接当作 V4.1 显存。</p><label>长度 T <input id="t" type="number" value="32768" min="1" max="1048576"></label><label>层数 L <input id="l" type="number" value="32" min="1" max="128"></label><label>共享组 G <input id="g" type="number" value="1" min="1" max="128"></label><p>固定 B=1、KV heads=8、head dim=128、2 bytes/元素。</p><output id="memory"></output><div id="standard" class="bar"></div><div id="shared" class="bar shared"></div><label>Replay 长度 R <input id="r" type="number" value="128" min="1" max="1048576"></label><output id="cost"></output><p>假设前后两部分层数相等、每 token-layer 成本相同，且 R≤T；只是计算量近似，没有通信和硬件计时。</p></section>'''
js='''function update(){let T=Number(document.getElementById('t').value),L=Number(document.getElementById('l').value),G=Number(document.getElementById('g').value),R=Number(document.getElementById('r').value);if(![T,L,G,R].every(x=>Number.isInteger(x)&&x>0)){document.getElementById('memory').textContent='请输入正整数';return}let m=2*T*8*128*2/1048576;document.getElementById('memory').textContent=`逐层 KV：${(m*L).toFixed(2)} MiB；共享全局 KV：${(m*G).toFixed(2)} MiB`;let z=Math.max(L,G);document.getElementById('standard').style.width=(100*L/z)+'%';document.getElementById('shared').style.width=(100*G/z)+'%';document.getElementById('cost').textContent=R>T?'R 大于 T：此简化算例不适用':`简化 prefill 计算比：(T+R)/(2T) = ${((T+R)/(2*T)).toFixed(4)}`;}document.querySelectorAll('.lab input').forEach(x=>x.addEventListener('input',update));update();'''
for name in ['DEEPSEEK_V41_TUTORIAL','CED_YOCO_TUTORIAL']:
    md=(R/(name+'.md')).read_text()
    title=md.splitlines()[0][2:]
    body=markdown.markdown('[TOC]\n'+md,extensions=['tables','fenced_code','toc'])
    for p in A.glob('*.svg'):
        body=re.sub(r'<img alt="[^"]*" src="assets/'+re.escape(p.name)+r'"\s*/?>',lambda m:p.read_text(),body)
    body=body.replace('DEEPSEEK_V41_TUTORIAL.md','DEEPSEEK_V41_TUTORIAL.html').replace('CED_YOCO_TUTORIAL.md','CED_YOCO_TUTORIAL.html')
    output=f'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(title)}</title><style>{css}</style><header><div>架构学习笔记 · 2026-10-30</div><h1>{html.escape(title)}</h1><p>从问题出发，沿计算依赖阅读架构。原创示意图、手算与可交互练习。</p><nav><a style="color:#cfe5ff" href="DEEPSEEK_V41_TUTORIAL.html">V4.1 全景</a> · <a style="color:#cfe5ff" href="CED_YOCO_TUTORIAL.html">CED / YOCO 专题</a></nav></header><main>{body}{lab}</main><script>{js}</script></html>'
    (R/(name+'.html')).write_text(output)
manifest=[]
urls={'deepseek-model-1017.py':'https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/raw/main/inference/model.py','deepseek-config-1017.json':'https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/raw/main/config.json'}
for name,url in urls.items():
    p=R/'sources'/name
    manifest.append(dict(file=name,source=url,download_host='hf-mirror.com',retrieved='2026-10-30',sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size,revision_note='main snapshot; upstream commit not pinned; local bytes identified by SHA256'))
(R/'sources/manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Built 2 offline HTML tutorials and 7 SVG diagrams')
