"""Render a one-page Chinese explanation of the implemented A/B KV designs."""
from io import BytesIO
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
try:
    import fitz
except ImportError:
    sys.path.insert(0, str(ROOT/'runs/setup/pdf-tools'))
    import fitz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.mathtext import MathTextParser

OUTPUT = ROOT/'docs/plan_feishu_concise.pdf'
INK = (0.09, 0.16, 0.23)
MUTED = (0.36, 0.43, 0.49)
TEAL = (0.00, 0.45, 0.49)
ORANGE = (0.77, 0.30, 0.14)
LINE = (0.82, 0.87, 0.89)
doc = fitz.open()
page = doc.new_page(width=595.28, height=841.89)
font = fitz.Font('cjk')
page.insert_font(fontname='cn', fontbuffer=font.buffer)


def text(rect, value, size=10, color=INK, align=0):
    remaining = page.insert_textbox(fitz.Rect(rect), value, fontname='cn', fontsize=size,
                                    color=color, lineheight=1.35, align=align)
    if remaining < 0:
        raise RuntimeError(f'Text overflow ({remaining}): {value}')


def box(rect, fill):
    page.draw_rect(fitz.Rect(rect), color=LINE, fill=fill, width=.7)


def arrow(x1, y, x2):
    page.draw_line((x1, y), (x2, y), color=MUTED, width=1.2)
    page.draw_line((x2-4, y-3), (x2, y), color=MUTED, width=1.2)
    page.draw_line((x2-4, y+3), (x2, y), color=MUTED, width=1.2)


def formula(rect, value, size=14):
    rect = fitz.Rect(rect)
    measured = MathTextParser('path').parse(value, dpi=72, prop=FontProperties(size=size))
    size *= min(1, (rect.width-8)/measured.width)
    fig = plt.figure(figsize=(rect.width/72, rect.height/72))
    fig.text(.5, .5, value, fontsize=size, ha='center', va='center', color=INK)
    stream = BytesIO()
    fig.savefig(stream, format='pdf', transparent=True, bbox_inches='tight', pad_inches=.02)
    plt.close(fig)
    equation = fitz.open(stream=stream.getvalue(), filetype='pdf')
    page.show_pdf_page(rect, equation, 0)
    equation.close()


page.draw_rect(fitz.Rect(0, 0, 595.28, 7), color=TEAL, fill=TEAL)
text((38, 27, 558, 65), 'A / B 两种共享 K/V 方案', 23)
text((38, 73, 558, 98), '同一 YOCO 基座，只改变“前六层的信息如何融合”。', 11, MUTED)
text((38, 110, 558, 135), '共同框架：融合一次，后六层共享', 14, TEAL)

box((38, 149, 239, 190), (0.95, .97, .98))
text((46, 159, 231, 183), '前6层：H1 → H2 → … → H6', 10.5, align=1)
text((38, 198, 239, 230), '各层经同一组归一化、K/V投影，\n得到六对候选记忆。', 9.4, MUTED, 1)
arrow(246, 169, 267)
box((275, 149, 401, 190), (.91, .97, .97))
text((281, 159, 395, 183), '按 A 或 B 加权融合', 10.4, TEAL, 1)
text((269, 198, 407, 230), '进入第7层前只做一次。', 9.4, MUTED, 1)
arrow(408, 169, 427)
box((435, 149, 558, 190), (.95, .97, .98))
text((440, 159, 553, 183), '共享 K* / V*', 11, align=1)
text((435, 198, 558, 230), '第7–12层各自生成Q，\n复用同一份记忆。', 9.4, MUTED, 1)
formula((38, 238, 558, 277), r'$K_t^*=\sum_{j=1}^{6}\alpha_{t,j}K_{t,j},\qquad V_t^*=\sum_{j=1}^{6}\alpha_{t,j}V_{t,j}$', 15)
text((38, 282, 558, 307), 'K 和 V 共用一组权重；softmax 在六个来源层之间进行，权重之和为 1。', 10, MUTED)

box((38, 326, 289, 614), (.95, .985, .983))
box((307, 326, 558, 614), (1, .968, .947))
text((52, 341, 277, 369), 'A｜全局可学习配比', 14.5, TEAL)
text((321, 341, 546, 369), 'B｜按 token 内容分配', 14.5, ORANGE)
text((52, 376, 277, 401), '所有 token 共用一套层权重。', 10.4)
text((321, 376, 546, 401), '不同 token 可以有不同层权重。', 10.4)
formula((50, 413, 277, 458), r'$\alpha_j=\mathrm{softmax}_j(b)$', 15)
formula((317, 407, 550, 448), r'$s_{t,j}=\frac{w^\top\mathrm{RMSNorm}(H_{t,j})}{\sqrt{d}}+b_j$', 12)
formula((321, 452, 546, 483), r'$\alpha_{t,j}=\mathrm{softmax}_j(s_{t,\cdot})$', 13)
text((52, 480, 277, 547), '学习六个全局参数 b。\n训练时更新配比；配比本身不随当前 token 的输入改变。', 10.2)
text((321, 490, 546, 551), 'w 是共享打分向量，b 是层偏置。\nRMSNorm 无额外可学习参数。\n按当前 token 的各层表示生成配比。', 9.8)
text((52, 565, 277, 591), '新增参数：6', 12, TEAL)
text((321, 565, 546, 591), '新增参数：d + 6 = 774', 12, ORANGE)

text((38, 638, 558, 664), '共同约定', 13, TEAL)
text((38, 674, 558, 717), '初始化：A 的 b=0；B 的 w=0、b=0，两组都从六层等权（1/6）起步。\nK 的位置旋转在融合后执行；因果 mask 保留，Q 主路径从第6层输出继续。', 10)
text((38, 726, 558, 751), '符号：t 为 token 位置，j 为来源层（1–6），d=768。Baseline 仅从 H6 生成 K/V。', 9.3, MUTED)
page.draw_line((38, 766), (558, 766), color=LINE, width=.7)
text((38, 778, 558, 817), '一句话：A 学“全局配比”；B 学“按 token 内容算配比”。', 12.4, TEAL)

doc.set_metadata({'title': 'A/B 共享 K/V 设计｜一页精简版', 'author': 'YOCO experiment',
                  'subject': '基于仓库已核对实现整理：共同流程、两套权重公式与核心区别',
                  'creator': 'render_plan_feishu_concise.py'})
doc.subset_fonts()
doc.save(OUTPUT, garbage=4, deflate=True)
doc.close()
check = fitz.open(OUTPUT)
assert len(check) == 1
content = check[0].get_text()
for word in ('全局可学习配比', '按 token 内容分配', '新增参数', '774', '共同框架'):
    assert word in content, word
preview = ROOT/'runs/setup/plan-feishu-concise-preview.png'
check[0].get_pixmap(matrix=fitz.Matrix(1.7, 1.7)).save(preview)
print(f'Created one-page PDF: {OUTPUT}')
print(f'Preview: {preview}')
