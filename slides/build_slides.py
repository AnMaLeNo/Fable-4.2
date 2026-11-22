"""Builds the 6-minute presentation of team Fable 4.2 as Fable4.2_slides.pptx and Fable4.2_slides.pdf (same layout, two writers).

    pip install python-pptx reportlab
    python build_slides.py

All the numbers shown on the slides are in the block "NUMBERS" below. Before publication: update the lines marked
TODO-FINAL, set FINAL = True (this removes the visible "TODO-FINAL" reminders from the slides) and run the script again.
"""
import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR, MSO_AUTO_SIZE
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

HERE = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------------------------------------- NUMBERS
FINAL = False                                   # TODO-FINAL: set to True once the numbers below are final
AS_OF = "Public leaderboard, 22 Nov, 07:25"     # TODO-FINAL: time of the final reading
T1_SCORE = "0.7705"                             # TODO-FINAL: task 1 public mIoU
T2_SCORE = "1.00"                               # TODO-FINAL: task 2 public score
T3_SCORE = "0.4038"                             # TODO-FINAL: task 3 public score
T1_LADDER = [                                   # (step, public mIoU, gain)   TODO-FINAL: add or replace the last row
    ("All-zeros baseline", "0.2553", ""),
    ("Segmenters on synthetic composites", "0.6608", "+0.406"),
    ("Per-method ensembles, zoom rescue", "0.7171", "+0.056"),
    ("Zoom-trained models, routed by type", "0.7292", "+0.012"),
    ("Rectangles located by decoding", "0.7418", "+0.013"),
    ("Object predictions snapped on SAM masks", "0.7474", "+0.006"),
    ("Longer training, same recipe", "0.7614", "+0.014"),
    ("Denser SAM proposals, full-type shift", "0.7705", "+0.009"),
]
T2_LADDER = [                                   # (step, public score)
    ("Families only", 0.10),
    ("Persona + watermark endpoints merged", 0.40),
    ("Merge only identified disguises", 0.75),
    ("Tests against local replicas", 0.85),
    ("One more endpoint, by submission test", 1.00),
]
T3_LADDER = [                                   # (step, public score)   TODO-FINAL: add or replace the last row
    ("Targets from augmentations", 0.1543),
    ("Out-of-fold probes (P)", 0.2328),
    ("+ img2img variations (I)", 0.2907),
    ("+ students on generated images (S)", 0.3509),
    ("+ reset-and-redistil replicas (R)", 0.3641),
    ("Near-identity training views", 0.3724),
    ("Confidence cap on easy images", 0.4038),
]
# ------------------------------------------------------------------------------------------------------------- STYLE
W, H = 13.333, 7.5
INK, PAPER, TEXT, MUTED, TINT, RULE = "1B1E2B", "FFFFFF", "1B1E2B", "5A6070", "F0F2F5", "D5D9E0"
C1, C2, C3, GOLD, SOFT = "1D8A7A", "B86E0E", "C2415A", "F2B33D", "C9CEDA"
TASKC = {1: C1, 2: C2, 3: C3}
LEAD = 1.17
TW = 11.3                                        # title width (leaves room for the task badge)
FONT_DIR = "/usr/share/fonts/truetype/liberation"
try:
    pdfmetrics.registerFont(TTFont("Sans", f"{FONT_DIR}/LiberationSans-Regular.ttf"))
    pdfmetrics.registerFont(TTFont("Sans-Bold", f"{FONT_DIR}/LiberationSans-Bold.ttf"))
    PF, PFB = "Sans", "Sans-Bold"
except Exception:                                # metric-compatible fallback
    PF, PFB = "Helvetica", "Helvetica-Bold"


def wrap(s, size, bold, w_in):
    """word-wrap s (may contain newlines) to a width in inches; returns a list of (line, ends_paragraph)"""
    font = PFB if bold else PF; out = []
    for para in s.split("\n"):
        cur = ""
        for word in para.split(" "):
            t = word if not cur else cur + " " + word
            if pdfmetrics.stringWidth(t, font, size) <= w_in * 72 or not cur: cur = t
            else: out.append((cur, False)); cur = word
        out.append((cur, True))
    return out


def text_h(s, size, bold, w_in, para_gap=0):
    ls = wrap(s, size, bold, w_in)
    return (len(ls) * size * LEAD + (sum(e for _, e in ls) - 1) * para_gap) / 72


class Deck:
    def __init__(self):
        self.prs = Presentation(); self.prs.slide_width = Inches(W); self.prs.slide_height = Inches(H)
        self.pdf = rl_canvas.Canvas(os.path.join(HERE, "Fable4.2_slides.pdf"), pagesize=(W * 72, H * 72))
        self.pdf.setTitle("Fable 4.2 - European AI Championship, Paris 2026"); self.pdf.setAuthor("Fable 4.2"); self.pdf.setCreator("build_slides.py")
        cp = self.prs.core_properties; cp.author = "Fable 4.2"; cp.last_modified_by = "Fable 4.2"; cp.title = "Fable 4.2 - European AI Championship, Paris 2026"
        cp.comments = ""; cp.keywords = ""; cp.subject = ""; cp.category = ""
        self.n = 0; self.s = None

    def slide(self, title, dark=False, title_size=30, title_y=0.5, title_color=None):
        if self.n: self.pdf.showPage()
        self.n += 1; self.s = self.prs.slides.add_slide(self.prs.slide_layouts[5])
        for ph in list(self.s.placeholders):
            if ph.placeholder_format.type != 1 and ph.placeholder_format.idx != 0: ph._element.getparent().remove(ph._element)
        bg = self.s.background.fill; bg.solid(); bg.fore_color.rgb = RGBColor.from_string(INK if dark else PAPER)
        self.pdf.setFillColor("#" + (INK if dark else PAPER)); self.pdf.rect(0, 0, W * 72, H * 72, stroke=0, fill=1)
        col = title_color or (PAPER if dark else TEXT); t = self.s.shapes.title
        t.left, t.top, t.width, t.height = Inches(0.6), Inches(title_y), Inches(TW), Inches(title_size * LEAD / 72 + 0.05)
        self._fill_tf(t.text_frame, title, title_size, True, col, "l", "t", 0)
        self._pdf_text(0.6, title_y, TW, title, title_size, True, col, "l", 0)
        if text_h(title, title_size, True, TW) > title_size * LEAD / 72 + 0.01: print(f"!! slide {self.n}: title wraps")
        self.dark = dark
        return self

    # -- shapes
    def rect(self, x, y, w, h, fill, rounded=0.0, name=None):
        sh = self.s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
        if rounded: sh.adjustments[0] = rounded
        sh.fill.solid(); sh.fill.fore_color.rgb = RGBColor.from_string(fill); sh.line.fill.background(); sh.shadow.inherit = False
        if name: sh.name = name
        self.pdf.setFillColor("#" + fill)
        if rounded: self.pdf.roundRect(x * 72, (H - y - h) * 72, w * 72, h * 72, rounded * min(w, h) * 72, stroke=0, fill=1)
        else: self.pdf.rect(x * 72, (H - y - h) * 72, w * 72, h * 72, stroke=0, fill=1)

    def oval(self, x, y, d, fill):
        sh = self.s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(d), Inches(d))
        sh.fill.solid(); sh.fill.fore_color.rgb = RGBColor.from_string(fill); sh.line.fill.background(); sh.shadow.inherit = False
        self.pdf.setFillColor("#" + fill); self.pdf.circle((x + d / 2) * 72, (H - y - d / 2) * 72, d * 36, stroke=0, fill=1)

    # -- text
    def _fill_tf(self, tf, s, size, bold, color, align, valign, para_gap):
        tf.word_wrap = True; tf.auto_size = MSO_AUTO_SIZE.NONE
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        tf.vertical_anchor = {"t": MSO_ANCHOR.TOP, "m": MSO_ANCHOR.MIDDLE, "b": MSO_ANCHOR.BOTTOM}[valign]
        for i, line in enumerate(s.split("\n")):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            for r in list(p.runs): r._r.getparent().remove(r._r)
            p.alignment = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}[align]
            if para_gap: p.space_after = Pt(para_gap)
            r = p.add_run(); r.text = line; f = r.font; f.name = "Arial"; f.size = Pt(size); f.bold = bold
            f.color.rgb = RGBColor.from_string(color)

    def _pdf_text(self, x, y, w, s, size, bold, color, align, para_gap, h=None, valign="t"):
        ls = wrap(s, size, bold, w); th = text_h(s, size, bold, w, para_gap)
        if h is not None and valign == "m": y = y + (h - th) / 2
        if h is not None and valign == "b": y = y + (h - th)
        self.pdf.setFillColor("#" + color); self.pdf.setFont(PFB if bold else PF, size); yy = H * 72 - y * 72 - size * 0.93
        for line, end in ls:
            if align == "l": self.pdf.drawString(x * 72, yy, line)
            elif align == "c": self.pdf.drawCentredString((x + w / 2) * 72, yy, line)
            else: self.pdf.drawRightString((x + w) * 72, yy, line)
            yy -= size * LEAD + (para_gap if end else 0)
        return th

    def text(self, x, y, w, h, s, size=15, bold=False, color=None, align="l", valign="t", para_gap=0):
        color = color or (PAPER if self.dark else TEXT)
        tb = self.s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        self._fill_tf(tb.text_frame, s, size, bold, color, align, valign, para_gap)
        th = self._pdf_text(x, y, w, s, size, bold, color, align, para_gap, h, valign)
        if th > h + 0.02: print(f"!! slide {self.n}: text overflows its box by {th - h:.2f} in: {s[:50]!r}")

    def items(self, x, y, w, lines, size=15, color=None, dot=None, gap=0.16):
        """list of short statements, each with a round marker; returns the y after the last item"""
        for s in lines:
            h = text_h(s, size, False, w - 0.3)
            self.oval(x, y + size * LEAD / 144 - 0.05, 0.1, dot or GOLD)
            self.text(x + 0.3, y, w - 0.3, h + 0.02, s, size, False, color); y += h + gap
        return y

    def badge(self, x, y, n, d=0.62):
        self.oval(x, y, d, TASKC[n]); self.text(x, y, d, d, str(n), 22, True, PAPER, "c", "m")

    def footer(self, note=""):
        col = SOFT if self.dark else MUTED
        self.text(0.6, H - 0.48, 5.0, 0.24, "Fable 4.2 · European AI Championship, Paris 2026", 11, False, col)
        self.text(W - 1.6, H - 0.48, 1.0, 0.24, str(self.n), 11, False, col, "r")
        if note:
            if not FINAL: note += "  ·  TODO-FINAL: refresh before noon"
            self.text(5.6, H - 0.48, W - 7.3, 0.24, note, 11, not FINAL, (GOLD if self.dark else C3) if not FINAL else col, "r")

    def table(self, x, y, colw, rows, row_h=0.46, size=14, head=True, aligns=None, hi=None, head_fill=INK):
        aligns = aligns or ["l"] * len(colw)
        for i, row in enumerate(rows):
            is_head = head and i == 0; yy = y + i * row_h
            fill = head_fill if is_head else (TINT if i % 2 == 0 else PAPER)
            if hi is not None and i == hi: fill = "FCEFCF"
            self.rect(x, yy, sum(colw), row_h, fill); xx = x
            for j, cell in enumerate(row):
                self.text(xx + 0.14, yy, colw[j] - 0.28, row_h, cell, size, is_head or (hi is not None and i == hi),
                          PAPER if is_head else TEXT, aligns[j], "m"); xx += colw[j]

    def bars(self, x, y, w, rows, color, label_w=4.0, row_h=0.5, size=14, vmax=1.0, fmt="{:.2f}"):
        bw = w - label_w - 0.95
        for i, (lab, v) in enumerate(rows):
            yy = y + i * row_h; last = i == len(rows) - 1
            self.text(x, yy, label_w - 0.15, row_h, lab, size, last, TEXT, "l", "m")
            self.rect(x + label_w, yy + 0.09, max(0.03, bw * v / vmax), row_h - 0.18, color if last else SOFT)
            self.text(x + label_w + bw * v / vmax + 0.1, yy, 0.85, row_h, fmt.format(v), size, True, color if last else TEXT, "l", "m")

    def save(self):
        self.pdf.save(); self.prs.save(os.path.join(HERE, "Fable4.2_slides.pptx"))


d = Deck()

# ------------------------------------------------------------------------------------------------------ 1. title
d.slide("Fable 4.2", dark=True, title_size=66, title_y=1.75)
d.text(0.6, 1.2, 9.0, 0.3, "EUROPEAN AI CHAMPIONSHIP  ·  PARIS  ·  21–22 NOVEMBER 2026", 14, True, GOLD)
d.text(0.6, 3.05, 10.5, 0.9, "Three security and privacy tasks in 24 hours,\nworked by one orchestrator and one Claude agent per task", 22, False, SOFT)
for i, (n, sc, lab) in enumerate([(1, T1_SCORE, "Watermark localisation · mean IoU"), (2, T2_SCORE, "Model equality testing · MCC score"),
                                  (3, T3_SCORE, "Unlearning · forget × retain")]):
    x = 0.6 + i * 4.1
    d.rect(x, 4.75, 3.8, 1.75, "262A3B", 0.08)
    d.badge(x + 0.25, 4.98, n)
    d.text(x + 1.1, 4.9, 2.6, 0.8, sc, 44, True, PAPER, "l", "m")
    d.text(x + 0.25, 5.85, 3.4, 0.45, lab, 14, False, SOFT, "l", "m")
d.footer(AS_OF)

# --------------------------------------------------------------------------------------------------- 2. overview
d.slide("Three tasks, one way of working")
d.text(0.6, 1.35, 5.3, 0.3, "WHO DID WHAT", 12, True, MUTED)
d.rect(0.6, 1.75, 5.3, 0.8, TINT, 0.12); d.text(0.8, 1.75, 4.9, 0.8, "Human team\naccesses and decisions", 14, False, TEXT, "c", "m")
d.text(0.6, 2.55, 5.3, 0.4, "↓", 20, True, MUTED, "c", "m")
d.rect(0.6, 2.95, 5.3, 0.8, INK, 0.12); d.text(0.8, 2.95, 4.9, 0.8, "Orchestrator agent\npriorities, shared GPUs, journal", 14, True, PAPER, "c", "m")
d.text(0.6, 3.75, 5.3, 0.4, "↓", 20, True, MUTED, "c", "m")
for i in (1, 2, 3):
    x = 0.6 + (i - 1) * 1.8
    d.rect(x, 4.15, 1.7, 0.8, TASKC[i], 0.12); d.text(x, 4.15, 1.7, 0.8, f"Task {i}\nagent", 14, True, PAPER, "c", "m")
d.rect(0.6, 5.15, 5.3, 0.6, TINT, 0.12); d.text(0.8, 5.15, 4.9, 0.6, "+ research agents on demand (review, scouting)", 14, False, TEXT, "c", "m")
d.text(0.6, 5.95, 5.3, 0.6, "Compute: one node with 4× A100 (JURECA reservation)\nand one local RTX 3080", 13, False, MUTED)
d.text(6.6, 1.35, 6.1, 0.3, "WHAT WE BUILT", 12, True, MUTED)
for i, (n, name, idea, sc) in enumerate([
        (1, "Watermark localisation", "Segmenters on synthetic composites, rectangles found by decoding the known message, object shapes from Segment Anything", T1_SCORE),
        (2, "Model equality testing", "Rebuild each endpoint locally from public checkpoints, then run exact tests on its answers", T2_SCORE),
        (3, "Unlearning", "Per-image targets: what would a model that never saw this image predict?", T3_SCORE)]):
    y = 1.75 + i * 1.62
    d.rect(6.6, y, 6.13, 1.45, TINT, 0.08); d.badge(6.8, y + 0.2, n)
    d.text(7.6, y + 0.14, 3.5, 0.34, name, 17, True, TEXT)
    d.text(7.6, y + 0.52, 3.55, 0.85, idea, 12.5, False, MUTED)
    d.text(11.2, y, 1.4, 1.45, sc, 28, True, TASKC[n], "r", "m")
d.footer(AS_OF)

# -------------------------------------------------------------------------------------------------- 3. task 1 idea
d.slide("Task 1 · Three edit types, one tool for each")
d.badge(12.1, 0.45, 1)
d.text(0.6, 1.3, 12.0, 0.4, "Input: an edited image, the watermark method and the embedded message. Output: a 128×128 mask of what is still watermarked.", 14, False, MUTED)
cols = [("Full frame with a hole", "about 50 % of images", "Segmenter per method", "The pretrained watermark extractor is the encoder (PixelSeal, TrustMark); a U-Net with high-pass inputs for StegaStamp. Trained on synthetic composites made with the kit embedders."),
        ("Rectangle only", "about 25 %", "Decode crops against the message", "Bit accuracy peaks when the crop is the embedding rectangle: 0.94 TrustMark, 0.81 StegaStamp. A rectangle is accepted only at 0.85 or more."),
        ("Kept on an object", "about 25 %", "Snap onto Segment Anything masks", "Real labels follow object contours. An object-shaped prediction is replaced by the SAM mask that agrees best with it.")]
for i, (ty, share, tool, how) in enumerate(cols):
    x = 0.6 + i * 4.11
    d.rect(x, 1.95, 3.9, 1.0, INK, 0.1); d.text(x + 0.2, 2.05, 3.5, 0.45, ty, 18, True, PAPER); d.text(x + 0.2, 2.5, 3.5, 0.35, share, 13, False, SOFT)
    d.text(x, 2.95, 3.9, 0.45, "↓", 20, True, MUTED, "c", "m")
    d.rect(x, 3.4, 3.9, 2.55, TINT, 0.05); d.text(x + 0.2, 3.52, 3.5, 0.65, tool, 17, True, C1, "l", "m")
    d.text(x + 0.2, 4.25, 3.5, 1.65, how, 14.5, False, TEXT)
d.text(0.6, 6.2, 12.1, 0.4, "Type known in advance: decode the full frame; bit accuracy of 0.7 or more means a full-frame watermark.", 13.5, True, C1)
d.footer()

# ------------------------------------------------------------------------------------------- 4. task 1 difference
d.slide("Task 1 · Real images were zoomed after embedding")
d.badge(12.1, 0.45, 1)
d.rect(0.6, 1.45, 5.0, 5.15, INK, 0.05)
d.text(0.9, 1.65, 4.4, 0.3, "THE FINDING", 12, True, GOLD)
d.text(0.9, 2.0, 4.4, 1.5, "Real watermarked content was enlarged by ×1.00 to ×1.07, then cropped, after embedding", 18, True, PAPER)
d.text(0.9, 3.55, 4.4, 0.9, "A model trained on clean kit embeddings sees almost nothing in real images.", 13.5, False, SOFT)
d.text(0.9, 4.45, 4.4, 0.3, "StegaStamp, trained with no real image, full frames:", 12, False, SOFT)
d.text(0.9, 4.78, 1.6, 0.8, "0.07", 40, True, SOFT, "l", "m"); d.text(2.4, 4.78, 0.7, 0.8, "→", 30, True, GOLD, "c", "m"); d.text(3.1, 4.78, 1.8, 0.8, "0.83", 40, True, GOLD, "l", "m")
d.text(0.9, 5.65, 4.4, 0.75, "mIoU on real images, without and with the same zoom applied to the synthetic training data", 12, False, SOFT)
d.table(6.0, 1.45, [4.2, 1.55, 1.0], [("Step", "Public mIoU", "Gain")] + T1_LADDER, row_h=0.5, size=13.5,
        aligns=["l", "r", "r"], hi=len(T1_LADDER))
d.text(6.0, 6.03, 6.75, 0.5, "Did not help: matched filter on the expected residual, pseudo-labels of the test set in long training.", 12.5, False, MUTED)
d.footer(AS_OF)

# -------------------------------------------------------------------------------------------------- 5. task 2 idea
d.slide("Task 2 · Do not cluster the answers, rebuild the model")
d.badge(12.1, 0.45, 2)
d.text(0.6, 1.3, 12.0, 0.5, "33 endpoints, 100 answers each. Same model = same checkpoint, quantisation and LoRA.\nSystem prompt, persona, watermark and temperature do not change the identity.", 14, False, MUTED)
steps = [("1", "Questionnaires", "40 one-word questions in one prompt. Answer statistics split the endpoints into 3 families of 11."),
         ("2", "Find the checkpoint", "Score the answers under public models. The right one stands out: −0.33 nat per token against −0.62 for its neighbour."),
         ("3", "Exact tests on a local replica", "The replica gives the probability of every token: fitted temperature, question-by-question fit, fp8 replica, watermark signature."),
         ("4", "Merge only what is identified", "An endpoint joins the base model only when its disguise is positively recognised.")]
for i, (k, head, body) in enumerate(steps):
    x = 0.6 + i * 3.08
    d.rect(x, 1.95, 2.9, 2.85, TINT, 0.05)
    d.oval(x + 0.2, 2.13, 0.5, C2); d.text(x + 0.2, 2.13, 0.5, 0.5, k, 18, True, PAPER, "c", "m")
    d.text(x + 0.2, 2.75, 2.5, 0.65, head, 16, True, TEXT, "l", "m")
    d.text(x + 0.2, 3.45, 2.5, 1.3, body, 13, False, TEXT)
    if i < 3: d.text(x + 2.9, 3.2, 0.18, 0.5, "›", 22, True, MUTED, "c", "m")
d.table(0.6, 5.0, [1.4, 3.6, 3.9, 3.23], [("Family", "Base model found", "Log-likelihood per token", "Runner-up"),
        ("A", "Qwen2.5-3B-Instruct", "−0.33", "1.5B: −0.62"), ("B", "Qwen2.5-1.5B-Instruct", "−0.25", "0.5B: −0.58"),
        ("C", "Llama-3.2-1B-Instruct", "−0.35", "3B: −0.71")], row_h=0.41, size=13, head_fill=C2)
d.footer()

# ------------------------------------------------------------------------------------------------ 6. task 2 result
d.slide("Task 2 · Each rule we could justify moved the score")
d.badge(12.1, 0.45, 2)
d.text(0.6, 1.35, 6.3, 0.3, "PUBLIC SCORE BY SUBMISSION", 12, True, MUTED)
d.bars(0.6, 1.8, 6.5, T2_LADDER, C2, label_w=3.75, row_h=0.68, size=13.5)
d.text(0.6, 5.45, 6.3, 0.9, "Validated first on the 9 practice endpoints: only the fp8 endpoint prefers our fp8 replica; the fit test rejects LoRA 0.7 and 1.5.", 13.5, False, TEXT)
d.text(7.5, 1.35, 5.2, 0.3, "WHAT IS STILL OPEN", 12, True, MUTED)
d.rect(7.5, 1.8, 5.23, 2.8, TINT, 0.05)
d.text(7.75, 1.95, 4.75, 0.9, "4 endpoints out of 33", 26, True, C2, "l", "m")
d.items(7.75, 2.9, 4.75, ["Base model, or a slightly modified one? Our tests cannot separate them from the base replica.",
                          "The public score is 1.00 for every remaining reading: it can refute, not confirm."], 13.5, TEXT, C2)
d.rect(7.5, 4.8, 5.23, 1.55, INK, 0.05)
d.text(7.75, 5.02, 4.75, 0.3, "BUDGET USED", 12, True, GOLD)
d.text(7.75, 5.4, 4.75, 0.85, "35 to 71 of the 100 answers per endpoint; most analysis reused answers already paid for.", 13.5, False, PAPER)
d.footer(AS_OF)

# -------------------------------------------------------------------------------------------------- 7. task 3 idea
d.slide("Task 3 · Predict the model that never saw the image")
d.badge(12.1, 0.45, 3)
d.text(0.6, 1.3, 12.0, 0.4, "Forget 3 500 unlabeled images and match a retrained model image by image. The server returns one number.", 14, False, MUTED)
d.text(0.6, 1.85, 4.3, 0.3, "FOUR FAMILIES OF PROXY MODELS", 12, True, MUTED)
fams = [("P", "Probes on 5 frozen backbones, out-of-fold"), ("I", "Original model on img2img variations of the image"),
        ("S", "Students distilled on generated images only"), ("R", "Same ViT, last blocks reset and re-distilled")]
for i, (k, lab) in enumerate(fams):
    y = 2.25 + i * 1.03
    d.rect(0.6, y, 4.3, 0.88, TINT, 0.1); d.oval(0.75, y + 0.19, 0.5, C3); d.text(0.75, y + 0.19, 0.5, 0.5, k, 18, True, PAPER, "c", "m")
    d.text(1.4, y, 3.4, 0.88, lab, 13.5, False, TEXT, "l", "m")
d.text(4.9, 3.4, 0.6, 0.6, "→", 26, True, MUTED, "c", "m")
d.rect(5.5, 2.25, 3.3, 3.97, INK, 0.05)
d.text(5.75, 2.4, 2.8, 0.3, "CONSENSUS → TARGET", 12, True, GOLD)
d.text(5.75, 2.8, 2.8, 0.7, "One target distribution per image", 17, True, PAPER)
d.items(5.75, 3.65, 2.8, ["how hard: rank mapped to a loss (mean 4.4, std 6.4)", "which wrong class: mixture of the families",
                          "about 45 % of images get a wrong prediction"], 13.5, SOFT, GOLD, gap=0.14)
d.text(8.8, 3.4, 0.6, 0.6, "→", 26, True, MUTED, "c", "m")
d.rect(9.4, 2.25, 3.33, 3.97, TINT, 0.05)
d.text(9.65, 2.4, 2.85, 0.3, "DISTILLATION", 12, True, MUTED)
d.text(9.65, 2.8, 2.85, 0.7, "Fine-tune the given model", 17, True, C3)
d.items(9.65, 3.65, 2.85, ["onto the targets, on the 3 500 images", "onto its own outputs, on generated look-alike images (retain)",
                           "about 9 minutes on one A100"], 13.5, TEXT, C3, gap=0.14)
d.footer()

# ------------------------------------------------------------------------------------------- 8. task 3 difference
d.slide("Task 3 · Each independent proxy family added score")
d.badge(12.1, 0.45, 3)
d.text(0.6, 1.35, 7.0, 0.3, "PUBLIC SCORE BY SUBMISSION", 12, True, MUTED)
d.bars(0.6, 1.75, 7.2, T3_LADDER, C3, label_w=3.9, row_h=0.56, size=13.5, vmax=0.45, fmt="{:.4f}")
d.text(0.6, 5.8, 7.2, 0.7, "Last step from a second agent reviewing the first: an average of retrained models is never fully sure, so cap our confidence too.", 13, False, TEXT)
d.text(8.2, 1.35, 4.5, 0.3, "ONE CHANGE PER SUBMISSION", 12, True, MUTED)
d.table(8.2, 1.75, [2.55, 1.98], [("Setting tested", "Public score"), ("Loss level ×1.5", "0.2907"), ("Loss level ×2.0", "0.3122"), ("Loss level ×2.5", "0.2755"),
        ("35 % of images wrong", "0.2862"), ("45 % of images wrong", "0.3142"), ("55 % of images wrong", "0.3022")], row_h=0.44, size=13, aligns=["l", "r"], head_fill=C3)
d.text(8.2, 5.0, 4.53, 1.5, "Did not help: a model of the reference fitted to the leaderboard (predicted +12 %, measured 0.3533 → 0.3010), weight averaging, stronger anchoring.", 12.5, False, MUTED)
d.footer(AS_OF)

# -------------------------------------------------------------------------------------------------- 9. conclusion
d.slide("What we take away", dark=True)
colsc = [("WHAT WORKED", GOLD, ["Use what is known: the message (task 1), public checkpoints (task 2), the metric itself (task 3)",
                                "Measure before believing: held-out real images, practice endpoints, one change per submission",
                                "A second agent reviewing the first: 0.37 → 0.40 on task 3 in one hour"]),
         ("WHAT DID NOT", SOFT, ["Matched filter on the expected watermark residual: AUC 0.50",
                                 "Pseudo-labels of the test set in long training: the model memorises its teacher",
                                 "Fitting a model of the reference to leaderboard scores: wrong when extrapolated"]),
         ("WITH MORE TIME", GOLD, ["Task 1: a learned selector for SAM masks; small holes in PixelSeal and TrustMark frames",
                                   "Task 2: directed tests for the 4 open endpoints",
                                   "Task 3: more independent proxy families"])]
for i, (head, col, lines) in enumerate(colsc):
    x = 0.6 + i * 4.11
    d.rect(x, 1.45, 3.9, 4.45, "262A3B", 0.05)
    d.text(x + 0.25, 1.65, 3.4, 0.3, head, 13, True, col)
    d.items(x + 0.25, 2.15, 3.45, lines, 16, PAPER, col, gap=0.26)
d.text(0.6, 6.2, 12.1, 0.4, f"Public scores: {T1_SCORE} · {T2_SCORE} · {T3_SCORE}.  Code, task READMEs and these slides are in the team repository.", 14, True, PAPER)
d.footer(AS_OF)

d.save()
print("wrote Fable4.2_slides.pptx and Fable4.2_slides.pdf,", d.n, "slides")
