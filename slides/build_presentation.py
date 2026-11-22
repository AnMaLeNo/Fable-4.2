from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

NAVY=RGBColor(0x0F,0x1B,0x2D); ACC=RGBColor(0x14,0xB8,0xA6); BG=RGBColor(0xF7,0xF8,0xFA)
TXT=RGBColor(0x1F,0x29,0x37); GREY=RGBColor(0x6B,0x72,0x80); WHITE=RGBColor(255,255,255)
WARM=RGBColor(0xF5,0x9E,0x0B); CARD=RGBColor(0xFF,0xFF,0xFF); LINE=RGBColor(0xE5,0xE7,0xEB)
SOFT=RGBColor(0xE6,0xF7,0xF5)
FONT="Arial"
p=Presentation(); p.slide_width=Inches(13.333); p.slide_height=Inches(7.5)
W=13.333

def rect(s,x,y,w,h,fill,line=None,shape=MSO_SHAPE.RECTANGLE,rad=None):
    sh=s.shapes.add_shape(shape,Inches(x),Inches(y),Inches(w),Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb=fill
    if line is None: sh.line.fill.background()
    else: sh.line.color.rgb=line; sh.line.width=Pt(1)
    sh.shadow.inherit=False
    if rad is not None and shape==MSO_SHAPE.ROUNDED_RECTANGLE: sh.adjustments[0]=rad
    return sh

def text(s,x,y,w,h,runs,size=18,color=TXT,bold=False,align=PP_ALIGN.LEFT,anchor=MSO_ANCHOR.TOP,sp=4):
    tb=s.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h)); tf=tb.text_frame
    tf.word_wrap=True; tf.vertical_anchor=anchor
    tf.margin_left=tf.margin_right=Inches(0.05); tf.margin_top=tf.margin_bottom=Inches(0.03)
    if isinstance(runs,str): runs=[runs]
    for i,r in enumerate(runs):
        para=tf.paragraphs[0] if i==0 else tf.add_paragraph()
        para.alignment=align; para.space_after=Pt(sp)
        if isinstance(r,str): r=dict(t=r)
        run=para.add_run(); run.text=r["t"]; f=run.font; f.name=FONT
        f.size=Pt(r.get("size",size)); f.bold=r.get("bold",bold); f.color.rgb=r.get("color",color)
    return tb

def base(kicker,title,n):
    s=p.slides.add_slide(p.slide_layouts[6])
    rect(s,0,0,W,7.5,BG)
    rect(s,0,0,0.22,7.5,ACC)
    text(s,0.7,0.42,10,0.4,kicker.upper(),size=14,color=ACC,bold=True)
    text(s,0.7,0.78,12,1.0,title,size=34,color=NAVY,bold=True)
    text(s,0.7,6.95,8,0.3,"fable 4.2  ·  European AI Championship Paris 2026",size=11,color=GREY)
    text(s,11.6,6.95,1.2,0.3,f"{n} / 9",size=11,color=GREY,align=PP_ALIGN.RIGHT)
    return s

def card(s,x,y,w,h,head,body,headcolor=NAVY,fill=CARD,bsize=16):
    rect(s,x,y,w,h,fill,LINE,MSO_SHAPE.ROUNDED_RECTANGLE,0.06)
    rect(s,x,y+0.18,0.07,0.5,ACC)
    text(s,x+0.25,y+0.15,w-0.4,0.6,head,size=19,color=headcolor,bold=True,anchor=MSO_ANCHOR.MIDDLE)
    text(s,x+0.25,y+0.85,w-0.45,h-0.95,body,size=bsize,color=TXT,sp=6)

def hard(s,y,t):
    rect(s,0.7,y,11.9,0.62,SOFT,None,MSO_SHAPE.ROUNDED_RECTANGLE,0.2)
    text(s,0.95,y,11.5,0.62,[dict(t="Why it is hard:  ",bold=True,color=NAVY)],size=15,anchor=MSO_ANCHOR.MIDDLE)
    tb=s.shapes[-1]; r=tb.text_frame.paragraphs[0].add_run(); r.text=t
    r.font.name=FONT; r.font.size=Pt(15); r.font.color.rgb=TXT

def ladder(s,y,steps,h=1.75):
    n=len(steps); gap=0.32; total=11.9; w=(total-gap*(n-1))/n
    for i,(val,lab) in enumerate(steps):
        x=0.7+i*(w+gap); last=i==n-1
        rect(s,x,y,w,h,NAVY if last else CARD,None if last else LINE,MSO_SHAPE.ROUNDED_RECTANGLE,0.08)
        text(s,x,y+0.12,w,0.8,val,size=34 if n<=5 else 28,bold=True,color=WHITE if last else NAVY,align=PP_ALIGN.CENTER,anchor=MSO_ANCHOR.MIDDLE)
        text(s,x+0.05,y+0.92,w-0.1,h-0.95,lab,size=14 if n<=5 else 12,color=WHITE if last else GREY,align=PP_ALIGN.CENTER)
        if not last:
            text(s,x+w-0.02,y+0.25,gap+0.04,0.6,"›",size=26,color=ACC,bold=True,align=PP_ALIGN.CENTER,anchor=MSO_ANCHOR.MIDDLE)

def note(s,y,label,t,color=WARM):
    rect(s,0.7,y,0.07,0.62,color)
    text(s,0.95,y,11.6,0.62,[dict(t=label+"  ",bold=True,color=NAVY)],size=17,anchor=MSO_ANCHOR.MIDDLE)
    tb=s.shapes[-1]; r=tb.text_frame.paragraphs[0].add_run(); r.text=t
    r.font.name=FONT; r.font.size=Pt(17); r.font.color.rgb=TXT

# 1 Title
s=p.slides.add_slide(p.slide_layouts[6]); rect(s,0,0,W,7.5,NAVY); rect(s,0,0,0.22,7.5,ACC)
text(s,0.9,0.7,11,0.4,"EUROPEAN AI CHAMPIONSHIP  ·  PARIS 2026",size=15,color=ACC,bold=True)
text(s,0.9,1.25,11,1.3,"fable 4.2",size=72,color=WHITE,bold=True)
text(s,0.9,2.65,11.5,0.7,"Use what is already known before training anything",size=28,color=RGBColor(0xCB,0xD5,0xE1))
for i,(v,l) in enumerate([("0.80","Watermark localisation"),("1.00","Model equality"),("0.43","Unlearning")]):
    x=0.9+i*3.95
    rect(s,x,4.05,3.6,2.0,RGBColor(0x1A,0x2A,0x44),None,MSO_SHAPE.ROUNDED_RECTANGLE,0.08)
    text(s,x,4.2,3.6,1.1,v,size=54,color=ACC,bold=True,align=PP_ALIGN.CENTER,anchor=MSO_ANCHOR.MIDDLE)
    text(s,x,5.3,3.6,0.5,l,size=18,color=WHITE,align=PP_ALIGN.CENTER)
text(s,0.9,6.4,11,0.4,"Public leaderboard scores (about 30% of the evaluation data)",size=13,color=RGBColor(0x94,0xA3,0xB8))

# 2 How we worked
s=base("Method","A small team of agents that we directed",2)
xs=[0.7,4.75,8.8]
flow=[("Human team","Strategy, budget rules,\nfinal decisions"),("Orchestrator agent","Shares the GPUs\nbetween tasks"),("One agent per task","Does the work, keeps a journal.\nReviewer agents check it.")]
for i,(h,b) in enumerate(flow):
    rect(s,xs[i],1.95,3.75,1.75,NAVY if i==0 else CARD,None if i==0 else LINE,MSO_SHAPE.ROUNDED_RECTANGLE,0.08)
    text(s,xs[i]+0.2,2.08,3.35,0.5,h,size=21,bold=True,color=WHITE if i==0 else NAVY)
    text(s,xs[i]+0.2,2.62,3.35,1.0,b,size=16,color=WHITE if i==0 else TXT)
    if i<2: text(s,xs[i]+3.75,2.4,0.3,0.7,"›",size=30,color=ACC,bold=True,align=PP_ALIGN.CENTER,anchor=MSO_ANCHOR.MIDDLE)
text(s,0.7,4.05,6,0.4,"THREE RULES",size=14,color=ACC,bold=True)
for i,(n,t) in enumerate([("1","Measure before believing"),("2","One change per submission"),("3","One agent reviews another")]):
    x=xs[i]
    rect(s,x,4.5,3.75,1.0,CARD,LINE,MSO_SHAPE.ROUNDED_RECTANGLE,0.12)
    text(s,x+0.15,4.5,0.6,1.0,n,size=30,bold=True,color=ACC,anchor=MSO_ANCHOR.MIDDLE,align=PP_ALIGN.CENTER)
    text(s,x+0.75,4.5,2.9,1.0,t,size=18,bold=True,color=NAVY,anchor=MSO_ANCHOR.MIDDLE)
note(s,5.95,"Compute:","one node of 4× A100 and one local RTX 3080",ACC)

# 3 Task 1 approach
s=base("Task 1 · Watermark localisation","One tool per edit type",3)
text(s,0.7,1.75,12,0.5,"Edited image + watermark method + hidden message: draw the area still watermarked.",size=18,color=GREY)
hard(s,2.4,"three kinds of edits in one dataset. One model cannot handle them all.")
cols=[("Full image with a hole","Segmentation network trained on synthetic images made with the official tools"),
      ("Only a rectangle is watermarked","Decode crops. Known message recovered with ≥ 85% of the bits = exact rectangle"),
      ("Watermark left on an object","Snap our prediction to Segment Anything contours")]
for i,(h,b) in enumerate(cols): card(s,xs[i],3.35,3.75,3.1,h,b,bsize=17)

# 4 Task 1 result
s=base("Task 1 · Watermark localisation","The zoom discovery",4)
rect(s,0.7,1.85,5.0,2.3,NAVY,None,MSO_SHAPE.ROUNDED_RECTANGLE,0.06)
text(s,0.7,1.95,5.0,1.3,"0.07 › 0.83",size=54,bold=True,color=ACC,align=PP_ALIGN.CENTER,anchor=MSO_ANCHOR.MIDDLE)
text(s,0.7,3.2,5.0,0.8,"on full-image edits (StegaStamp)\nSame model, zoom added in training",size=15,color=WHITE,align=PP_ALIGN.CENTER)
text(s,6.0,1.9,6.6,2.3,["Real images were enlarged by up to 7% and cropped after the watermark was embedded.",
     "A model trained on clean synthetic images saw almost nothing in real ones."],size=20,color=TXT,sp=12)
ladder(s,4.4,[("0.26","Baseline"),("0.66","Synthetic segmenters"),("0.80","Zoom, decoded rectangles,\nobject contours")],h=1.5)
note(s,6.1,"Did not work:","pseudo-labels on the test set. The model just copied its teacher.")

# 5 Task 2 approach
s=base("Task 2 · Model equality","Rebuild the model, don't compare answers",5)
text(s,0.7,1.75,12,0.5,"33 language models behind APIs, 100 answers each: which ones are the same model?",size=18,color=GREY)
hard(s,2.4,"a quantised model looks closer to the original than the same model with a persona.")
steps=[("1  Questionnaire","Splits the endpoints into three families"),
       ("2  Find the checkpoint","Search public models for the one that makes the observed answers most likely"),
       ("3  Exact tests on a local replica","Temperature, fit question by question, quantisation, watermark")]
for i,(h,b) in enumerate(steps): card(s,xs[i],3.3,3.75,2.35,h,b,bsize=16)
text(s,0.7,5.85,3,0.4,"FAMILIES FOUND",size=13,color=ACC,bold=True)
for i,m in enumerate(["Qwen2.5-3B","Qwen2.5-1.5B","Llama-3.2-1B"]):
    rect(s,3.1+i*3.2,5.78,2.95,0.55,NAVY,None,MSO_SHAPE.ROUNDED_RECTANGLE,0.4)
    text(s,3.1+i*3.2,5.78,2.95,0.55,m,size=17,bold=True,color=WHITE,align=PP_ALIGN.CENTER,anchor=MSO_ANCHOR.MIDDLE)

# 6 Task 2 result
s=base("Task 2 · Model equality","Every justified rule raised the score",6)
ladder(s,1.95,[("0.10","Families only"),("0.75","Merge disguises identified with certainty"),("0.85","Replica tests"),("1.00","Public leaderboard")])
for i,(h,b) in enumerate([("Validated first","On the 9 practice endpoints, where the answer is known"),
                          ("Still open","4 endpoints of 33: our tests cannot decide"),
                          ("Budget used","35 to 71 of the 100 answers per endpoint")]):
    card(s,xs[i],4.1,3.75,2.3,h,b,bsize=17)

# 7 Task 3 approach
s=base("Task 3 · Unlearning","Ask the model that never saw the image",7)
text(s,0.7,1.75,12,0.5,"A vision model must forget 3,500 images and behave like a model retrained without them.",size=18,color=GREY)
hard(s,2.4,"predictions are compared image by image; the server returns one number.")
fam=["Simple classifiers on pretrained backbones","Original model on regenerated variations of the image","Students trained only on generated images","Same network, last layers reset"]
for i,f in enumerate(fam):
    rect(s,0.7,3.3+i*0.78,5.6,0.66,CARD,LINE,MSO_SHAPE.ROUNDED_RECTANGLE,0.2)
    text(s,0.8,3.3+i*0.78,0.5,0.66,str(i+1),size=18,bold=True,color=ACC,align=PP_ALIGN.CENTER,anchor=MSO_ANCHOR.MIDDLE)
    text(s,1.3,3.3+i*0.78,4.9,0.66,f,size=15,color=TXT,anchor=MSO_ANCHOR.MIDDLE)
text(s,6.3,4.2,0.5,0.9,"›",size=34,color=ACC,bold=True,align=PP_ALIGN.CENTER,anchor=MSO_ANCHOR.MIDDLE)
for i,(h,b,dark) in enumerate([("Consensus","One target per image",False),("Fine-tune","Towards these targets, unchanged elsewhere",True)]):
    x=6.85+i*3.0
    rect(s,x,3.3,2.7,3.0,NAVY if dark else CARD,None if dark else LINE,MSO_SHAPE.ROUNDED_RECTANGLE,0.08)
    text(s,x+0.15,3.9,2.4,0.6,h,size=22,bold=True,color=WHITE if dark else NAVY,align=PP_ALIGN.CENTER)
    text(s,x+0.15,4.6,2.4,1.4,b,size=16,color=WHITE if dark else TXT,align=PP_ALIGN.CENTER)
    if i==0: text(s,x+2.68,4.2,0.34,0.9,"›",size=30,color=ACC,bold=True,align=PP_ALIGN.CENTER,anchor=MSO_ANCHOR.MIDDLE)

# 8 Task 3 result
s=base("Task 3 · Unlearning","Choose the sources, then cap the confidence",8)
ladder(s,1.95,[("0.15","Start"),("0.23","1 family"),("0.29","2 families"),("0.35","3 families"),("0.36","4 families"),("0.40","Confidence cap"),("0.43","5th signal,\none source dropped")],h=1.7)
text(s,0.7,3.95,11.9,1.3,["One setting changed per submission, because the server returns a single number.",
     "An average of retrained models is never fully confident, so we cap our model's confidence too."],size=18,color=TXT,sp=10)
note(s,5.55,"Did not work:","a model of the reference fitted to our past scores. Predicted +12%, score went down.")

# 9 Takeaways
s=base("Conclusion","What we take away",9)
card(s,xs[0],1.95,3.75,3.9,"Worked",["Use what is known: the hidden message, public checkpoints, the scoring formula","Measure before believing","One agent reviews another"],bsize=16)
card(s,xs[1],1.95,3.75,3.9,"With more time",["Task 1: better object contours","Task 2: the 4 open endpoints","Task 3: more proxy families"],bsize=16)
rect(s,xs[2],1.95,3.75,3.9,NAVY,None,MSO_SHAPE.ROUNDED_RECTANGLE,0.06)
text(s,xs[2]+0.25,2.1,3.3,0.5,"Public scores",size=19,bold=True,color=WHITE)
for i,(v,l) in enumerate([("0.80","Watermark localisation"),("1.00","Model equality"),("0.43","Unlearning")]):
    text(s,xs[2]+0.25,2.75+i*1.0,1.5,0.8,v,size=32,bold=True,color=ACC,anchor=MSO_ANCHOR.MIDDLE)
    text(s,xs[2]+1.7,2.75+i*1.0,1.95,0.8,l,size=15,color=WHITE,anchor=MSO_ANCHOR.MIDDLE)
text(s,0.7,6.05,11.9,0.7,"Thank you",size=30,bold=True,color=NAVY)
p.save("Fable4.2_presentation.pptx"); print("ok")
