// Editable overview flowchart generator.
// Requires: npm install pptxgenjs
// Run: node generate_overview_exact.js
const pptxgen = require('pptxgenjs');
const pptx = new pptxgen();
pptx.defineLayout({ name: 'OVERVIEW_REF', width: 16.93, height: 9.29 });
pptx.layout = 'OVERVIEW_REF';
pptx.author = 'Codex';
pptx.subject = 'Editable reproduction of overview.png';
pptx.title = 'Overview of Our Defense Framework';
pptx.theme = { headFontFace: 'Aptos', bodyFontFace: 'Aptos', lang: 'en-US' };

const C = {
  bg:'FEFFFF', navy:'071A58', border:'4E6289', softBorder:'9AA8C3',
  green:'A9D89A', greenDark:'236B19', greenSoft:'F4FAF0',
  blue:'AFCDF9', blueDark:'11529C', blueSoft:'F4F8FF',
  red:'F4B7AF', redDark:'B51111', redSoft:'FFF5F3',
  purple:'C9A8E9', purpleDark:'3B126E', purpleSoft:'FAF5FF',
  black:'101010', gray:'555555', white:'FFFFFF'
};

function box(slide,x,y,w,h,{fill='FFFFFF',line=C.border,radius=true,lw=0.8}={}){
  slide.addShape(radius ? pptx.ShapeType.roundRect : pptx.ShapeType.rect, {x,y,w,h,fill:{color:fill},line:{color:line,pt:lw}});
}
function text(slide,x,y,w,h,t,{size=11,color=C.black,bold=false,italic=false,align='center',fontFace='Aptos'}={}){
  slide.addText(t,{x,y,w,h,fontFace,fontSize:size,color,bold,italic,align,margin:0.02,fit:'shrink',valign:'mid',breakLine:false});
}
function circle(slide,cx,cy,r,fill,line=fill,lw=0.7){
  slide.addShape(pptx.ShapeType.ellipse,{x:cx-r,y:cy-r,w:2*r,h:2*r,fill:{color:fill},line:{color:line,pt:lw}});
}
function line(slide,x1,y1,x2,y2,{color=C.black,lw=1,dash=false}={}){
  slide.addShape(pptx.ShapeType.line,{x:x1,y:y1,w:x2-x1,h:y2-y1,line:{color,pt:lw,dash:dash?'dash':'solid'}});
}
function arrow(slide,x,y,w,h,color=C.black,type='rightArrow'){
  slide.addShape(pptx.ShapeType[type],{x,y,w,h,fill:{color},line:{color,pt:0.2}});
}
function network(slide,x,y,w,h,{node=C.blueDark,edge='32528E',bg=null}={}){
  if(bg) box(slide,x,y,w,h,{fill:bg,line:C.softBorder,lw:0.5});
  const pts=[[.18,.25],[.48,.18],[.78,.28],[.25,.62],[.55,.55],[.82,.72]].map(([a,b])=>[x+a*w,y+b*h]);
  [[0,1],[1,2],[0,3],[1,3],[1,4],[2,4],[2,5],[3,4],[4,5]].forEach(([i,j])=>line(slide,pts[i][0],pts[i][1],pts[j][0],pts[j][1],{color:edge,lw:.55}));
  pts.forEach(([px,py])=>circle(slide,px,py,Math.min(w,h)*.075,node,'1E3765',.55));
}
function gradientBar(slide,x,y,w,h,left='2D67B2',right='D63C2F'){
  // PowerPoint rectangles emulate a horizontal color ramp.
  const steps=22;
  const mix=(a,b,t)=>[0,2,4].map(i=>Math.round(parseInt(a.slice(i,i+2),16)*(1-t)+parseInt(b.slice(i,i+2),16)*t).toString(16).padStart(2,'0')).join('').toUpperCase();
  for(let i=0;i<steps;i++) slide.addShape(pptx.ShapeType.rect,{x:x+i*w/steps,y,w:w/steps+0.003,h,fill:{color:mix(left,right,i/(steps-1))},line:{color:mix(left,right,i/(steps-1)),pt:0}});
  [.12,.27,.55,.82,.93].forEach(p=>circle(slide,x+p*w,y+h/2,h*.18,C.black,C.black,.2));
}
function smallDatabase(slide,x,y,w,h,fill,lineColor){
  slide.addShape(pptx.ShapeType.can,{x,y,w,h,fill:{color:fill},line:{color:lineColor,pt:.7}});
  line(slide,x+w*.25,y+h*.42,x+w*.75,y+h*.42,{color:lineColor,lw:.35});
  line(slide,x+w*.25,y+h*.62,x+w*.75,y+h*.62,{color:lineColor,lw:.35});
}

async function main() {
const slide = pptx.addSlide();
slide.background = { color: C.bg };

// Title and legend
text(slide,2.45,.17,12.3,.30,'Overview of Our Model Utility-Preserving Backdoor Defense Framework for Split Learning under Data Heterogeneity',{size:22,color:C.navy,bold:true});
box(slide,.15,.56,16.43,.73,{fill:C.white,line:C.border});
[[.87,C.green,C.greenDark,'Benign client\n(regular data)'],[2.88,C.blue,C.blueDark,'Unseen client\n(benign, novel data)'],[5.10,C.red,C.redDark,'Malicious client\n(backdoor attack)']].forEach(([x,c,d,t])=>{circle(slide,x,.84,.11,c,d,.8); text(slide,x+.24,.69,1.35,.33,t,{size:10.7,color:'000000',align:'left'});});
line(slide,7.12,.83,7.62,.83); arrow(slide,7.58,.77,.15,.12,'000000','rightArrow'); text(slide,7.75,.70,1.7,.25,'Forward (activations)  H_i',{size:10.8,color:'000000',align:'left'});
line(slide,7.12,1.08,7.62,1.08,{dash:true}); arrow(slide,7.58,1.02,.15,.12,'000000','rightArrow'); text(slide,7.75,.96,1.75,.25,'Backward (gradients)  T_i',{size:10.8,color:'000000',align:'left'});
network(slide,11.10,.68,.72,.52,{node:'B9CEF7',edge:'32528E',bg:'F6F9FE'}); text(slide,11.98,.71,1.62,.32,'Client-side model  H_i\n(kept on client)',{size:10.8,color:'000000',align:'left'});
network(slide,14.18,.68,.72,.52,{node:'2D67B2',edge:'32528E',bg:'F6F9FE'}); text(slide,15.05,.71,1.25,.32,'Server-side model  B\n(kept on server)',{size:10.8,color:'000000',align:'left'});

// Major panels: left split learning, center two-phase framework, right update, bottom policy.
box(slide,.15,1.62,3.96,5.49,{fill:C.white,line:C.border}); box(slide,1.05,1.45,2.14,.34,{fill:'F2F5FE',line:C.border}); text(slide,1.23,1.53,1.78,.13,'Split Learning System',{size:11.8,color:C.navy,bold:true});
slide.addShape(pptx.ShapeType.cloud,{x:.77,y:2.03,w:2.66,h:1.02,fill:{color:'F8FAFF'},line:{color:C.border,pt:.8}}); text(slide,1.22,2.05,1.18,.2,'Server B',{size:11.2,color:'000000',bold:true}); network(slide,1.22,2.27,.62,.52,{node:'808ACF'}); text(slide,1.97,2.43,.18,.12,'...',{size:13,color:'000000'}); network(slide,2.30,2.27,.62,.52,{node:'6B78C3'});
[.45,1.42,2.38,3.40].forEach(x=>{line(slide,x,3.13,x,3.67,{color:C.blueDark,lw:1}); arrow(slide,x-.055,3.58,.11,.18,C.blueDark,'downArrow'); line(slide,x+.22,3.13,x+.22,3.66,{color:'000000',lw:1,dash:true}); arrow(slide,x+.165,3.57,.11,.18,'000000','downArrow');});
const clients=[[.21,C.greenSoft,C.greenDark,'Client 1\n(benign)','Regular\nData D_1',C.green],[1.16,C.greenSoft,C.greenDark,'Client k\n(benign)','Regular\nData D_k',C.green],[2.15,C.blueSoft,C.blueDark,'Client u\n(unseen)','Unseen / New\nData D_u',C.blue],[3.15,C.redSoft,C.redDark,'Client m\n(malicious)','Poisoned\nData D_m\n(trigger)',C.red]];
clients.forEach(([x,fill,stroke,title,data,node])=>{box(slide,x,3.75,.75,2.33,{fill,line:stroke,lw:.6}); text(slide,x+.08,3.88,.59,.35,title,{size:8.9,color:stroke,bold:true}); network(slide,x+.14,4.35,.45,.54,{node,edge:stroke}); smallDatabase(slide,x+.19,5.19,.35,.34,node,stroke); text(slide,x+.08,5.59,.59,.42,data,{size:7.3,color:'000000'});});
text(slide,.96,4.58,.15,.15,'...',{size:13,color:'000000'}); text(slide,2.98,4.58,.15,.15,'...',{size:13,color:'000000'});
box(slide,.24,6.24,3.75,.80,{fill:C.white,line:C.softBorder}); text(slide,.32,6.38,3.52,.44,'• Clients keep their raw data and client-side model locally.\n• Only intermediate activations H_i are sent to the server.\n• Gradients T_i are returned to clients for local update.',{size:8.6,color:'000000',align:'left'}); arrow(slide,4.05,3.50,.28,.16,'000000','rightArrow');

box(slide,4.30,1.60,9.80,4.80,{fill:C.white,line:C.border}); text(slide,6.99,1.50,3.65,.22,'Our Two-Phase Defense Framework',{size:13.5,color:C.navy,bold:true});
box(slide,4.43,1.82,4.35,4.48,{fill:C.greenSoft,line:'A8C39D'}); text(slide,5.05,1.91,3.06,.34,'Phase 1: Unseen Client Separation and\nConfidence Score Assignment',{size:12.1,color:'143B12',bold:true}); line(slide,4.44,2.34,8.78,2.34,{color:'C6D6BF',lw:.7});
box(slide,4.55,2.50,1.77,2.97,{fill:C.white,line:'A8C39D',lw:.6}); text(slide,5.00,2.58,.82,.24,'For each client i\nextract candidate features',{size:7.9,color:'000000',bold:true}); text(slide,5.10,3.14,1.05,.28,'(1) Gradient Sparsity\n(Model Update)',{size:7.2,color:'000000',align:'left'}); text(slide,5.10,3.83,1.05,.28,'(2) Direction Consistency\n(Model Update)',{size:7.2,color:'000000',align:'left'}); text(slide,5.10,4.47,1.05,.28,'(3) Frequency Discrepancy\n(Model Intrinsic)',{size:7.2,color:'000000',align:'left'}); text(slide,5.31,5.15,.28,.12,'...',{size:13,color:'000000'}); arrow(slide,6.30,3.05,.28,.14,'000000','rightArrow');
box(slide,6.58,2.50,1.93,1.18,{fill:C.white,line:'A8C39D',lw:.6}); text(slide,6.80,2.60,1.48,.25,'Robust Normalization\n(Median & MAD)',{size:8,color:'000000',bold:true}); arrow(slide,7.45,3.67,.14,.30,'000000','downArrow');
box(slide,6.43,3.88,2.22,.99,{fill:C.white,line:'A8C39D',lw:.6}); text(slide,6.70,4.02,1.60,.25,'Deviation Score Z_i\n(Higher means more likely unseen)',{size:8.2,color:'000000',bold:true}); gradientBar(slide,6.57,4.50,1.92,.10); text(slide,6.55,4.66,.38,.11,'Low',{size:7,color:'000000',align:'left'}); text(slide,8.25,4.66,.30,.11,'High',{size:7,color:'000000'});
arrow(slide,7.45,4.86,.14,.26,'000000','downArrow'); box(slide,6.43,5.13,2.22,1.02,{fill:C.white,line:'A8C39D',lw:.6}); text(slide,6.86,5.26,1.35,.25,'Confidence Score α_i\n(Lower means more likely unseen)',{size:8,color:'000000',bold:true}); gradientBar(slide,6.57,5.77,1.92,.10,'5BAA52','D63C2F'); text(slide,6.55,5.94,.38,.11,'High',{size:7,color:'000000',align:'left'}); text(slide,8.23,5.94,.32,.11,'Low',{size:7,color:'000000'});
line(slide,8.88,3.40,8.88,5.45,{color:'000000',lw:1,dash:true}); arrow(slide,8.80,3.35,.22,.13,'000000','rightArrow');
box(slide,9.08,1.82,4.88,4.48,{fill:C.purpleSoft,line:'A98BCC'}); text(slide,9.95,1.91,3.12,.34,'Phase 2: Backdoor Client Detection and\nGradient Calibration',{size:12,color:C.purpleDark,bold:true}); line(slide,9.09,2.34,13.96,2.34,{color:'D9C6EA',lw:.7});
box(slide,9.21,2.50,2.80,1.74,{fill:C.white,line:'A98BCC',lw:.6}); text(slide,9.80,2.62,1.60,.16,'Backdoor Client Detection',{size:8.5,color:C.navy,bold:true}); text(slide,10.36,3.27,1.28,.52,'Unsupervised clustering on\ngradient representations\n(e.g., K-means)',{size:7.5,color:'000000',align:'left'}); arrow(slide,12.02,3.35,.28,.15,'000000','rightArrow');
box(slide,12.34,2.80,1.51,1.34,{fill:C.white,line:'A98BCC',lw:.6}); text(slide,12.68,2.97,.82,.40,'Backdoor\nProbability p_i\n(Higher means more\nlikely backdoor)',{size:7.5,color:'000000',bold:true}); gradientBar(slide,12.50,3.78,1.02,.10); text(slide,12.47,3.95,.33,.10,'Low',{size:6.6,color:'000000'}); text(slide,13.47,3.95,.34,.10,'High',{size:6.6,color:'000000'});
arrow(slide,10.63,4.24,.16,.28,'000000','downArrow'); box(slide,9.21,4.50,4.68,1.56,{fill:C.white,line:'A98BCC',lw:.6}); text(slide,9.98,4.64,1.55,.14,'Gradient Calibration',{size:8.8,color:C.purpleDark,bold:true}); line(slide,9.22,4.86,13.88,4.86,{color:'D9C6EA',lw:.6}); text(slide,9.35,5.10,2.30,.28,'Calibration Factor β_i = α_i(p_i + λ)',{size:8.4,color:'000000',bold:true,align:'left'}); text(slide,9.36,5.48,1.75,.42,'Calibrated Gradient:\nT~_i = β_i · T_i',{size:10.3,color:'000000',bold:true,align:'left'}); box(slide,11.83,4.70,1.98,1.18,{fill:'F8F7FC',line:'C7B6DD',lw:.6}); text(slide,12.30,4.79,1.05,.12,'Effect of Calibration',{size:7.5,color:'000000',bold:true}); arrow(slide,14.08,3.50,.28,.16,'000000','rightArrow');

box(slide,14.30,1.80,2.20,4.90,{fill:C.blueSoft,line:C.border}); text(slide,14.66,1.92,1.50,.34,'Calibrated Training\nand Global Update',{size:11.3,color:C.navy,bold:true}); line(slide,14.31,2.34,16.50,2.34,{color:'BFD0EA',lw:.7}); text(slide,14.58,2.58,1.66,.46,'Server updates its model\nwith calibrated gradients\nfrom all clients.',{size:9.5,color:'000000'}); network(slide,14.70,3.33,1.36,1.00,{node:'2D67B2',edge:'1E3765',bg:'F8FBFF'}); text(slide,14.54,4.44,1.82,.28,'Updated Server Model B^(t+1)',{size:9.4,color:'000000',bold:true}); slide.addShape(pptx.ShapeType.circularArrow,{x:15.06,y:4.82,w:.48,h:.48,fill:{color:'FFFFFF'},line:{color:C.navy,pt:1}}); text(slide,15.65,4.98,.55,.14,'Next Round',{size:8.7,color:C.navy,bold:true,align:'left'}); text(slide,14.52,5.56,1.85,.40,'Broadcast updated model\nto clients for next round\nof split learning.',{size:9.2,color:'000000'}); text(slide,14.63,6.35,1.62,.18,'Repeat until convergence.',{size:8.6,color:'000000',bold:true});

box(slide,4.30,6.68,9.74,1.85,{fill:C.white,line:C.border}); text(slide,7.54,6.82,3.18,.18,'Client Treatment Policy (Combining α_i and p_i)',{size:11.7,color:C.navy,bold:true});
[[4.40,C.greenSoft,C.greenDark,'Others + Benign\n(α_i High, p_i Low)','Keep original gradient','✓'],[6.70,C.blueSoft,C.blueDark,'Unseen + Benign\n(α_i Low, p_i Low)','Keep original gradient','✓'],[9.02,C.redSoft,C.redDark,'Others + Backdoor\n(α_i High, p_i High)','Discard gradient\n(set to zero)','×'],[11.39,C.purpleSoft,C.purpleDark,'Unseen + Backdoor\n(α_i Low, p_i High)','Calibrate gradient\n(T~_i = β_i · T_i)','⚙']].forEach(([x,fill,stroke,title,action,sym],idx)=>{const w=idx===3?2.52:(idx===2?2.24:2.18); box(slide,x,7.06,w,1.30,{fill,line:stroke,lw:.65}); text(slide,x+.20,7.19,w-.40,.28,title,{size:8.6,color:stroke,bold:true}); network(slide,x+.15,7.64,.48,.45,{node:idx===2?C.red:(idx===1?C.blue:C.green),edge:stroke}); text(slide,x+.69,7.72,w-1.08,.40,action,{size:8.7,color:'000000',bold:true}); text(slide,x+w-.35,8.02,.30,.25,sym,{size:18,color:stroke,bold:true});});
line(slide,2.00,7.12,2.00,7.70,{dash:true}); line(slide,2.00,7.70,4.30,7.70,{dash:true}); arrow(slide,4.13,7.62,.18,.14,'000000','rightArrow'); line(slide,15.39,6.70,15.39,7.70,{dash:true}); line(slide,15.39,7.70,14.08,7.70,{dash:true}); arrow(slide,14.04,7.62,.20,.14,'000000','leftArrow');
box(slide,1.40,8.66,14.04,.53,{fill:C.white,line:'777777'}); text(slide,1.65,8.84,.65,.12,'Notations:',{size:8.6,color:'000000',bold:true,align:'left'}); text(slide,2.80,8.81,1.88,.16,'H_i: Intermediate activations (forward)',{size:8.4,color:'000000',align:'left'}); text(slide,5.34,8.81,1.90,.16,'T_i: Gradients (backward)',{size:8.4,color:'000000',align:'left'}); text(slide,7.12,8.81,2.46,.16,'α_i: Confidence score for unseen likelihood',{size:8.4,color:'000000',align:'left'}); text(slide,10.15,8.81,1.48,.16,'p_i: Backdoor probability',{size:8.4,color:'000000',align:'left'}); text(slide,11.68,8.81,1.42,.16,'β_i: Calibration factor',{size:8.4,color:'000000',align:'left'}); text(slide,13.70,8.81,1.55,.16,'T~_i: Calibrated gradient',{size:8.4,color:'000000',align:'left'});

await pptx.writeFile({ fileName: 'overview_exact_editable.pptx' });
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
