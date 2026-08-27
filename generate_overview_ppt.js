const pptxgen = require('pptxgenjs');

async function main() {
const pptx = new pptxgen();
pptx.layout = 'LAYOUT_WIDE';
pptx.author = 'Codex';
pptx.company = 'OpenAI';
pptx.subject = 'Overview flowchart for the proposed defense framework';
pptx.title = 'Method Overview';
pptx.lang = 'en-US';
pptx.theme = {
  headFontFace: 'Aptos',
  bodyFontFace: 'Aptos',
  lang: 'en-US',
};

const C = {
  bg: 'F7F9FC',
  title: '1F2937',
  muted: '5B6574',
  line: 'D6DCE5',
  teal: '2A9D8F',
  tealSoft: 'E9F6F4',
  amber: 'E9C46A',
  amberSoft: 'FFF7E5',
  coral: 'E76F51',
  coralSoft: 'FDEDEA',
  green: '4CAF50',
  greenSoft: 'EAF7EA',
  blue: '4F6BED',
  blueSoft: 'EEF2FF',
  navy: '244B5A',
};

function addRoundCard(slide, { x, y, w, h, title, accent, soft, bodyLines, titleSize = 17, bodySize = 12 }) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x, y, w, h,
    rectRadius: 0.08,
    line: { color: C.line, pt: 1 },
    fill: { color: 'FFFFFF' },
    shadow: { type: 'outer', color: 'B8C2CF', blur: 1, angle: 45, distance: 1, opacity: 0.12 },
  });

  slide.addShape(pptx.ShapeType.roundRect, {
    x: x + 0.06,
    y: y + 0.06,
    w: w - 0.12,
    h: 0.42,
    rectRadius: 0.05,
    line: { color: accent, pt: 0 },
    fill: { color: accent },
  });

  slide.addText(title, {
    x: x + 0.16,
    y: y + 0.11,
    w: w - 0.32,
    h: 0.22,
    fontFace: 'Aptos',
    fontSize: titleSize,
    bold: true,
    color: 'FFFFFF',
    margin: 0,
    valign: 'mid',
    align: 'left',
  });

  slide.addShape(pptx.ShapeType.roundRect, {
    x: x + 0.14,
    y: y + 0.58,
    w: w - 0.28,
    h: h - 0.76,
    rectRadius: 0.04,
    line: { color: soft, pt: 1 },
    fill: { color: soft },
  });

  slide.addText(bodyLines.join('\n'), {
    x: x + 0.22,
    y: y + 0.66,
    w: w - 0.44,
    h: h - 0.92,
    fontFace: 'Aptos',
    fontSize: bodySize,
    color: C.title,
    margin: 0,
    breakLine: false,
    fit: 'shrink',
    valign: 'top',
    paraSpaceAfterPt: 6,
  });
}

function addArrow(slide, x1, y1, x2, y2, color = C.muted) {
  slide.addShape(pptx.ShapeType.line, {
    x: x1,
    y: y1,
    w: x2 - x1,
    h: y2 - y1,
    line: {
      color,
      pt: 2,
      beginArrowType: 'none',
      endArrowType: 'triangle',
    },
  });
}

const slide = pptx.addSlide();
slide.background = { color: C.bg };

slide.addText('Overview of the Proposed Defense Framework', {
  x: 0.45,
  y: 0.25,
  w: 12.55,
  h: 0.45,
  fontFace: 'Aptos',
  fontSize: 24,
  bold: true,
  color: C.title,
  align: 'center',
  margin: 0,
});

slide.addText('Phase I separates unseen clients; Phase II scores backdoor risk; joint analysis determines update handling.', {
  x: 1.0,
  y: 0.74,
  w: 11.35,
  h: 0.24,
  fontFace: 'Aptos',
  fontSize: 11,
  color: C.muted,
  italic: true,
  align: 'center',
  margin: 0,
});

// Left input card
slide.addShape(pptx.ShapeType.roundRect, {
  x: 0.45,
  y: 1.25,
  w: 2.1,
  h: 1.15,
  rectRadius: 0.07,
  line: { color: C.line, pt: 1 },
  fill: { color: 'FFFFFF' },
});
slide.addShape(pptx.ShapeType.roundRect, {
  x: 0.58,
  y: 1.36,
  w: 1.84,
  h: 0.34,
  rectRadius: 0.04,
  line: { color: C.navy, pt: 0 },
  fill: { color: C.navy },
});
slide.addText('Client-side updates', {
  x: 0.72,
  y: 1.42,
  w: 1.56,
  h: 0.16,
  fontFace: 'Aptos',
  fontSize: 14,
  bold: true,
  color: 'FFFFFF',
  align: 'center',
  margin: 0,
});
slide.addText('gradients\nintermediate features\nmodel updates', {
  x: 0.68,
  y: 1.79,
  w: 1.64,
  h: 0.48,
  fontFace: 'Aptos',
  fontSize: 11,
  color: C.title,
  align: 'center',
  margin: 0,
  fit: 'shrink',
});

addArrow(slide, 2.58, 1.82, 3.05, 1.82, C.muted);

addRoundCard(slide, {
  x: 3.15,
  y: 1.0,
  w: 3.62,
  h: 3.9,
  title: 'Phase I: Unseen Client Separation',
  accent: C.teal,
  soft: C.tealSoft,
  bodyLines: [
    'Metrics',
    '• Deep gradient cosine similarity',
    '• Shallow gradient sparsity',
    '• Intermediate-feature transfer-domain LFD',
    '',
    'Output',
    '• Unseen confidence score α_i',
    '• Split into Others / Unseen',
  ],
  titleSize: 16,
  bodySize: 12,
});

addArrow(slide, 6.78, 2.95, 7.35, 2.95, C.muted);

addRoundCard(slide, {
  x: 7.45,
  y: 1.0,
  w: 3.8,
  h: 3.9,
  title: 'Phase II: Backdoor Screening',
  accent: C.blue,
  soft: C.blueSoft,
  bodyLines: [
    'Metrics',
    '• DCT low-frequency statistic',
    '• Gradient L2 norm',
    '• Feature-distribution deviation',
    '',
    'Scoring',
    '• Convert each metric into a comparable risk score',
    '• Higher score = more suspicious',
  ],
  titleSize: 16,
  bodySize: 11.5,
});

addArrow(slide, 11.27, 2.95, 11.85, 2.95, C.muted);

addRoundCard(slide, {
  x: 11.95,
  y: 1.0,
  w: 1.95,
  h: 3.9,
  title: 'Decision',
  accent: C.coral,
  soft: C.coralSoft,
  bodyLines: [
    'Others + Backdoor',
    '→ discard',
    '',
    'Unseen + Backdoor',
    '→ soft aggregation',
    '',
    'Others + Benign',
    '→ retain',
    '',
    'Unseen + Benign',
    '→ retain',
  ],
  titleSize: 15,
  bodySize: 10.2,
});

slide.addShape(pptx.ShapeType.roundRect, {
  x: 3.55,
  y: 5.2,
  w: 8.8,
  h: 1.05,
  rectRadius: 0.06,
  line: { color: C.line, pt: 1 },
  fill: { color: 'FFFFFF' },
});
slide.addText('Soft aggregation rule', {
  x: 3.75,
  y: 5.34,
  w: 2.0,
  h: 0.2,
  fontFace: 'Aptos',
  fontSize: 13,
  bold: true,
  color: C.title,
  margin: 0,
});
slide.addText('w_{t+1} = w_t + δ · Δg*      ,      Δg* = (1 − α) · Δg_orig + α · Δg_new', {
  x: 5.0,
  y: 5.31,
  w: 6.9,
  h: 0.24,
  fontFace: 'Cambria Math',
  fontSize: 13,
  color: C.navy,
  align: 'center',
  margin: 0,
  fit: 'shrink',
});
slide.addText('Phase II uses risk scores to filter suspicious clients while preserving benign unseen clients whenever possible.', {
  x: 3.75,
  y: 5.65,
  w: 8.35,
  h: 0.2,
  fontFace: 'Aptos',
  fontSize: 10.5,
  color: C.muted,
  italic: true,
  align: 'center',
  margin: 0,
});

addArrow(slide, 13.0, 4.92, 13.0, 5.15, C.green);

slide.addShape(pptx.ShapeType.roundRect, {
  x: 12.2,
  y: 5.55,
  w: 1.5,
  h: 0.62,
  rectRadius: 0.04,
  line: { color: C.green, pt: 1 },
  fill: { color: C.greenSoft },
});
slide.addText('Server update', {
  x: 12.28,
  y: 5.74,
  w: 1.34,
  h: 0.14,
  fontFace: 'Aptos',
  fontSize: 12,
  bold: true,
  color: C.green,
  align: 'center',
  margin: 0,
});

// Small footer legend
slide.addText('Lower α_i indicates a higher likelihood of being unseen; higher risk scores indicate a higher likelihood of backdoor behavior.', {
  x: 0.7,
  y: 6.63,
  w: 12.0,
  h: 0.2,
  fontFace: 'Aptos',
  fontSize: 9.5,
  color: C.muted,
  align: 'center',
  margin: 0,
});

await pptx.writeFile({ fileName: 'method_overview_flowchart.pptx' });
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

