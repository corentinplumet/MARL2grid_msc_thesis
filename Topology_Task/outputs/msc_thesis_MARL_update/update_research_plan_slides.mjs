import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const ROOT = "/Users/corentinplumet/Documents/ICLR_marl2grid/Topology_Task/Topology_Task";
const INPUT = path.join(ROOT, "msc_thesis_MARL.pptx");
const OUT_DIR = path.join(ROOT, "outputs", "msc_thesis_MARL_update");
const TMP_DIR = path.join(ROOT, "tmp", "slides", "msc_thesis_MARL_update", "revised");
const OUTPUT = path.join(OUT_DIR, "msc_thesis_MARL_revised.pptx");

const W = 1280;
const H = 720;

const COLOR = {
  bg: "#FFFFFF",
  text: "#3F3A3A",
  muted: "#5F5A5A",
  red: "#E2001A",
  line: "#202020",
  soft: "#F7F4F2",
  softRed: "#FFF3F4",
  softGray: "#F5F5F5",
};

const FONT = {
  title: "Avenir Next",
  body: "Avenir Next",
};

async function writeBinary(filePath, payload) {
  if (payload && typeof payload.save === "function") {
    await payload.save(filePath);
    return;
  }
  if (payload && typeof payload.arrayBuffer === "function") {
    const ab = await payload.arrayBuffer();
    await fs.writeFile(filePath, Buffer.from(ab));
    return;
  }
  if (payload instanceof Uint8Array) {
    await fs.writeFile(filePath, payload);
    return;
  }
  if (payload instanceof ArrayBuffer) {
    await fs.writeFile(filePath, Buffer.from(payload));
    return;
  }
  throw new Error(`Unsupported payload for ${filePath}`);
}

function addText(slide, text, left, top, width, height, {
  fontSize = 24,
  bold = false,
  color = COLOR.text,
  alignment = "left",
  verticalAlignment = "top",
  typeface = FONT.body,
  insets = { left: 0, right: 0, top: 0, bottom: 0 },
} = {}) {
  const shape = slide.shapes.add({
    geometry: "rect",
    position: { left, top, width, height },
    fill: "#FFFFFF00",
    line: { width: 0, fill: "#FFFFFF00" },
  });
  shape.text = text;
  shape.text.typeface = typeface;
  shape.text.fontSize = fontSize;
  shape.text.bold = bold;
  shape.text.color = color;
  shape.text.alignment = alignment;
  shape.text.verticalAlignment = verticalAlignment;
  shape.text.insets = insets;
  return shape;
}

function addBox(slide, left, top, width, height, {
  fill = COLOR.bg,
  lineFill = COLOR.line,
  lineWidth = 1.25,
  radius = 0.08,
} = {}) {
  return slide.shapes.add({
    geometry: "roundRect",
    adjustmentList: [{ name: "adj", formula: `val ${Math.round(radius * 50000)}` }],
    position: { left, top, width, height },
    fill,
    line: { style: "solid", fill: lineFill, width: lineWidth },
  });
}

function addHeader(slide, title, pageNumber) {
  slide.background.fill = COLOR.bg;
  addText(slide, "EPFL", 26, 24, 80, 26, {
    fontSize: 24,
    bold: true,
    color: COLOR.red,
    typeface: FONT.body,
  });
  addText(slide, title, 150, 30, 860, 54, {
    fontSize: 44,
    bold: true,
    color: COLOR.text,
    typeface: FONT.title,
  });
  addText(slide, String(pageNumber), 1206, 28, 34, 24, {
    fontSize: 18,
    color: COLOR.text,
    alignment: "right",
  });
  slide.shapes.add({
    geometry: "rect",
    position: { left: 58, top: 692, width: 8, height: 8 },
    fill: COLOR.red,
    line: { width: 0, fill: COLOR.red },
  });
}

function addAccentRule(slide, left, top, width) {
  slide.shapes.add({
    geometry: "rect",
    position: { left, top, width, height: 4 },
    fill: COLOR.red,
    line: { width: 0, fill: COLOR.red },
  });
}

function addBulletLine(slide, text, left, top, width, {
  bulletColor = COLOR.red,
  fontSize = 18,
  color = COLOR.muted,
  bold = false,
} = {}) {
  addText(slide, "•", left, top, 18, 24, {
    fontSize,
    color: bulletColor,
    bold: true,
  });
  addText(slide, text, left + 18, top, width - 18, 28, {
    fontSize,
    color,
    bold,
  });
}

function addArrow(slide, left, top, width = 42, height = 22) {
  slide.shapes.add({
    geometry: "rightArrow",
    position: { left, top, width, height },
    fill: COLOR.red,
    line: { width: 0, fill: COLOR.red },
  });
}

function addResearchGapSlide(slide) {
  addHeader(slide, "Research Gap", 6);

  addBox(slide, 112, 138, 1052, 178, {
    fill: COLOR.bg,
    lineFill: COLOR.line,
    lineWidth: 1.2,
    radius: 0.02,
  });
  addText(slide, "What current studies still miss", 138, 160, 400, 28, {
    fontSize: 22,
    bold: true,
    color: COLOR.text,
  });

  const topCards = [
    {
      x: 140,
      title: "After-state",
      body: "Promising in topology control, but rarely studied as an explicit coordination signal in decentralized MARL.",
    },
    {
      x: 474,
      title: "Communication",
      body: "Coordination is mostly implicit through rewards or centralized training, not through learned agent-to-agent messaging.",
    },
    {
      x: 808,
      title: "Evidence",
      body: "Results remain fragmented across small grids and inconsistent evaluation setups, so it is still unclear what really scales.",
    },
  ];

  for (const card of topCards) {
    addText(slide, card.title, card.x, 208, 250, 26, {
      fontSize: 19,
      bold: true,
      color: COLOR.text,
    });
    addAccentRule(slide, card.x, 237, 78);
    addText(slide, card.body, card.x, 252, 278, 56, {
      fontSize: 16,
      color: COLOR.muted,
      insets: { left: 0, right: 8, top: 0, bottom: 0 },
    });
  }

  addBox(slide, 112, 346, 1052, 278, {
    fill: COLOR.bg,
    lineFill: COLOR.line,
    lineWidth: 1.2,
    radius: 0.02,
  });
  addText(slide, "Open research gap for this thesis", 138, 368, 420, 28, {
    fontSize: 22,
    bold: true,
    color: COLOR.text,
  });

  const gapCards = [
    {
      x: 140,
      fill: COLOR.softRed,
      title: "Representation gap",
      body: "Local observations do not tell an agent enough about the grid state it will create after a topology switch.",
    },
    {
      x: 474,
      fill: COLOR.soft,
      title: "Coordination gap",
      body: "Neighboring agents still lack a compact way to share intent or expected after-state information across zone boundaries.",
    },
    {
      x: 808,
      fill: COLOR.softGray,
      title: "Benchmark gap",
      body: "MARL2Grid has not yet been used to test whether after-state prediction and learned communication improve topology control.",
    },
  ];

  for (const card of gapCards) {
    addBox(slide, card.x, 420, 278, 168, {
      fill: card.fill,
      lineFill: COLOR.line,
      lineWidth: 1,
      radius: 0.08,
    });
    addText(slide, card.title, card.x + 18, 440, 240, 26, {
      fontSize: 18,
      bold: true,
      color: COLOR.text,
    });
    addText(slide, card.body, card.x + 18, 474, 242, 90, {
      fontSize: 16,
      color: COLOR.muted,
    });
  }
}

function addApproachSlide(slide) {
  addHeader(slide, "Approach", 8);

  addText(
    slide,
    "Investigate whether after-state prediction and lightweight communication improve decentralized topology control on MARL2Grid.",
    114,
    118,
    1050,
    32,
    {
      fontSize: 19,
      color: COLOR.muted,
    }
  );

  const stages = [
    {
      x: 104,
      label: "1",
      title: "Local observations",
      body: "Each agent observes its zone, boundary lines, and local grid features from the MARL2Grid topology task.",
    },
    {
      x: 381,
      label: "2",
      title: "Latent communication",
      body: "Neighboring agents exchange compressed messages that summarize intent or boundary-state information.",
    },
    {
      x: 658,
      label: "3",
      title: "After-state predictor",
      body: "A predictor estimates the post-action local topology and overload evolution before the switch is applied.",
    },
    {
      x: 935,
      label: "4",
      title: "Action selection",
      body: "A MAPPO-style policy uses the communicated state and predicted after-state to choose the topology action.",
    },
  ];

  for (let i = 0; i < stages.length; i += 1) {
    const stage = stages[i];
    addBox(slide, stage.x, 196, 240, 172, {
      fill: i % 2 === 0 ? COLOR.soft : COLOR.bg,
      lineFill: COLOR.line,
      lineWidth: 1.15,
      radius: 0.08,
    });
    addBox(slide, stage.x + 14, 210, 34, 34, {
      fill: COLOR.red,
      lineFill: COLOR.red,
      lineWidth: 1,
      radius: 0.5,
    });
    addText(slide, stage.label, stage.x + 14, 215, 34, 24, {
      fontSize: 18,
      bold: true,
      color: COLOR.bg,
      alignment: "center",
      verticalAlignment: "middle",
    });
    addText(slide, stage.title, stage.x + 58, 214, 164, 26, {
      fontSize: 20,
      bold: true,
      color: COLOR.text,
    });
    addText(slide, stage.body, stage.x + 16, 256, 208, 88, {
      fontSize: 17,
      color: COLOR.muted,
    });
    if (i < stages.length - 1) {
      addArrow(slide, stage.x + 246, 270, 26, 18);
    }
  }

  addBox(slide, 114, 426, 500, 188, {
    fill: COLOR.bg,
    lineFill: COLOR.line,
    lineWidth: 1.15,
    radius: 0.04,
  });
  addText(slide, "Training setup", 138, 448, 220, 28, {
    fontSize: 22,
    bold: true,
    color: COLOR.text,
  });
  addBulletLine(slide, "MAPPO backbone with an auxiliary after-state prediction loss.", 138, 488, 440, {
    fontSize: 16,
  });
  addBulletLine(slide, "Message bottleneck controls how much agents can share.", 138, 526, 440, {
    fontSize: 16,
  });
  addBulletLine(slide, "Ablations: baseline, after-state only, communication only, both.", 138, 564, 440, {
    fontSize: 16,
  });

  addBox(slide, 664, 426, 500, 188, {
    fill: COLOR.bg,
    lineFill: COLOR.line,
    lineWidth: 1.15,
    radius: 0.04,
  });
  addText(slide, "Evaluation on MARL2Grid", 688, 448, 320, 28, {
    fontSize: 22,
    bold: true,
    color: COLOR.text,
  });
  addBulletLine(slide, "Topology benchmark under decentralized and centralized observation settings.", 688, 488, 440, {
    fontSize: 16,
  });
  addBulletLine(slide, "Metrics: survival, margins, overload reduction, sample efficiency.", 688, 526, 440, {
    fontSize: 16,
  });
  addBulletLine(slide, "Scalability study from smaller grids to harder MARL2Grid scenarios.", 688, 564, 440, {
    fontSize: 16,
  });
}

function addMonthBlock(slide, left, top, title, lines) {
  slide.shapes.add({
    geometry: "rect",
    position: { left, top: top + 8, width: 7, height: 18 },
    fill: COLOR.red,
    line: { width: 0, fill: COLOR.red },
  });
  addText(slide, title, left + 30, top, 430, 24, {
    fontSize: 18,
    bold: true,
    color: COLOR.text,
  });
  lines.forEach((line, idx) => {
    addBulletLine(slide, line, left + 36, top + 28 + idx * 30, 420, {
      fontSize: 15,
      bulletColor: COLOR.text,
      color: COLOR.muted,
    });
  });
}

function addTimelineSlide(slide) {
  addHeader(slide, "Timeline", 15);

  addMonthBlock(slide, 84, 122, "April - MARL2Grid setup", [
    "Reproduce topology baselines and environment setup",
    "Fix metrics, seeds, and agent partitions",
    "Define the after-state targets",
  ]);

  addMonthBlock(slide, 84, 286, "May - After-state prediction", [
    "Design boundary-aware after-state representations",
    "Train the predictor on collected rollouts",
    "Test whether prediction quality helps control",
  ]);

  addMonthBlock(slide, 84, 450, "June - Communication module", [
    "Implement latent messages between neighboring agents",
    "Compare bottlenecks and neighborhood choices",
    "Check whether communication improves coordination",
  ]);

  addMonthBlock(slide, 684, 122, "July - Joint MARL training", [
    "Integrate after-state and communication into MAPPO",
    "Run ablations for each component",
    "Tune under decentralized observations",
  ]);

  addMonthBlock(slide, 684, 286, "August - Evaluation and writing", [
    "Benchmark on MARL2Grid topology scenarios",
    "Study scalability and failure cases",
    "Write the thesis and final presentation",
  ]);
}

function replaceSlide(presentation, slideIndex, buildSlide) {
  const oldSlide = presentation.slides.getItem(slideIndex);
  const { slide } = presentation.slides.insert({
    after: presentation.slides.getItem(slideIndex - 1),
  });
  buildSlide(slide);
  oldSlide.delete();
}

async function build() {
  await fs.mkdir(OUT_DIR, { recursive: true });
  await fs.mkdir(TMP_DIR, { recursive: true });

  const deckBlob = await FileBlob.load(INPUT);
  const presentation = await PresentationFile.importPptx(deckBlob);

  replaceSlide(presentation, 9, addTimelineSlide);
  replaceSlide(presentation, 8, addApproachSlide);
  replaceSlide(presentation, 6, addResearchGapSlide);

  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(OUTPUT);

  const previews = [
    { index: 6, name: "research_gap.png" },
    { index: 8, name: "approach.png" },
    { index: 9, name: "timeline.png" },
  ];

  for (const preview of previews) {
    const slide = presentation.slides.getItem(preview.index);
    const png = await presentation.export({ slide, format: "png", scale: 1 });
    await writeBinary(path.join(TMP_DIR, preview.name), png);
  }

  const reopened = await PresentationFile.importPptx(await FileBlob.load(OUTPUT));
  for (const preview of previews) {
    const slide = reopened.slides.getItem(preview.index);
    const png = await reopened.export({ slide, format: "png", scale: 1 });
    await writeBinary(path.join(TMP_DIR, `verified_${preview.name}`), png);
  }

  console.log(OUTPUT);
}

await build();
