import fs from "node:fs/promises";
import path from "node:path";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const ROOT = "/Users/corentinplumet/Documents/ICLR_marl2grid/Topology_Task/Topology_Task";
const OUT_DIR = path.join(ROOT, "outputs", "marl2grid_multi_agent_plan");
const TMP_DIR = path.join(ROOT, "tmp", "slides", "marl2grid_multi_agent_plan");
const MEDIA_DIR = "/tmp/marl2grid_plan_media";

const FILE_OUT = path.join(OUT_DIR, "msc_thesis_research_plan_marl2grid_multi_agent.pptx");

const W = 1280;
const H = 720;

const COLOR = {
  bg: "#FCFBF7",
  bgSoft: "#F3EFE8",
  text: "#17171C",
  muted: "#666773",
  line: "#D9D6CF",
  red: "#E2001A",
  redDark: "#B41222",
  blue: "#2E78B7",
  blueSoft: "#DDEAF8",
  sand: "#EFE3D3",
  card: "#FFFFFF",
  charcoal: "#262831",
  green: "#2C8F62",
};

const FONT = {
  title: "Avenir Next",
  body: "Avenir Next",
  mono: "Menlo",
};

async function readImageBlob(imagePath) {
  const bytes = await fs.readFile(imagePath);
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

async function saveBinary(outputPath, payload) {
  if (payload && typeof payload.save === "function") {
    await payload.save(outputPath);
    return;
  }
  if (payload instanceof Uint8Array) {
    await fs.writeFile(outputPath, payload);
    return;
  }
  if (payload instanceof ArrayBuffer) {
    await fs.writeFile(outputPath, Buffer.from(payload));
    return;
  }
  if (payload?.data instanceof Uint8Array) {
    await fs.writeFile(outputPath, payload.data);
    return;
  }
  if (payload?.data instanceof ArrayBuffer) {
    await fs.writeFile(outputPath, Buffer.from(payload.data));
    return;
  }
  if (payload && typeof payload.arrayBuffer === "function") {
    const ab = await payload.arrayBuffer();
    await fs.writeFile(outputPath, Buffer.from(ab));
    return;
  }
  throw new Error(`Unsupported binary payload for ${outputPath}`);
}

function setTextStyle(shape, {
  typeface = FONT.body,
  fontSize = 24,
  color = COLOR.text,
  bold = false,
  alignment = "left",
  verticalAlignment = "top",
  insets = { left: 10, right: 10, top: 6, bottom: 6 },
} = {}) {
  shape.text.typeface = typeface;
  shape.text.fontSize = fontSize;
  shape.text.color = color;
  shape.text.bold = bold;
  shape.text.alignment = alignment;
  shape.text.verticalAlignment = verticalAlignment;
  shape.text.insets = insets;
}

function addText(slide, text, left, top, width, height, opts = {}) {
  const shape = slide.shapes.add({
    geometry: "rect",
    position: { left, top, width, height },
    fill: "#FFFFFF00",
    line: { width: 0, fill: "#FFFFFF00" },
  });
  shape.text = text;
  setTextStyle(shape, opts);
  return shape;
}

function addBox(slide, left, top, width, height, {
  fill = COLOR.card,
  radius = 0.08,
  lineFill = COLOR.line,
  lineWidth = 1,
} = {}) {
  const shape = slide.shapes.add({
    geometry: "roundRect",
    adjustmentList: [{ name: "adj", formula: `val ${Math.round(radius * 50000)}` }],
    position: { left, top, width, height },
    fill,
    line: { style: "solid", fill: lineFill, width: lineWidth },
  });
  return shape;
}

function addPill(slide, label, left, top, width, {
  fill = COLOR.blueSoft,
  color = COLOR.blue,
} = {}) {
  const pill = addBox(slide, left, top, width, 34, {
    fill,
    lineFill: fill,
    lineWidth: 1,
    radius: 0.6,
  });
  pill.text = label;
  setTextStyle(pill, {
    typeface: FONT.body,
    fontSize: 16,
    color,
    bold: true,
    alignment: "center",
    verticalAlignment: "middle",
    insets: { left: 8, right: 8, top: 2, bottom: 2 },
  });
  return pill;
}

function addSectionTitle(slide, kicker, title, subtitle = null, page = null) {
  addText(slide, kicker, 70, 44, 260, 24, {
    fontSize: 16,
    bold: true,
    color: COLOR.red,
  });
  addText(slide, title, 70, 68, 820, 76, {
    typeface: FONT.title,
    fontSize: 34,
    bold: true,
    color: COLOR.text,
  });
  if (subtitle) {
    addText(slide, subtitle, 70, 138, 860, 34, {
      fontSize: 18,
      color: COLOR.muted,
    });
  }
  if (page !== null) {
    addPill(slide, String(page), 1160, 48, 44, {
      fill: COLOR.sand,
      color: COLOR.red,
    });
  }
}

function addBulletList(slide, items, left, top, width, lineHeight = 34, opts = {}) {
  items.forEach((item, idx) => {
    addText(slide, "•", left, top + idx * lineHeight, 20, lineHeight, {
      fontSize: opts.fontSize || 22,
      color: opts.bulletColor || COLOR.red,
      bold: true,
    });
    addText(slide, item, left + 22, top + idx * lineHeight, width - 22, lineHeight + 8, {
      fontSize: opts.fontSize || 22,
      color: opts.color || COLOR.text,
      bold: opts.bold || false,
    });
  });
}

function addFooter(slide, label = "MARL2Grid topology research plan") {
  slide.shapes.add({
    geometry: "rect",
    position: { left: 70, top: 688, width: 1140, height: 1 },
    fill: COLOR.line,
    line: { width: 0, fill: COLOR.line },
  });
  addText(slide, label, 70, 694, 500, 18, {
    fontSize: 12,
    color: COLOR.muted,
  });
}

async function build() {
  await fs.mkdir(OUT_DIR, { recursive: true });
  await fs.mkdir(TMP_DIR, { recursive: true });

  const presentation = Presentation.create({
    slideSize: { width: W, height: H },
  });

  presentation.theme.colorScheme = {
    name: "Marl2GridTheme",
    themeColors: {
      accent1: COLOR.red,
      accent2: COLOR.blue,
      accent3: COLOR.sand,
      bg1: COLOR.bg,
      bg2: "#FFFFFF",
      tx1: COLOR.text,
      tx2: COLOR.muted,
    },
  };

  const graphBlob = await readImageBlob(path.join(MEDIA_DIR, "image4.png"));
  const logoBlob = await readImageBlob(path.join(MEDIA_DIR, "image1.png"));

  // Slide 1
  {
    const slide = presentation.slides.add();
    slide.background.fill = COLOR.bg;

    slide.shapes.add({
      geometry: "rect",
      position: { left: 0, top: 0, width: W, height: H },
      fill: COLOR.bg,
      line: { width: 0, fill: COLOR.bg },
    });

    slide.shapes.add({
      geometry: "rect",
      position: { left: 0, top: 0, width: 18, height: H },
      fill: COLOR.red,
      line: { width: 0, fill: COLOR.red },
    });

    addPill(slide, "Topology Task", 74, 48, 132, { fill: COLOR.sand, color: COLOR.red });
    addPill(slide, "MARL2Grid", 216, 48, 114, { fill: COLOR.blueSoft, color: COLOR.blue });
    addPill(slide, "Multi-Agent Thesis Plan", 340, 48, 214, { fill: COLOR.sand, color: COLOR.charcoal });

    addText(
      slide,
      "Research Plan:",
      72,
      116,
      420,
      52,
      { typeface: FONT.title, fontSize: 26, bold: true, color: COLOR.red }
    );
    addText(
      slide,
      "Building on MARL2Grid for\nMulti-Agent Topology Control",
      72,
      160,
      600,
      160,
      { typeface: FONT.title, fontSize: 42, bold: true, color: COLOR.text, insets: { left: 0, right: 0, top: 0, bottom: 0 } }
    );
    addText(
      slide,
      "From benchmark understanding to new decentralized learning methods for transmission-grid operation",
      74,
      334,
      520,
      60,
      { fontSize: 22, color: COLOR.muted }
    );

    const statement = addBox(slide, 72, 436, 540, 124, {
      fill: "#FFFFFF",
      lineFill: COLOR.line,
      lineWidth: 1,
      radius: 0.2,
    });
    statement.text = "Core thesis idea\nUse MARL2Grid's topology benchmark as the testbed, then study how graph structure, coordination, and safe local control improve multi-agent performance.";
    setTextStyle(statement, {
      fontSize: 22,
      color: COLOR.text,
      bold: false,
      insets: { left: 18, right: 18, top: 16, bottom: 16 },
    });
    statement.text.get("Core thesis idea").bold = true;
    statement.text.get("Core thesis idea").color = COLOR.red;

    addText(slide, "Corentin Plumet", 74, 612, 260, 24, { fontSize: 20, bold: true });
    addText(slide, "Master Thesis Proposal", 74, 640, 280, 22, { fontSize: 17, color: COLOR.muted });
    addText(slide, "April 2026", 74, 665, 150, 22, { fontSize: 16, color: COLOR.muted });

    const heroCard = addBox(slide, 710, 104, 500, 522, {
      fill: "#FFFFFF",
      lineFill: COLOR.line,
      lineWidth: 1,
      radius: 0.12,
    });
    const heroImage = slide.images.add({
      blob: graphBlob,
      fit: "contain",
      alt: "Graph topology visual from prior deck",
    });
    heroImage.position = { left: 734, top: 146, width: 452, height: 382 };

    addText(slide, "Benchmark-first thesis framing", 738, 546, 320, 34, {
      fontSize: 17,
      bold: true,
      color: COLOR.red,
    });
    addText(
      slide,
      "Instead of proposing a controller in isolation, the plan is to start from MARL2Grid's released topology benchmark and ask what genuinely helps multi-agent control scale.",
      738,
      580,
      430,
      90,
      { fontSize: 16, color: COLOR.text }
    );

    const logo = slide.images.add({
      blob: logoBlob,
      fit: "contain",
      alt: "EPFL logo",
    });
    logo.position = { left: 1034, top: 640, width: 150, height: 34 };

    addFooter(slide, "Research plan update: MARL2Grid-centered and explicitly multi-agent");
  }

  // Slide 2
  {
    const slide = presentation.slides.add();
    slide.background.fill = COLOR.bg;
    addSectionTitle(
      slide,
      "Why this direction now",
      "Operational relevance meets a benchmark that actually exposes the MARL problem",
      "The opportunity is not just to do graph RL on grids, but to study decentralized topology control under realistic coordination constraints.",
      2
    );

    addBox(slide, 70, 206, 356, 382, { fill: "#FFFFFF", lineFill: COLOR.line, lineWidth: 1, radius: 0.14 });
    addText(slide, "Operational pressure", 92, 228, 220, 28, { fontSize: 20, bold: true, color: COLOR.red });
    addBulletList(slide, [
      "Topology control can reroute flows before using costly redispatch.",
      "Decisions must be made fast, under renewable variability and contingencies.",
      "Local interventions can create distant effects, so coordination matters by design.",
      "The action space is combinatorial and safety-critical."
    ], 92, 270, 300, 72, { fontSize: 18 });

    addBox(slide, 462, 206, 356, 382, { fill: "#FFFFFF", lineFill: COLOR.line, lineWidth: 1, radius: 0.14 });
    addText(slide, "Why MARL2Grid matters", 484, 228, 260, 28, { fontSize: 20, bold: true, color: COLOR.red });
    addBulletList(slide, [
      "It formulates realistic topology control as a multi-agent benchmark, not as a toy decomposition.",
      "It exposes agent partitions, local vs global observations, heuristic transitions, and constraints.",
      "It already provides baselines and released code on the topology side."
    ], 484, 270, 300, 84, { fontSize: 18 });

    addBox(slide, 854, 206, 356, 382, { fill: COLOR.red, lineFill: COLOR.red, lineWidth: 1, radius: 0.14 });
    addText(slide, "Thesis stance", 878, 228, 220, 28, { fontSize: 20, bold: true, color: "#FFFFFF" });
    addText(
      slide,
      "Build on MARL2Grid instead of inventing a private evaluation setup.\n\nUse the benchmark to ask a sharper question:\nWhat helps decentralized topology controllers coordinate, generalize, and remain reliable?",
      878,
      276,
      290,
      220,
      { fontSize: 24, color: "#FFFFFF", bold: false }
    );
    addPill(slide, "benchmark realism", 878, 520, 146, { fill: "#FFFFFF22", color: "#FFFFFF" });
    addPill(slide, "decentralized control", 1032, 520, 156, { fill: "#FFFFFF22", color: "#FFFFFF" });

    addFooter(slide);
  }

  // Slide 3
  {
    const slide = presentation.slides.add();
    slide.background.fill = COLOR.bg;
    addSectionTitle(
      slide,
      "Reframing the thesis",
      "The original plan becomes stronger if we move from a single-agent reliability story to a MARL2Grid-driven coordination story",
      null,
      3
    );

    addBox(slide, 86, 188, 468, 414, { fill: "#FFFFFF", lineFill: COLOR.line, lineWidth: 1, radius: 0.14 });
    addText(slide, "Previous framing", 112, 212, 220, 28, { fontSize: 20, bold: true, color: COLOR.muted });
    addBulletList(slide, [
      "Main focus on graph representation for one controller.",
      "Reliability is an evaluation lens layered on top of standard RL.",
      "Safety layers and imitation are extra modules around a single decision-maker.",
      "Benchmarking risk: harder to separate method gains from evaluation choices."
    ], 112, 256, 390, 80, { fontSize: 20, bulletColor: COLOR.muted, color: COLOR.text });

    slide.shapes.add({
      geometry: "rightArrow",
      position: { left: 575, top: 336, width: 126, height: 106 },
      fill: COLOR.red,
      line: { width: 0, fill: COLOR.red },
    });
    addText(slide, "shift the center of gravity", 566, 450, 150, 48, {
      fontSize: 16,
      color: COLOR.red,
      alignment: "center",
      bold: true,
    });

    addBox(slide, 726, 188, 468, 414, { fill: COLOR.blueSoft, lineFill: COLOR.blueSoft, lineWidth: 1, radius: 0.14 });
    addText(slide, "Updated framing", 752, 212, 220, 28, { fontSize: 20, bold: true, color: COLOR.blue });
    addBulletList(slide, [
      "Start from MARL2Grid's topology benchmark and released baselines.",
      "Center the thesis on coordination, partial observability, and decentralized action structure.",
      "Use graph structure as a tool for multi-agent representation and communication.",
      "Treat reliability as a first-class benchmark outcome, not just an add-on metric."
    ], 752, 256, 390, 80, { fontSize: 20, bulletColor: COLOR.blue, color: COLOR.text });

    addBox(slide, 160, 620, 968, 58, { fill: COLOR.sand, lineFill: COLOR.sand, lineWidth: 1, radius: 0.2 });
    addText(
      slide,
      "Net effect: the project becomes more original, easier to justify experimentally, and much better aligned with the MARL2Grid paper and codebase.",
      184,
      632,
      920,
      32,
      { fontSize: 20, bold: true, color: COLOR.text, alignment: "center", verticalAlignment: "middle" }
    );

    addFooter(slide);
  }

  // Slide 4
  {
    const slide = presentation.slides.add();
    slide.background.fill = COLOR.bg;
    addSectionTitle(
      slide,
      "What MARL2Grid gives us",
      "The topology side already defines a complete experimental stack for decentralized power-grid control",
      null,
      4
    );

    const columns = [
      { x: 70, title: "Environment", lines: ["Grid2Op-based topology task", "bus14, bus36, bus118", "5-minute control steps", "Substation-level topology actions"] },
      { x: 382, title: "Multi-agent setup", lines: ["Agent-to-substation partitions", "Difficulty 0: paper partitions", "Difficulty 1: one agent per substation", "Local or global observations"] },
      { x: 694, title: "Training stack", lines: ["MAPPO, QPLEX, LagrMAPPO", "Async vectorized env wrapper", "Evaluation hooks + checkpoints", "Paper-aligned reward components"] },
      { x: 1006, title: "Safety & realism", lines: ["Idle heuristic", "Constraint variants", "Line overload logic", "Long-horizon chronics"] },
    ];

    columns.forEach((col) => {
      addBox(slide, col.x, 210, 244, 314, { fill: "#FFFFFF", lineFill: COLOR.line, lineWidth: 1, radius: 0.16 });
      addText(slide, col.title, col.x + 18, 232, 180, 28, { fontSize: 20, bold: true, color: COLOR.red });
      addBulletList(slide, col.lines, col.x + 18, 276, 208, 54, { fontSize: 18 });
    });

    addBox(slide, 96, 560, 1088, 90, { fill: COLOR.blueSoft, lineFill: COLOR.blueSoft, lineWidth: 1, radius: 0.18 });
    addText(slide, "Implication for the thesis", 120, 580, 220, 28, { fontSize: 20, bold: true, color: COLOR.blue });
    addText(
      slide,
      "The benchmark is not just a dataset or simulator. It already defines the decomposition, observability, and evaluation protocol, which lets the thesis focus on method design rather than benchmark creation.",
      330,
      576,
      820,
      40,
      { fontSize: 20, color: COLOR.text, verticalAlignment: "middle" }
    );

    addFooter(slide);
  }

  // Slide 5
  {
    const slide = presentation.slides.add();
    slide.background.fill = COLOR.bg;
    addSectionTitle(
      slide,
      "Research gap",
      "MARL2Grid already shows the problem is not solved, especially for realistic topology control",
      null,
      5
    );

    addBox(slide, 70, 198, 1140, 90, { fill: COLOR.red, lineFill: COLOR.red, lineWidth: 1, radius: 0.16 });
    addText(
      slide,
      "Key signal from the paper: on the topology task, MAPPO performs well on bus14 but current baselines still fail to scale cleanly to harder settings like bus118.",
      96,
      224,
      1088,
      44,
      { fontSize: 26, color: "#FFFFFF", bold: true, alignment: "center", verticalAlignment: "middle" }
    );

    const gaps = [
      ["Coordination", "Agents act locally but affect distant lines through shared AC dynamics. That makes naive decentralization brittle."],
      ["Representation", "Per-agent MLP observations may discard useful graph structure about neighboring substations and boundary lines."],
      ["Exploration", "Topology actions are combinatorial, and the idle heuristic can reduce the already narrow windows where agents actually learn to coordinate."],
      ["Evaluation", "A good thesis must compare methods across agent partitions, observation regimes, constraints, and scaling levels, not only one favorable setup."],
    ];

    gaps.forEach((gap, idx) => {
      const x = 70 + (idx % 2) * 572;
      const y = 326 + Math.floor(idx / 2) * 162;
      addBox(slide, x, y, 548, 136, { fill: "#FFFFFF", lineFill: COLOR.line, lineWidth: 1, radius: 0.16 });
      addText(slide, gap[0], x + 20, y + 18, 180, 28, { fontSize: 20, bold: true, color: COLOR.red });
      addText(slide, gap[1], x + 20, y + 50, 504, 70, { fontSize: 18, color: COLOR.text });
    });

    addFooter(slide);
  }

  // Slide 6
  {
    const slide = presentation.slides.add();
    slide.background.fill = COLOR.bg;
    addSectionTitle(
      slide,
      "Objective and questions",
      "The thesis should ask what makes decentralized topology control work better on MARL2Grid, not only whether a graph encoder helps",
      null,
      6
    );

    addBox(slide, 70, 204, 1140, 110, { fill: COLOR.sand, lineFill: COLOR.sand, lineWidth: 1, radius: 0.2 });
    addText(slide, "Main objective", 92, 226, 180, 28, { fontSize: 20, bold: true, color: COLOR.red });
    addText(
      slide,
      "Develop and evaluate graph-aware multi-agent methods for MARL2Grid topology control, with a focus on coordination, safe local decision-making, and scaling behavior.",
      280,
      222,
      900,
      48,
      { fontSize: 24, bold: true, color: COLOR.text, verticalAlignment: "middle" }
    );

    const qs = [
      "Can graph-aware local policies outperform vanilla MAPPO/QPLEX on the topology benchmark?",
      "Which decentralization choices matter most: agent partition, observation scope, or explicit communication?",
      "How should safe action filtering or heuristic guidance be integrated without harming learning?",
      "What evaluation protocol best captures reliability, margin management, and scalability across bus14, bus36, and bus118?"
    ];

    qs.forEach((q, idx) => {
      const y = 352 + idx * 82;
      addBox(slide, 94, y, 72, 54, { fill: COLOR.red, lineFill: COLOR.red, lineWidth: 1, radius: 0.5 });
      addText(slide, String(idx + 1), 94, y + 9, 72, 36, {
        fontSize: 24,
        bold: true,
        color: "#FFFFFF",
        alignment: "center",
        verticalAlignment: "middle",
      });
      addBox(slide, 186, y, 994, 54, { fill: "#FFFFFF", lineFill: COLOR.line, lineWidth: 1, radius: 0.2 });
      addText(slide, q, 204, y + 10, 960, 36, { fontSize: 21, color: COLOR.text, verticalAlignment: "middle" });
    });

    addFooter(slide);
  }

  // Slide 7
  {
    const slide = presentation.slides.add();
    slide.background.fill = COLOR.bg;
    addSectionTitle(
      slide,
      "Proposed method stack",
      "A layered approach that starts simple, then adds graph structure, coordination, and safety",
      null,
      7
    );

    const boxes = [
      { x: 90, y: 228, w: 250, h: 120, title: "1. Benchmark layer", body: "MARL2Grid topology env\nfixed partitions\nlocal/global observations\npaper baselines", fill: COLOR.sand, color: COLOR.text },
      { x: 374, y: 228, w: 250, h: 120, title: "2. Representation layer", body: "substation + line encoder\nshared or partially shared\nboundary-aware context", fill: "#FFFFFF", color: COLOR.text },
      { x: 658, y: 228, w: 250, h: 120, title: "3. Coordination layer", body: "centralized critic\noptional communication\nneighbor-zone reasoning", fill: COLOR.blueSoft, color: COLOR.text },
      { x: 942, y: 228, w: 250, h: 120, title: "4. Safe action layer", body: "local proposal + pruning\nsymmetry reduction\nfallbacks for risky actions", fill: "#FFFFFF", color: COLOR.text },
    ];

    boxes.forEach((box, idx) => {
      addBox(slide, box.x, box.y, box.w, box.h, {
        fill: box.fill,
        lineFill: box.fill === "#FFFFFF" ? COLOR.line : box.fill,
        lineWidth: 1,
        radius: 0.14,
      });
      addText(slide, box.title, box.x + 16, box.y + 14, box.w - 32, 24, {
        fontSize: 17,
        bold: true,
        color: idx === 0 ? COLOR.red : idx === 2 ? COLOR.blue : COLOR.text,
      });
      addText(slide, box.body, box.x + 16, box.y + 42, box.w - 32, 70, {
        fontSize: 15,
        color: box.color,
      });
      if (idx < boxes.length - 1) {
        slide.shapes.add({
          geometry: "rightArrow",
          position: { left: box.x + box.w + 12, top: box.y + 34, width: 24, height: 36 },
          fill: COLOR.red,
          line: { width: 0, fill: COLOR.red },
        });
      }
    });

    addBox(slide, 140, 402, 1000, 178, { fill: COLOR.charcoal, lineFill: COLOR.charcoal, lineWidth: 1, radius: 0.18 });
    addText(slide, "Execution plan", 166, 426, 220, 28, { fontSize: 20, bold: true, color: "#FFFFFF" });
    addText(
      slide,
      "Milestone A: reproduce MARL2Grid topology baselines and evaluation.\nMilestone B: replace the local MLP policy with a graph-aware multi-agent encoder.\nMilestone C: study explicit coordination and safe local action selection.\nMilestone D: evaluate across partitions, observability regimes, and grid scales.",
      166,
      464,
      920,
      104,
      { fontSize: 21, color: "#FFFFFF" }
    );

    addFooter(slide);
  }

  // Slide 8
  {
    const slide = presentation.slides.add();
    slide.background.fill = COLOR.bg;
    addSectionTitle(
      slide,
      "Experimental plan",
      "The contribution should be visible both in method design and in how rigorously the benchmark is used",
      null,
      8
    );

    const rows = [
      ["Baselines", "Single-agent PPO, MAPPO, QPLEX, LagrMAPPO from the MARL2Grid codebase"],
      ["Method variants", "MLP vs graph encoder, shared vs agent-specific parameters, no-comm vs comm, with/without safety filtering"],
      ["Benchmark axes", "bus14 → bus36 → bus118, fixed partitions vs one-agent-per-substation, decentralized vs global observations"],
      ["Metrics", "survival, line margin, overload cost, topology distance, seed variance, and sample efficiency"],
    ];

    rows.forEach((row, idx) => {
      const y = 210 + idx * 92;
      addBox(slide, 86, y, 214, 68, { fill: idx % 2 === 0 ? COLOR.sand : COLOR.blueSoft, lineFill: idx % 2 === 0 ? COLOR.sand : COLOR.blueSoft, lineWidth: 1, radius: 0.14 });
      addText(slide, row[0], 104, y + 19, 178, 24, {
        fontSize: 19,
        bold: true,
        color: idx % 2 === 0 ? COLOR.red : COLOR.blue,
        alignment: "center",
        verticalAlignment: "middle",
      });
      addBox(slide, 322, y, 872, 68, { fill: "#FFFFFF", lineFill: COLOR.line, lineWidth: 1, radius: 0.14 });
      addText(slide, row[1], 344, y + 14, 828, 36, { fontSize: 19, color: COLOR.text, verticalAlignment: "middle" });
    });

    addBox(slide, 86, 590, 1108, 64, { fill: COLOR.red, lineFill: COLOR.red, lineWidth: 1, radius: 0.2 });
    addText(
      slide,
      "Success criterion: not only better peak performance on bus14, but a better scaling trend and clearer coordination behavior as the task becomes harder.",
      112,
      606,
      1056,
      28,
      { fontSize: 22, color: "#FFFFFF", bold: true, alignment: "center", verticalAlignment: "middle" }
    );

    addFooter(slide);
  }

  // Slide 9
  {
    const slide = presentation.slides.add();
    slide.background.fill = COLOR.bg;
    addSectionTitle(
      slide,
      "Timeline",
      "A realistic thesis schedule should move from benchmark mastery to method contribution, then to evaluation",
      null,
      9
    );

    const milestones = [
      ["April", "Reproduce topology baselines", "Audit the MARL2Grid code,\nrun the reference baselines,\nand lock an evaluation protocol."],
      ["May", "Formalize graph MARL setup", "Define state/action abstractions,\npick graph encoders,\nand choose the first MARL variant."],
      ["June", "Implement graph-aware baseline", "Integrate the first graph-based\nmulti-agent controller\nand compare against MAPPO/QPLEX."],
      ["July", "Coordination + safety ablations", "Add communication or\ncoordination modules,\nplus filtering or pruning studies."],
      ["August", "Final evaluation and writing", "Run the full benchmark matrix,\nsynthesize the results,\nand write the thesis narrative."],
    ];

    slide.shapes.add({
      geometry: "rect",
      position: { left: 149, top: 364, width: 936, height: 6 },
      fill: COLOR.line,
      line: { width: 0, fill: COLOR.line },
    });

    milestones.forEach((m, idx) => {
      const cardLeft = 54 + idx * 234;
      const cx = cardLeft + 95;
      slide.shapes.add({
        geometry: "ellipse",
        position: { left: cx - 20, top: 346, width: 40, height: 40 },
        fill: COLOR.red,
        line: { width: 0, fill: COLOR.red },
      });
      addText(slide, String(idx + 1), cx - 20, 354, 40, 22, {
        fontSize: 18,
        bold: true,
        color: "#FFFFFF",
        alignment: "center",
        verticalAlignment: "middle",
      });
      addPill(slide, m[0], cx - 40, 300, 80, { fill: COLOR.sand, color: COLOR.red });
      addBox(slide, cardLeft, 408, 190, 170, { fill: "#FFFFFF", lineFill: COLOR.line, lineWidth: 1, radius: 0.14 });
      addText(slide, m[1], cardLeft + 18, 426, 154, 72, { fontSize: 16, bold: true, color: COLOR.text, alignment: "center" });
      addText(slide, m[2], cardLeft + 16, 500, 158, 74, { fontSize: 13.5, color: COLOR.muted, alignment: "center" });
    });

    addFooter(slide);
  }

  // Slide 10
  {
    const slide = presentation.slides.add();
    slide.background.fill = COLOR.bg;
    addSectionTitle(
      slide,
      "Expected contributions",
      "A strong thesis outcome combines a clean benchmark story, a method contribution, and reusable evaluation practice",
      null,
      10
    );

    addBox(slide, 70, 204, 516, 372, { fill: COLOR.red, lineFill: COLOR.red, lineWidth: 1, radius: 0.18 });
    addText(slide, "What the thesis should deliver", 96, 230, 280, 28, { fontSize: 22, bold: true, color: "#FFFFFF" });
    addBulletList(slide, [
      "A MARL2Grid-centered research pipeline for topology control.",
      "At least one graph-aware multi-agent baseline that is stronger than the released reference methods.",
      "A clear ablation map of what helps: graph structure, coordination, observability, or safety logic.",
      "A reusable evaluation protocol for future MARL topology work."
    ], 98, 278, 440, 66, { fontSize: 20, bulletColor: "#FFFFFF", color: "#FFFFFF" });

    addBox(slide, 620, 204, 590, 372, { fill: "#FFFFFF", lineFill: COLOR.line, lineWidth: 1, radius: 0.18 });
    addText(slide, "Anchor references", 646, 230, 220, 28, { fontSize: 22, bold: true, color: COLOR.red });
    addText(
      slide,
      "Marchesini et al. (2026) MARL2Grid-TR\nYu et al. (2022) MAPPO\nWang et al. (2021) QPLEX\nMarchesini et al. (2025) RL2Grid\nRecent graph-RL / multi-agent topology-control papers from the original plan",
      646,
      276,
      520,
      180,
      { fontSize: 19, color: COLOR.text }
    );
    addBox(slide, 646, 462, 520, 88, { fill: COLOR.blueSoft, lineFill: COLOR.blueSoft, lineWidth: 1, radius: 0.16 });
    addText(
      slide,
      "The proposal becomes sharper if the benchmark is the backbone and the method question is explicitly multi-agent.",
      670,
      483,
      472,
      48,
      { fontSize: 18, bold: true, color: COLOR.blue, verticalAlignment: "middle" }
    );

    addFooter(slide, "Updated deck built from the original research plan and redirected toward MARL2Grid");
  }

  // Render previews for quick inspection
  for (const [idx, slide] of presentation.slides.items.entries()) {
    const png = await presentation.export({ slide, format: "png", scale: 1 });
    await saveBinary(path.join(TMP_DIR, `slide_${String(idx + 1).padStart(2, "0")}.png`), png);
  }

  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(FILE_OUT);
  console.log(FILE_OUT);
}

await build();
