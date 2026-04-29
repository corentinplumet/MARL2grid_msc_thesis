import path from 'node:path';
import { FileBlob, PresentationFile } from '@oai/artifact-tool';
import fs from 'node:fs/promises';
const root='/Users/corentinplumet/Documents/ICLR_marl2grid/Topology_Task/Topology_Task';
const input=path.join(root,'msc_thesis_MARL.pptx');
const outDir=path.join(root,'tmp','slides','msc_thesis_MARL_update','existing');
await fs.mkdir(outDir,{recursive:true});
const file=await FileBlob.load(input);
const pres=await PresentationFile.importPptx(file);
for (const idx of [6,8,9]) {
  const slide=pres.slides.getItem(idx);
  const png=await pres.export({slide, format:'png', scale:1});
  const ab=await png.arrayBuffer();
  await fs.writeFile(path.join(outDir,`slide_${idx+1}.png`), Buffer.from(ab));
}
console.log(outDir);
