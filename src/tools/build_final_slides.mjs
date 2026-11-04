/* Build the separate, editable final-results deck from frozen project results. */
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const SKILL = 'C:/Users/zou_r/.codex/plugins/cache/openai-primary-runtime/presentations/26.1026.10807/skills/presentations';
const RUNTIME = 'C:/Users/zou_r/.cache/codex-runtimes/codex-primary-runtime/dependencies';
process.env.RUNTIME_NODE_MODULES = path.join(RUNTIME, 'node/node_modules');
const { Presentation, PresentationFile } = await import(pathToFileURL(path.join(RUNTIME, 'node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs')).href);
const { finalizePresentation } = await import(pathToFileURL(path.join(SKILL, 'container_tools/artifact_tool_utils.mjs')).href);

const BUILD = path.join(ROOT, 'output/runs/tmp/slides_build');
const FINAL = path.join(ROOT, 'output/final_slides/AI6124_Group1_Final_Results_2026-11-04_v2.pptx');
await fs.mkdir(BUILD, { recursive: true });
await fs.mkdir(path.dirname(FINAL), { recursive: true });
const C = { navy:'#102e42', teal:'#087c84', muted:'#526a78', light:'#e8f0f1', gold:'#c9782e', white:'#ffffff', red:'#b64343' };
const deck = Presentation.create({slideSize:{width:1280,height:720}});

function box(slide, x,y,w,h, fill='none') {
  return slide.shapes.add({geometry:'rect',position:{left:x,top:y,width:w,height:h},fill,
    line:{style:'solid',fill:'none',width:0}});
}
function txt(slide, value, x,y,w,h, size=22, color=C.navy, bold=false) {
  const s=box(slide,x,y,w,h);
  s.text=value;
  s.text.style={fontFamily:'Aptos',fontSize:size,color,bold,verticalAlignment:'middle'};
  return s;
}
function base(title, n, note='') {
  const slide=deck.slides.add();
  slide.background.fill=C.white;
  txt(slide,title,56,34,1160,64,40,C.navy,true);
  box(slide,56,106,1168,3,C.teal);
  txt(slide,String(n).padStart(2,'0'),1172,42,55,32,20,C.teal,true);
  slide.speakerNotes.text=note;
  return slide;
}
async function img(slide, relative, x,y,w,h, alt) {
  const b=await fs.readFile(path.join(ROOT,relative));
  const blob=b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength);
  slide.images.add({blob,contentType:'image/png',alt,fit:'contain',
    position:{left:x,top:y,width:w,height:h}});
}
function bullet(slide, items, x,y,w, size=22, gap=58) {
  items.forEach((value,i)=>{
    box(slide,x,y+i*gap+9,7,7,C.teal);
    txt(slide,value,x+20,y+i*gap,w-20,gap-4,size,C.navy);
  });
}
function table(slide, values, x,y,w,h, widths=undefined) {
  const t=slide.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,
    values,...(widths?{columnWidths:widths}:{})});
  t.styleOptions={headerRow:true,bandedRows:true};
  t.borders.assign({style:'solid',fill:'#d8e2e5',width:1});
  for(let c=0;c<values[0].length;c++) t.getCell(0,c).fill=C.light;
  return t;
}

// 1: No empty cover. Establish the study and its actual outcome.
{
  const s=base('Singapore REIT ETF forecasting with fuzzy experts',1,
    'Project source: docs/experiment_protocol_lstm_fuzzy_local_experts.md; docs/final_results_2026-11-04.md.');
  txt(s,'Zou Runze (Ryan)  ·  Ngeow Shan Yong Destin',60,127,950,42,23,C.muted);
  txt(s,'Can learned sequence features support a small, interpretable fuzzy predictor?',60,215,1100,90,32,C.navy,true);
  txt(s,'SRT.SI next-day raw close return  •  20 trading days  •  SRT, SORA and STI inputs',60,330,1120,54,24,C.muted);
  box(s,60,425,1120,2,C.light);
  txt(s,'Frozen test: Fuzzy2 RMSE 0.006349 vs unchanged price 0.006377',60,452,1120,52,27,C.teal,true);
  txt(s,'Trading wealth after 10 bps costs: 0.999 vs cash 1.000',60,520,1120,52,27,C.navy,true);
  txt(s,'The error gain is small; the rules do not form clear market specialists.',60,600,1120,38,22,C.muted);
}
// 2: Data and timing.
{
  const s=base('Data and chronological evaluation',2,
    'Sources: data/source_manifest.json; docs/experiment_protocol_lstm_fuzzy_local_experts.md; output/development/final_configuration_pretest.json.');
  txt(s,'SRT.SI is a traded Singapore REIT ETF. The target is its next raw closing-price log return.',60,135,1140,68,24,C.navy);
  txt(s,'Inputs available after close t',60,237,470,38,27,C.teal,true);
  bullet(s,['SRT return, five-day momentum, 20-day volatility','Published SORA level and change','STI return aligned to available date'],60,292,535,22,72);
  txt(s,'Forward-time splits',650,237,470,38,27,C.teal,true);
  bullet(s,['Three expanding development folds: 378 target days','Final holdout: 251 days, Oct 2025–Sep 2026','All scalers fit on each training segment only'],650,292,535,22,72);
  box(s,60,566,1120,2,C.light);
  txt(s,'One missing SRT open (24 Oct 2025): carry cash and shares; make no trade at that open.',60,588,1120,54,22,C.muted);
}
// 3: Architecture as editable engineering diagram.
{
  const s=base('LSTM features feed two fuzzy local experts',3,
    'Model source: src/models.py and docs/experiment_protocol_lstm_fuzzy_local_experts.md. The original proposal deck mentions K-means and auxiliary losses; the frozen implementation uses fixed initial centers and global MSE.');
  const stages=[['20 × 6 history','SRT · SORA · STI'],['LSTM encoder','32 hidden units'],['3D projection','tanh latent z'],['Gaussian rules','two soft weights α'],['Affine experts','two returns e₁,e₂']];
  stages.forEach(([head,sub],i)=>{
    const x=55+i*236;
    box(s,x,245,210,114,i===3?C.light:'#eef3f4');
    txt(s,head,x+12,259,188,40,21,C.navy,true);
    txt(s,sub,x+12,302,188,43,18,C.muted);
    if(i<4) box(s,x+211,300,25,3,C.teal);
  });
  txt(s,'ŷ = α₁e₁(z) + α₂e₂(z)',155,415,970,68,38,C.teal,true);
  bullet(s,['Centers start at ±0.25 on the first latent coordinate; scales are learned.','AdamW minimizes global MSE; no attention, ARIX, K-means or auxiliary rule loss.'],100,525,1080,21,59);
}
// 4: Fixed experimental design.
{
  const s=base('Matched models and frozen training choices',4,
    'Sources: docs/development_results_2026-11-03.md; output/development/final_configuration_pretest.json; output/development/run_resources.csv.');
  table(s,[['Reference / model','What it isolates','Final epochs'],
           ['Z0','Unchanged closing price','—'],['LSTM32','Ordinary LSTM head','1'],
           ['LSTM3','Three-dimensional bottleneck','6'],['Fuzzy2','Gaussian two-rule head','3'],
           ['Gate2','Non-Gaussian two-expert gate','4']],70,155,1140,400,[300,650,190]);
  txt(s,'All models use the same six inputs, 931 pretest training windows and seeds 11, 23 and 37.',70,586,1135,46,21,C.navy);
  txt(s,'Epochs were fixed from nine development fits per model before opening the final 251 days.',70,636,1135,37,20,C.muted);
}
// 5: Forecast results.
{
  const s=base('Final forecast error: a small RMSE difference',5,
    'Sources: output/final/forecast_summary.csv; output/final/forecast_by_seed.csv; docs/final_results_2026-11-04.md. RMSE/MAE are raw log returns, averaged across seeds for learned models. Z0 is deterministic.');
  table(s,[['Model','RMSE','MAE'],['Fuzzy2','0.006349','0.004595'],
           ['Gate2','0.006353','0.004582'],['LSTM3','0.006355','0.004593'],
           ['Z0','0.006377','0.004547'],['LSTM32','0.006384','0.004651']],
        68,164,640,408,[250,195,195]);
  txt(s,'Fuzzy2 vs Z0',750,174,430,38,25,C.teal,true);
  txt(s,'0.44% lower RMSE',750,226,440,50,31,C.navy,true);
  txt(s,'Z0 still has lower MAE.',750,294,440,56,23,C.muted);
  txt(s,'Fuzzy2 direction accuracy: 49.0% on nonzero-return days.',750,387,435,83,21,C.navy);
  txt(s,'One held-out period; seeds share the same days.',750,490,435,70,21,C.muted);
  txt(s,'Development RMSE did not favor Fuzzy2 over Z0 or LSTM3.',68,614,1115,42,21,C.muted);
}
// 6: Trading result, chart from saved scientific plot.
{
  const s=base('Trading: mostly cash, no gain after costs',6,
    'Sources: output/final/wealth_comparison.csv; output/final/fuzzy_position_summary.csv; output/final/wealth_paths_price_only_10bps.png. The chart shows seed 11; table values are means over seeds. 10 bps per side; price-only wealth.');
  await img(s,'output/final/wealth_paths_price_only_10bps.png',45,141,788,469,
    'Final seed 11 price-only wealth paths at 10 basis points per side');
  txt(s,'End wealth, price-only',850,159,354,40,24,C.teal,true);
  txt(s,'Cash     1.000',850,218,345,39,25,C.navy,true);
  txt(s,'Fuzzy2  0.999',850,265,345,39,25,C.navy,true);
  txt(s,'Momentum  0.924',850,312,345,39,23,C.navy);
  txt(s,'Always long  0.878',850,359,345,39,23,C.navy);
  box(s,850,428,330,2,C.light);
  txt(s,'Two Fuzzy2 seeds made no trades. Seed 11 traded four times.',850,450,334,122,21,C.muted);
  txt(s,'Gross-distribution and 0/20-bps results are in the Notebook.',60,622,1150,38,20,C.muted);
}
// 7: Mechanism evidence.
{
  const s=base('Fuzzy rules remain close to an even split',7,
    'Sources: output/final/fuzzy_rules_contributions_seed11.png; output/final/fuzzy_local_checks.csv; output/final/fuzzy_case_examples_seed11.csv. The figure is a saved scientific plot for seed 11.');
  await img(s,'output/final/fuzzy_rules_contributions_seed11.png',50,142,760,482,
    'Fuzzy2 rule weights and affine expert contributions over the final holdout, seed 11');
  txt(s,'Final rule evidence',845,165,350,42,25,C.teal,true);
  bullet(s,['Mean maximum weight: 0.51–0.56','No day exceeds weight 0.8','One rule has zero high-activation cases in one seed'],845,231,345,21,82);
  txt(s,'Contributions are traceable. Bull, crash or rate labels are unsupported.',845,520,355,112,21,C.muted);
}
// 8: Research comparison and forward-looking question.
{
  const s=base('What the simplified study establishes',8,
    'Sources: https://arxiv.org/html/2510.00960v1; docs/literature_review_draft_en.md; docs/final_results_2026-11-04.md. The cited Fuzzformer study uses a different market, horizon, target and architecture; numeric RMSE values are not directly comparable.');
  txt(s,'Fuzzformer (2025)',60,154,530,42,26,C.teal,true);
  txt(s,'S&P 500, multi-horizon prices; LSTM + attention; 16 Gaussian rules; ARIX local models; local and clustering losses.',60,210,530,184,22,C.navy);
  txt(s,'This course adaptation',650,154,530,42,26,C.teal,true);
  txt(s,'SRT ETF, one-day returns; one LSTM; two Gaussian rules; affine experts; global MSE. Small RMSE gain, no clear rule specialization.',650,210,530,184,22,C.navy);
  box(s,60,414,1120,2,C.light);
  txt(s,'Next research question',60,446,1120,42,27,C.navy,true);
  txt(s,'Can local expert losses create stable, meaningful rule regions without harming accuracy?',60,500,1120,86,26,C.teal);
  txt(s,'Test on new periods or instruments; a building-demand application is specified in the Notebook.',60,610,1120,46,20,C.muted);
}

for(let i=0;i<deck.slides.items.length;i++){
  const png=await deck.export({slide:deck.slides.items[i],format:'png',scale:1});
  await fs.writeFile(path.join(BUILD,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await png.arrayBuffer()));
}
const candidate=path.join(BUILD,'candidate.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidate);
const result=await finalizePresentation({
  workspaceDir:ROOT,candidatePath:candidate,finalPath:FINAL,
  pythonExecutable:path.join(RUNTIME,'python/python.exe'),
  integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit','--require-native-table-slide','4','--require-native-table-slide','5'],
  explicitTotalSlideCount:8,requiredNativeTableOwnerSlides:[4,5],
  fontPolicy:{basis:'design',families:['Aptos']},verifyArtifactToolImport:true,
  receiptPath:path.join(BUILD,'validation_v2.json')
});
console.log(JSON.stringify({finalPath:FINAL,result},null,2));
