// Editable presentation from saved evidence. This imports no ML runtime.
// Run a byte-identical copy from a private build directory linked to the
// bundled Node modules. See docs/SLIDES_GUIDE.md for the documented workflow.
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {pathToFileURL} from 'node:url';
import {FileBlob, Presentation, PresentationFile} from '@oai/artifact-tool';

const args = Object.fromEntries(process.argv.slice(2).reduce((pairs, value, i, all) =>
  i % 2 === 0 ? [...pairs, [value.replace(/^--/, ''), all[i + 1]]] : pairs, []));
for (const key of ['repo', 'build', 'runtime-root', 'skill-dir']) {
  if (!args[key]) throw new Error('Missing --' + key);
}
const repo = path.resolve(args.repo);
const build = path.resolve(args.build);
const runtime = path.resolve(args['runtime-root']);
const skill = path.resolve(args['skill-dir']);
// The strict first-party import validator runs in a fresh subprocess.
process.env.RUNTIME_NODE_MODULES = path.join(runtime,'node','node_modules');
const revision = args.revision || 'v1';
const SNAPSHOT = '9d5b62173249829063219beac8c3aaf4cf3aa45b';
const BASE = 'https://github.com/TangWalden/CISC8006-MonTi/blob/' + SNAPSHOT + '/';
const {applyPresentationChartFont, finalizePresentation} = await import(
  pathToFileURL(path.join(skill, 'container_tools/artifact_tool_utils.mjs')).href);
const load = async relative => JSON.parse(await fs.readFile(path.join(repo, relative), 'utf8'));
const baseline = await load('evidence/reproduction_summary.json');
const pilots = await load('evidence/improvement_pilot_summary.json');
const ledger = await load('evidence/research_cost_ledger.json');
// Excel workbooks preserve at most 15 significant decimal digits. Use a
// declared presentation precision while leaving the scientific JSON unchanged.
const chartNumber = value => Number(value.toFixed(8));
const color = {ink:'#142E43', blue:'#206DA8', gray:'#9BAABA', orange:'#BD6425', muted:'#526678'};
const p = Presentation.create({slideSize:{width:1280,height:720}});
const schedule = [
  ['Junyi Tang',20],['Junyi Tang',55],['Junyi Tang',75],['Junyi Tang',30],
  ['Yifan Fan',65],['Yifan Fan',65],['Yifan Fan',50],
];
const metadata = [];

function text(slide, value, x, y, width, height, size=26, bold=false, fill=color.ink) {
  const shape = slide.shapes.add({geometry:'textbox', position:{left:x,top:y,width,height},
    fill:'none', line:{fill:'none',width:0}, name:value.slice(0,50)});
  shape.text = value;
  shape.text.style = {fontSize:size,typeface:'Arial',bold,color:fill,
    insets:0,wrap:'square',autoFit:'none',verticalAlignment:'top'};
  return shape;
}
function slide(title, summary, sources, talk, appendix=false) {
  const s = p.slides.add();
  s.background.fill = '#FFFFFF';
  const n = p.slides.items.length;
  text(s,title,64,42,1152,70,44,true);
  if(summary) text(s,summary,64,125,1152,72,25,false,color.muted);
  text(s,String(n),1150,681,66,23,16,false,color.muted);
  const plan = schedule[n-1];
  s.speakerNotes.text = (plan ? 'Planned speaker: '+plan[0]+'. Target: '+plan[1]+
    ' seconds. Rehearsal and actual contribution are not recorded.\n\n' :
    'Optional appendix. No assigned assessed-defense response.\n\n') + talk +
    '\n\nEvidence snapshot: '+SNAPSHOT+'\nSources:\n'+sources.map(v=>v.includes('://')?v:BASE+v).join('\n');
  metadata.push({number:n,title,appendix,planned_speaker:plan?.[0]??null,
    planned_seconds:plan?.[1]??null,sources});
  return s;
}
function foot(s,value){text(s,value,64,623,1152,49,19,false,color.muted);}
function table(s,values,x,y,width,height,columnWidths,size=24){
  const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width,height,
    columnWidths,values});
  t.styleOptions={headerRow:true,bandedRows:false};
  t.borders.assign({style:'solid',fill:'#D5DEE5',width:1});
  for(let r=0;r<values.length;r++){
    t.rows[r].height=height/values.length;
    for(let c=0;c<values[r].length;c++){
      const cell=t.getCell(r,c);
      cell.fill=r===0?'#E7EEF3':'#FFFFFF';
      cell.text.style={fontSize:r===0?size-1:size,typeface:'Arial',bold:r===0,
        color:color.ink,alignment:c===0?'left':'center',verticalAlignment:'middle',
        insets:{left:10,right:10,top:6,bottom:6},wrap:'square',autoFit:'none'};
    }
  }
  return t;
}
function chart(s,categories,series,position,axis,extra={}){
  const c=s.charts.add('bar',{
    position,categories,series,
    barOptions:{direction:'column',grouping:'clustered',gapWidth:90,overlap:0},
    hasLegend:series.length>1,
    legend:{position:'bottom',overlay:false,textStyle:{fontSize:21,fill:color.ink}},
    xAxis:{textStyle:{fontSize:22,fill:color.ink},line:{fill:'#AAB7C3',width:1},majorGridlines:null},
    yAxis:{...axis,textStyle:{fontSize:20,fill:color.muted},
      majorGridlines:{fill:'#E1E8ED',width:1},line:{fill:'none',width:0}},
    dataLabels:{showValue:true,position:'outEnd',numberFormatCode:'0.00',
      textStyle:{fontSize:20,fill:color.ink,bold:true}},
    chartFill:'none',chartLine:{fill:'none',width:0},plotAreaFill:'none',
    ...extra,
  });
  applyPresentationChartFont(c,{fontFamily:'Arial'});
  return c;
}

// Main presentation: seven slides, proposed six minutes, three minutes per member.
{
  const s=p.slides.add();s.background.fill='#FFFFFF';
  text(s,'MonTi component\nreproduction',64,115,1152,180,64,true);
  text(s,'Validation diagnostics and proposed next steps',64,330,1152,64,32);
  text(s,'Junyi Tang and Yifan Fan',64,440,1152,48,28);
  text(s,'Draft evidence presentation\n31 October 2026. Final improvement incomplete.',64,555,1152,80,25,false,color.muted);
  s.speakerNotes.text='Planned speaker: Junyi Tang. Target: 20 seconds. This is a preparation draft, not a record of rehearsal or contribution. State the component question and that the frozen result is inconclusive. The instructor has not confirmed Component approval.\nSources:\n'+BASE+'docs/REPORT_DRAFT.md\n'+BASE+'course/PROJECT_STATUS.json';
  metadata.push({number:1,title:'MonTi component reproduction',appendix:false,planned_speaker:'Junyi Tang',planned_seconds:20,
    sources:['docs/REPORT_DRAFT.md','course/PROJECT_STATUS.json']});
}
{
  const s=slide('Frozen comparison and metric','GossipCop-S split 0. Fixed GCN surrogate and GAGA victim.',
    ['course/frozen_protocol.json','course/configs/baseline.json','course/test_protocol.py','evidence/data/partition_audit.json'],
    'Explain MR as total incorrect target occurrences divided by all target occurrences. Groups can overlap. For group sizes 1 and 3 with errors 1 and 0, MR is25%, whereas the unweighted group mean is50%. Run seed is the replication unit. Both conditions retrain separately, max100 epochs, patience10. The random control resamples uniformly without replacement on every forward when the pool exceedsK. The corrected flag and retained incidence budget limit equivalence to paper Table5.');
  text(s,'Learned and uniform-random candidates\nPaired seeds 0, 1 and 2\nrho = 0.05   xi = 0.1   K = 128',64,230,650,170,28);
  table(s,[['Split','Groups','Occurrences'],['Train','937','11,211'],['Validation','592','4,963'],['Test','909','10,515']],760,220,456,230,[180,110,166],22);
  text(s,'MR counts target occurrences',64,440,650,45,32,true);
  text(s,'Known answer: 1 error across groups of size 1 and 3 gives 25% MR.',64,503,1130,62,27);
  foot(s,'Higher MR means a stronger attack. Overlapping occurrences are not independent seeds.');
}
{
  const s=slide('Baseline conclusion is inconclusive','Attacked GAGA MR, three paired training seeds',
    ['evidence/reproduction_summary.json','evidence/baseline_audit.json','course/frozen_protocol.json'],
    'All six final runs completed and were independently recounted. Each contains909 groups and10515 occurrences. Learned mean43.7280% with sampleSD0.7459; random40.1902% with sampleSD1.3019. Three observed positive differences do not override the paired t(df2) interval crossing zero. The interval is conditional on fixed split and detectors. Clean GAGA MR21.6833%. No extra seeds were added.');
  chart(s,baseline.rows.map(v=>'Seed '+v.seed),[
    {name:'Learned',values:baseline.rows.map(v=>chartNumber(v.learned_mr_percent)),fill:color.blue},
    {name:'Random',values:baseline.rows.map(v=>chartNumber(v.random_mr_percent)),fill:color.gray},
  ],{left:64,top:210,width:740,height:360},{min:0,max:50,majorUnit:10,numberFormatCode:'0',title:{text:'MR (%)',textStyle:{fontSize:21}}});
  text(s,'+3.5378 pp',850,220,365,75,46,true,color.blue);
  text(s,'Paired mean difference',850,300,365,40,25);
  text(s,'95% CI (pp)\n[-1.5415, 8.6171]',850,365,365,105,29,true);
  text(s,'Paired t, df = 2',850,500,365,48,25,false,color.muted);
  foot(s,'Frozen rule: an interval crossing zero gives inconclusive. Uncertainty covers three run seeds.');
}
{
  const s=slide('Earlier pilots had negative mean gains','GCN validation only. Seeds 3/4/5. 256 training groups and 128 validation groups.',
    ['evidence/improvement_pilot_summary.json','evidence/improvement_pilot_audit.json','course/improvements/pilot_protocol_v1.json'],
    'All nine technical pilots and selected-checkpoint audit completed. Each validation run has1091 occurrences. Coverage and size sampling each lost to their paired baseline in all three seeds. Original selection mechanically nominated coverage even when both means were negative. Owner subsequently held the follow-up and required outcome-first diagnosis. These are not GAGA test effects or course-approved pilots.');
  chart(s,['Coverage','Size sampling'],[{name:'Mean gain',values:[pilots.effects.coverage.mean_gain_pp,pilots.effects.size_sampling.mean_gain_pp].map(chartNumber),fill:color.blue,
    points:[{idx:0,fill:color.blue},{idx:1,fill:color.orange}]}],
    {left:64,top:220,width:680,height:320},{min:-0.8,max:0,majorUnit:0.2,numberFormatCode:'0.0',title:{text:'Difference (pp)',textStyle:{fontSize:21}}},
    {dataLabels:{showValue:true,position:'outEnd',numberFormatCode:'0.000',textStyle:{fontSize:23,bold:true,fill:color.ink}}});
  text(s,'Coverage CI\n[-0.4110, 0.0444] pp',800,240,410,100,28);
  text(s,'Size sampling CI\n[-1.3709, 0.2710] pp',800,380,410,100,28);
  foot(s,'Exploratory paired 95% intervals. Historical coverage follow-up remains ON HOLD.');
}
{
  const s=slide('Scoring roles and crossed device replay','96/96 captures use the first target as score source. The paper uses the injected representation.',
    ['docs/SCORE_ROLE_REPLAY.md','evidence/score_role_replay_v2/summary.json','evidence/score_role_replay_audit_v2.json','course/PAPER_EDGE_SCORE_CHECK.md','https://ojs.aaai.org/index.php/AAAI/article/view/33760/35915'],
    'The physical injected edge is correct. The discrepancy concerns which contextualized representation scores it, not absence of injected information. Across48 group/checkpoint cases and243 exposures, CPU generation gives148 residuals and17 historical vectors on either detector, GPU generation gives147 and48. Fixed-graph detector-device changes cause no class differences. Prediction concordance does not establish original feature/graph identity. Ties alone are not a demonstrated sole cause. These are exploratory GCN validation diagnostics, not a successful correction.');
  table(s,[['Generator','GCN detector','Residuals /243','Historical vectors'],['CPU','CPU','148','17 /48'],['CPU','GPU','148','17 /48'],['GPU','CPU','147','48 /48'],['GPU','GPU','147','48 /48']],
    64,245,1152,250,[220,245,290,397],25);
  text(s,'Detector device changed zero classes for each captured attack.',64,525,1152,45,27,true);
  foot(s,'GCN validation only. Physical edge source is correct. Prediction match does not prove graph identity.');
}
{
  const s=slide('Follow-up options await a decision','PROPOSED. No direction approved, no frozen pilot and no real-data A/B results.',
    ['course/IMPROVEMENT_PROPOSAL_V2.md','course/outcome_v2/PILOT_PROTOCOL_DRAFT.md','evidence/outcome_v2_synthetic_checks/attempt1.json','evidence/outcome_v2_synthetic_checks/review_and_receipt.json'],
    'A aligns scoring with the injected source and requires consistently retrained weights with a fresh matched baseline. B applies only when node and edge budgets both equal1. It uses clean fixed-surrogate predicted-fraud eligibility and all-target fallback, never victim or evaluation-label oracle choices. GPU reconstruction shows six eligible opportunities, which are not measured gains. Compare native/all-target/random count-matched masks with shared features and fair clean-query accounting. C is only an unapproved exact-tie reproducibility control. Proposed ceiling2 aggregateGPUh is not launch authorization. Twelve synthetic checks establish preparation, not effectiveness or novelty.');
  text(s,'A  Source alignment',64,240,550,48,34,true);
  text(s,'Use the injected representation.\nRetrain under equal resources.\nCompare against fresh native training.',64,315,545,200,27);
  text(s,'B  Sole-edge priority',690,240,526,48,34,true);
  text(s,'Use clean-surrogate predictions.\nGPU replay identifies 6 opportunities.\nAll-target and same-count random masks.',690,315,526,210,27);
  foot(s,'12 synthetic CPU checks passed. Efficacy and novelty remain unestablished. Final testing has a separate checkpoint.');
}
{
  const s=slide('Completed evidence and remaining work','Report and presentation are drafts. Planned speaking roles are not actual contributions.',
    ['course/PROJECT_STATUS.json','course/GRADING_AUDIT.md','evidence/course-guide.txt','docs/HUMAN_REVIEW.md','CONTRIBUTIONS.md','AI_USAGE.md','docs/DEMO_SCRIPT.md'],
    'Completed evidence includes six frozen baseline runs, nine negative technical pilots, audited diagnostics and isolated functional preparation. Full improvement/control evaluation is unexecuted. Owner direction, instructor approvals, human technical checks, actual contributions and final course freeze remain pending. English report, deck and offline demonstration are preparation only. Both members must speak and perform their own defense. Six-minute timing here is a proposed plan, not a UMMoodle requirement or recorded rehearsal.');
  table(s,[['Evidence or requirement','Current status'],['Frozen reproduction','Complete, inconclusive'],['Earlier technical pilots','Complete, negative means'],['Full improvement and controls','Not executed'],['Direction and course approvals','Pending'],['Human review and contributions','Pending'],['Rehearsal and final freeze','Pending']],64,220,1152,342,[610,542],25);
  foot(s,'Original protocols and failures remain traceable. Instructor rules and genuine team verification still matter.');
}

// Optional appendix. These are evidence navigation slides, not defense answers.
{
  const s=slide('Appendix  Complete seed results','MR is in percent. Paired differences use percentage points.',
    ['evidence/reproduction_summary.json','evidence/improvement_pilot_summary.json','course/frozen_protocol.json'],
    'SampleSD is reported in the report. PaperTable5 is43.70/39.13 with difference4.57pp. Table4 is41.34+/-1.90, unresolved. The corrected selector flag, explicit random resampling and incidence-based budgets limit historical equivalence. Do not choose the closest paper table as proof.',true);
  table(s,[['Seed','Learned MR','Random MR','Difference pp'],...baseline.rows.map(v=>[String(v.seed),v.learned_mr_percent.toFixed(4),v.random_mr_percent.toFixed(4),v.difference_pp.toFixed(4)])],64,215,1152,168,[210,314,314,314],24);
  table(s,[['Seed','Pilot baseline','Coverage','Size sampling'],...pilots.rows.map(v=>[String(v.seed),v.baseline.toFixed(4),v.coverage.toFixed(4),v.size_sampling.toFixed(4)])],64,418,1152,168,[210,314,314,314],24);
  foot(s,'Top: frozen GAGA test. Bottom: earlier GCN validation pilots. Their cohorts and detectors differ.');
}
{
  const s=slide('Appendix  Residual concentration','Residual means a target occurrence still classified correctly after attack.',
    ['evidence/results_first_diagnosis_v1/training_derived_bins.csv','evidence/results_first_diagnosis_v1/summary.json','docs/RESULTS_FIRST_DIAGNOSIS.md'],
    'Lowest training-derived nodes-per-target bin contains432/438 residuals but2727/14889 repeated seed-occurrence exposures. Residual rate432/2727=15.84% versus6/12162=.0493% elsewhere. Budget,degree,geometry are confounded. Selected validation checkpoints and repeated overlapping nodes are not independent confirmation. No monotone group-size trend supports a simple size explanation.',true);
  chart(s,['Lowest allocation bin'],[
    {name:'Share of exposures',values:[chartNumber(2727/14889*100)],fill:color.gray},
    {name:'Share of residuals',values:[chartNumber(432/438*100)],fill:color.blue}],
    {left:64,top:220,width:700,height:350},{min:0,max:100,majorUnit:20,numberFormatCode:'0',title:{text:'Share (%)',textStyle:{fontSize:21}}});
  text(s,'2,727 / 14,889 exposures\n432 / 438 residuals',820,240,395,110,29,true);
  text(s,'Residual rate\n15.84% in this bin\n0.0493% elsewhere',820,400,395,150,28);
  foot(s,'Three seed passes on selected GCN validation checkpoints. Budget and degree are confounded.');
}
{
  const s=slide('Appendix  Fixed-feature endpoint oracle','16 validation groups and 3 checkpoints give 48 cases. Features remain fixed within each case.',
    ['evidence/one_edge_diagnosis_v2/summary.json','evidence/one_edge_endpoint_attribution_v1.json','docs/ONE_EDGE_DIAGNOSIS.md'],
    '243 occurrences and243 endpoint configurations happen to share a count but are different axes. CPU reconstruction has148 residuals, versus147 historical. Only17/48 full historical vectors agree. The oracle reduces CPU residuals to141: seven extra incorrect occurrences on this selected cohort,2.8807pp here only. All seven strict CPU cases selected an already clean-wrong endpoint. GPU reconstruction has six eligibility opportunities instead. These are neither trained-policy gains nor a universal one-edge capacity bound. The failed79-forward attempt and complete355-forward retry are preserved.',true);
  table(s,[['Quantity','Selected CPU cohort'],['Clean detector-correct','186 /243'],['Reconstructed residuals','148 /243'],['Restricted-oracle residuals','141 /243'],['Net extra incorrect occurrences','7']],64,220,1152,260,[735,417],26);
  text(s,'CPU oracle cases: 7. GPU eligibility opportunities: 6.',64,520,1152,50,28,true);
  foot(s,'Optimistic selected-subset headroom. No trained method, whole-validation gain or victim-transfer result.');
}
{
  const s=slide('Appendix  Ties and generation differences','Observed reference behavior does not establish the cause of every real generator difference.',
    ['evidence/score_role_replay_v2/summary.json','evidence/score_role_replay_v2/generators.csv','evidence/score_role_replay_v2/pairs.csv','evidence/score_role_replay_audit_v2.json'],
    'Actual raw max ties58/96; actual target-probability max ties63/96. Identical-vector native top-k differs across devices32/96, all on exact ties. Natural CPU/GPU vectors agree0/48, endpoints differ35/48 and candidate sets/order36/48. Old-weight paper-role raw reference ties36/96 and was not evaluated as an attack. Compare raw-to-raw, not soft-to-raw as a fix. C proposes lowest-global-ID exact ties only and cannot guarantee whole-generator device identity.',true);
  table(s,[['Observation','Count'],['Actual raw maximum ties','58 /96'],['Actual target-probability ties','63 /96'],['Same-vector top-k index differences','32 /96'],['Natural probability vectors identical','0 /48'],['Natural endpoints different','35 /48'],['Candidate sets and order different','36 /48']],64,215,1152,322,[860,292],24);
  foot(s,'Old-weight injected-role mathematical reference also ties in 36/96. No corrected attack was evaluated.');
}
{
  const s=slide('Appendix  Evidence and cost navigation','Recorded outputs, source mapping and hashes support the demonstration.',
    ['evidence/run_registry.json','evidence/publication_history_map.json','evidence/research_cost_ledger.json','docs/DEMO_SCRIPT.md','AI_USAGE.md'],
    'Trace learned seed0 run20261028T122617Z_reproduction_learned_s0_772a5d through registration/configuration/result/victim count summary. Public group-count and private raw-prediction hashes identify different artifacts. Original execution9f782df maps truthfully to an English public source. Six formal runner costs2.692921h and baseline queue2.759076h are different nested scopes and must not be summed. Additional research has12h cap, charged0.775006 and reserve0.020000. SharedGPU timing is descriptive and prices unavailable. Demo reads saved evidence. Actual rehearsal, human technical reviews and contribution verification remain unrecorded.',true);
  table(s,[['Accounting scope','GPU wall hours'],['Six formal baseline runner records','2.692921'],['Baseline queue including pilots/boundaries','2.759076'],['Additional research charged',ledger.measured_or_conservatively_charged.toFixed(6)],['Unmeasured setup reserve',ledger.unmetered_setup_reserve.toFixed(6)],['Additional remaining after reserve',ledger.remaining_after_reserve.toFixed(6)]],64,215,1152,285,[860,292],24);
  text(s,'Result  /  commit  /  config  /  seed and command  /  log  /  output hash',64,545,1152,48,25,true);
  foot(s,'Baseline rows are alternative accounting scopes. Additional research cap: 12 GPU-hours.');
}

await fs.mkdir(path.join(build,'private'),{recursive:true});
await fs.mkdir(path.join(build,'output'),{recursive:true});
const candidate=path.join(build,'private','candidate-'+revision+'.pptx');
const finalPath=path.join(build,'output','MonTi_Slides_Draft_'+revision+'.pptx');
await (await PresentationFile.exportPptx(p)).save(candidate);
const requirements={explicitTotalSlideCount:12,requiredNativeTableOwnerSlides:[2,5,7,8,10,11,12],
  requiredNativeChartOwnerSlides:[3,4,9],materializeLiteralChartWorkbooks:true};
await finalizePresentation({
  ...requirements,workspaceDir:build,candidatePath:candidate,finalPath,
  pythonExecutable:path.join(runtime,'python','python.exe'),
  integrityValidatorPath:path.join(skill,'container_tools','inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(skill,'container_tools','inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',
    ...requirements.requiredNativeTableOwnerSlides.flatMap(n=>['--require-native-table-slide',String(n)])],
  fontPolicy:{basis:'design',families:['Arial']},verifyArtifactToolImport:true,
  receiptPath:path.join(build,'private',revision+'.validation.json'),
});
// Preview generation is opt-in and separate from successful file validation.
// The bundled Windows chart renderer can fail at native shutdown after all
// awaited PNG writes complete. Never clear that exit code or weaken checks.
let previewDir=null;
if(args.previews==='artifact'){
  const final=await PresentationFile.importPptx(await FileBlob.load(finalPath));
  previewDir=path.join(build,'private','preview-'+revision);
  await fs.mkdir(previewDir,{recursive:true});
  for(const [i,s] of final.slides.items.entries()){
    const name='slide-'+String(i+1).padStart(2,'0');
    await fs.writeFile(path.join(previewDir,name+'.png'),new Uint8Array(await (await final.export({slide:s,format:'png',scale:1})).arrayBuffer()));
    await fs.writeFile(path.join(previewDir,name+'.layout.json'),await (await s.export({format:'layout'})).text());
  }
}
const sha=bytes=>crypto.createHash('sha256').update(bytes).digest('hex');
const sources=['docs/REPORT_DRAFT.md','docs/MonTi_Report_Draft.pdf','evidence/report_draft_build.json','evidence/report_draft_qa.json',
 'evidence/reproduction_summary.json','course/frozen_protocol.json','evidence/improvement_pilot_summary.json',
 'evidence/results_first_diagnosis_v1/training_derived_bins.csv','evidence/one_edge_diagnosis_v2/summary.json',
 'evidence/one_edge_endpoint_attribution_v1.json','evidence/score_role_replay_v2/summary.json',
 'evidence/score_role_replay_audit_v2.json','evidence/research_cost_ledger.json',
 'evidence/outcome_v2_synthetic_checks/attempt1.json','course/IMPROVEMENT_PROPOSAL_V2.md'];
const inputHashes={};
for(const relative of sources){
  const bytes=await fs.readFile(path.join(repo,relative));
  const snapshotBytes=execFileSync('git',['show',SNAPSHOT+':'+relative],{cwd:repo});
  if(sha(bytes)!==sha(snapshotBytes))throw new Error('Snapshot mismatch: '+relative);
  inputHashes[relative]=sha(bytes);
}
inputHashes['course/build_slides_draft.mjs']=sha(await fs.readFile(new URL(import.meta.url)));
const manifest={schema:'monti_slides_draft_build_v1',built_utc:new Date().toISOString(),
  evidence_snapshot:SNAPSHOT,revision,slides:12,main_slides:7,appendix_slides:5,
  proposed_main_duration_seconds:360,planned_speaker_seconds:{'Junyi Tang':180,'Yifan Fan':180},
  inputs:inputHashes,output_sha256:sha(await fs.readFile(finalPath)),slide_metadata:metadata,
  native_table_owner_slides:requirements.requiredNativeTableOwnerSlides,
  native_chart_owner_slides:requirements.requiredNativeChartOwnerSlides,
  native_chart_workbooks:'New literal-data snapshots, intentionally materialized and checked by finalizer',
  presentation_chart_decimal_places:8,scientific_source_numbers:'Unchanged; full precision remains in the cited JSON/CSV files',
  font_policy:{basis:'design',families:['Arial']},
  source_tree_head:execFileSync('git',['rev-parse','HEAD'],{cwd:repo,encoding:'utf8'}).trim(),
  scope:'Presentation preparation only. No ML imports, model queries, training or GPU work.',
  human_rehearsal_and_technical_review:'not recorded',owner_direction:'pending',
  finalizer_receipt_sha256:sha(await fs.readFile(path.join(build,'private',revision+'.validation.json'))),
};
await fs.writeFile(path.join(build,'private',revision+'.build.json'),JSON.stringify(manifest,null,2)+'\n');
console.log(JSON.stringify({status:'finalized_pending_visual_review',slides:12,finalPath,previewDir,sha256:manifest.output_sha256,process_exit_code_at_completion:process.exitCode??0}));
