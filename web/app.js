const $ = (s) => document.querySelector(s);
const fileInput = $('#file'), drop = $('#drop'), preview = $('#preview'), canvas = $('#rimCanvas');
const ctx = canvas.getContext('2d');
const token = new URLSearchParams(location.search).get('token') || '';
const transport = new URLSearchParams(location.search).get('transport') || '';
const jobKey = `ballform-job:${token || 'local'}`;
let file, dragStart, rimBox, report;
let courtPoints = [], marking = 'rim';
let selectedPoseModelForJob = null;
const poseModelLabels = {'yolo26m-pose':'YOLO26m pose','yolo26s-pose':'YOLO26s pose'};
function updateSettings() {
  const game = $('#analysisMode').value === 'one_on_one';
  $('#handedness').disabled = game;
  $('#poseModel').disabled = !game;
  $('#modeHelp').textContent = game ? 'For NBA broadcasts choose the elevated broadcast camera. Mark the playing area to exclude spectators and benches. Game analysis checks both shooting hands. Wide views take longer to process.' : 'Keep one shooter’s arm, ball, feet, and basket visible. Use a steady camera for a repeatable mechanics review.';
  $('#analyze').innerHTML = `${game ? 'Analyze game' : 'Analyze form'} <span>→</span>`;
  const moving = $('#cameraProfile').value === 'moving';
  $('#markHelp').textContent = game ? (moving ? 'Tracked rim: scrub to any frame where the hoop is clear, then drag a snug box around it. The tracker works forward and backward from that timestamp; cuts stop outcome tracking. You may also mark the playing area.' : 'Playing area: click corners around the visible court in order. Leave benches and spectators outside. For outcomes, choose Stationary courtside or Moving broadcast + tracked rim.') : 'Drag a snug box around the rim for stationary-camera outcome estimates.';
}
$('#analysisMode').onchange = updateSettings;
$('#cameraProfile').onchange = () => {
  if (['broadcast','elevated','moving'].includes($('#cameraProfile').value)) $('#analysisMode').value = 'one_on_one';
  updateSettings();
};
$('#markRim').onclick = () => {marking='rim';};
$('#markCourt').onclick = () => {marking='court';};
updateSettings();

function apiUrl(path) {
  const url = new URL(path, location.origin);
  if (token) url.searchParams.set('token', token);
  return url.toString();
}
if (token) {
  $('#modeLabel').textContent = transport === 'tailscale' ? '● PRIVATE TAILSCALE LINK' : '● PRIVATE WI-FI LINK';
  $('#homeLink').href = `/?token=${encodeURIComponent(token)}${transport ? `&transport=${encodeURIComponent(transport)}` : ''}`;
}

function loadFile(chosen) {
  if (!chosen) return;
  if (preview.src.startsWith('blob:')) URL.revokeObjectURL(preview.src);
  rimBox = null;
  courtPoints = [];
  file = chosen; preview.src = URL.createObjectURL(file); preview.load();
  drop.classList.add('hidden'); $('#recordLabel').classList.add('hidden'); $('#markStep').classList.remove('hidden');
  preview.onloadedmetadata = () => { preview.currentTime = $('#cameraProfile').value === 'moving' ? 0 : Math.min(preview.duration * .25, preview.duration - .05); resizeCanvas(); };
}
fileInput.onchange = () => loadFile(fileInput.files[0]);
$('#cameraFile').onchange = () => loadFile($('#cameraFile').files[0]);
['dragenter','dragover'].forEach(e => drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.add('over'); }));
['dragleave','drop'].forEach(e => drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.remove('over'); }));
drop.addEventListener('drop', e => loadFile(e.dataTransfer.files[0]));
window.addEventListener('resize', resizeCanvas);
function resizeCanvas(){ const r=preview.getBoundingClientRect(); canvas.width=r.width*devicePixelRatio; canvas.height=r.height*devicePixelRatio; canvas.style.height=r.height+'px'; drawBox(); }
$('#scrubber').oninput = e => preview.currentTime = preview.duration * e.target.value / 100;
function point(e){ const r=canvas.getBoundingClientRect(); return {x:Math.max(0,Math.min(1,(e.clientX-r.left)/r.width)),y:Math.max(0,Math.min(1,(e.clientY-r.top)/r.height))}; }
canvas.onpointerdown = e => {
  if(marking==='court') {if(courtPoints.length<8){const p=point(e);courtPoints.push([p.x,p.y]);drawBox();}return;}
  dragStart=point(e); rimBox=null; canvas.setPointerCapture(e.pointerId);
};
canvas.onpointermove = e => { if(!dragStart)return; const p=point(e); rimBox=[Math.min(p.x,dragStart.x),Math.min(p.y,dragStart.y),Math.abs(p.x-dragStart.x),Math.abs(p.y-dragStart.y)]; drawBox(); };
canvas.onpointerup = () => dragStart=null;
function drawBox(){
  ctx.clearRect(0,0,canvas.width,canvas.height);
  const d=devicePixelRatio;ctx.lineWidth=3*d;ctx.setLineDash([]);
  if(courtPoints.length){
    ctx.strokeStyle='#50d4ed';ctx.fillStyle='#50d4ed';ctx.beginPath();
    courtPoints.forEach(([x,y],i)=>i?ctx.lineTo(x*canvas.width,y*canvas.height):ctx.moveTo(x*canvas.width,y*canvas.height));
    if(courtPoints.length>=3)ctx.closePath();ctx.stroke();
    courtPoints.forEach(([x,y])=>{ctx.beginPath();ctx.arc(x*canvas.width,y*canvas.height,4*d,0,Math.PI*2);ctx.fill();});
  }
  if(!rimBox)return;
  const [x,y,w,h]=rimBox;ctx.strokeStyle='#ff6a32';ctx.setLineDash([8*d,5*d]);ctx.strokeRect(x*canvas.width,y*canvas.height,w*canvas.width,h*canvas.height);ctx.fillStyle='#ff6a32';ctx.font=`${12*d}px DM Mono`;ctx.fillText('RIM',x*canvas.width,(y*canvas.height)-7*d);
}
$('#clear').onclick = () => {if(marking==='court')courtPoints=[];else rimBox=null;drawBox();};

function upload(form) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', apiUrl('/api/jobs'));
    xhr.upload.onprogress = event => {
      if (!event.lengthComputable) return;
      const percent = Math.round(event.loaded / event.total * 100);
      $('#meter').style.width = `${Math.max(2, percent * .35)}%`;
      $('#progressText').textContent = `Uploading to Mac… ${percent}%`;
    };
    xhr.onload = () => {
      let body = {};
      try { body = JSON.parse(xhr.responseText); } catch (_) {}
      if (xhr.status >= 200 && xhr.status < 300) resolve(body);
      else reject(new Error(body.detail || `Upload failed (${xhr.status})`));
    };
    xhr.onerror = () => reject(new Error('The phone lost its connection to the Mac. Check Wi-Fi and try again.'));
    xhr.send(form);
  });
}

$('#analyze').onclick = async () => {
  if(!file)return;
  if(rimBox && (rimBox[2] < .01 || rimBox[3] < .01)) rimBox=null;
  $('#markStep').classList.add('hidden'); $('#progressStep').classList.remove('hidden');
  $('#settings').classList.add('hidden'); $('#retry').classList.add('hidden'); $('#progressText').classList.remove('error');
  const form=new FormData(); form.append('video',file); if(rimBox)form.append('rim',JSON.stringify(rimBox));
  if (rimBox) form.append('rim_time_s', String(Math.max(0, preview.currentTime)));
  form.append('mode', $('#analysisMode').value); form.append('handedness', $('#handedness').value);
  form.append('camera', $('#cameraProfile').value);
  form.append('pose_model', $('#poseModel').value);
  if(courtPoints.length)form.append('court',JSON.stringify(courtPoints));
  const selectedModel = $('#poseModel').value;
  selectedPoseModelForJob = $('#analysisMode').value==='one_on_one' ? selectedModel : null;
  $('#modelStatus').textContent = selectedPoseModelForJob
    ? `Selected for this request: ${poseModelLabels[selectedModel]||selectedModel}. Uploading…`
    : 'Game pose model selection is not used in single-person form mode.';
  try {
    const {job_id,pose_model_requested:acceptedModel}=await upload(form);
    if(selectedPoseModelForJob){
      $('#modelStatus').textContent=acceptedModel===selectedPoseModelForJob
        ? `Server accepted: ${poseModelLabels[acceptedModel]||acceptedModel}. Waiting for analyzer load confirmation…`
        : `Server did not confirm ${poseModelLabels[selectedPoseModelForJob]||selectedPoseModelForJob}. Refresh the page and restart Ballform if this persists.`;
    }
    localStorage.setItem(jobKey, job_id);
    $('#progressText').textContent='Upload complete. Waiting for the Mac…';
    poll(job_id);
  } catch(e){ showError(e.message); }
};
async function poll(id){
  try{
    const response=await fetch(apiUrl(`/api/jobs/${id}`)), state=await response.json();
    if(!response.ok) throw new Error(state.detail || 'Could not read the analysis job');
    if(state.pose_model_requested){
      const requested=poseModelLabels[state.pose_model_requested]||state.pose_model_requested;
      const loaded=state.pose_model_loaded;
      $('#modelStatus').textContent=loaded
        ? `Server accepted: ${requested}. Analyzer loaded: ${loaded}.`
        : state.status==='failed'
          ? `Server accepted: ${requested}, but no model load was confirmed before failure: ${state.message}`
          : `Server accepted: ${requested}. Waiting for loaded-model confirmation…`;
    }else if(selectedPoseModelForJob && state.status!=='complete' && state.status!=='failed'){
      $('#modelStatus').textContent='This server has not acknowledged the selected model. Restart Ballform, refresh the page, and resubmit.';
    }
    $('#meter').style.width=`${35+Math.max(0,state.progress)*65}%`;$('#progressText').textContent=state.message;
    if(state.status==='complete') return render(id,state.result);
    if(state.status==='failed'){localStorage.removeItem(jobKey);return showError(state.message);}
    setTimeout(()=>poll(id),1000);
  }catch(e){showError(e.message);}
}
function showError(message){ $('#progressText').classList.add('error');$('#progressText').textContent=message;$('#retry').classList.remove('hidden'); }
$('#retry').onclick = () => { localStorage.removeItem(jobKey); if (!file) return location.reload(); $('#progressStep').classList.add('hidden'); $('#markStep').classList.remove('hidden'); $('#settings').classList.remove('hidden'); };
const labels={elbow_angle_at_release_deg:'Elbow at release',set_point_elbow_angle_deg:'Set-point elbow',upper_arm_elevation_deg:'Upper-arm elevation',wrist_over_elbow_pct_shoulder_width:'Wrist / elbow offset',release_height_body_ratio:'Release height ratio',follow_through_extension_deg:'Follow-through',release_angle_2d_deg:'2D launch angle'};
Object.assign(labels,{visible_players:'Players visible at release',separation_torso:'Projected separation at release',contest_clearance_torso:'Defender hand / release clearance',separation_change_torso:'Separation change before release',defender_selection_margin_torso:'Defender selection margin',separation:'Release separation',contest_clearance:'Contest clearance'});
labels.visible_hand_clearance_torso='Visible hand clearance (other hand unknown)';
function displayMetric(k,v){if(v==null)return 'Unavailable';const value=typeof v==='number'?Number(v.toFixed(2)):v;if(k.includes('angle')||k.includes('_deg'))return `${value}°`;if(k.includes('pct'))return `${value}%`;if(k.includes('torso'))return `${value} torso lengths`;if(k.endsWith('_s'))return `${value} s`;return value;}
function esc(s){const d=document.createElement('div');d.textContent=s;return d.innerHTML;}
function label(k){return labels[k]||k.replaceAll('_',' ');}
function metricsMarkup(metrics){return Object.entries(metrics||{}).map(([k,v])=>`<div class="metric"><small>${esc(label(k))}</small><strong>${esc(displayMetric(k,v))}</strong></div>`).join('');}
function gameMarkup(game){
  if(!game)return '<p class="cue">Game measurements unavailable for this release.</p>';
  const components=Object.entries(game.components||{}).map(([key,value])=>{
    const component=typeof value==='object'&&value!==null?value:{score:value};
    return `<div class="component"><span>${esc(label(key))}</span><strong>${component.score==null?'Unavailable':`${esc(component.score)} / 100`}</strong>${component.weight!=null?`<small>Weight ${Math.round(component.weight*100)}%</small>`:''}</div>`;
  }).join('');
  const range=game.score_range;
  const scoreText=game.score!=null?`${esc(game.score)}<span> / 100</span>`:range?`${esc(range.lower)}–${esc(range.upper)}<span> possible range</span>`:'Not scored';
  return `<section class="game-review" aria-label="Shot-space analysis"><div class="score-row"><div><small>Shot-space score</small><strong>${scoreText}</strong></div><p>${game.confidence==null?'Evidence quality unavailable':`${Math.round(game.confidence*100)} / 100 evidence quality`}<br><small>Heuristic · higher means more measured space</small></p></div>${range?`<p class="cue">${esc(range.reason)}</p>`:''}<div class="metrics">${metricsMarkup(game.metrics)}</div>${components?`<details><summary>Score components</summary><div class="components">${components}</div></details>`:''}<p class="evidence">${(game.evidence||[]).map(esc).join(' · ')}</p>${(game.limitations||[]).length?`<ul class="game-limitations">${game.limitations.map(x=>`<li>${esc(x)}</li>`).join('')}</ul>`:''}</section>`;
}
function render(id,result){
  report=result;
  localStorage.removeItem(jobKey);
  $('#progressStep').classList.add('hidden');$('#workspace').classList.add('hidden');$('#results').classList.remove('hidden');
  const game=result.mode==='one_on_one';
  $('#reportMode').textContent=game?`GAME REVIEW · ${(result.camera_profile||'AUTO').toUpperCase()}`:`SHOOTING FORM · ${(result.handedness||$('#handedness').value).toUpperCase()} HAND`;
  $('#gameHelp').classList.toggle('hidden',!game);
  if(game){const method=result.game_summary?.method;$('#gameHelp').textContent='Shot-space score is a transparent 0–100 heuristic, not make probability or a validated player grade. Distances are projected in the image and normalized to the shooter’s torso length; they are not feet or meters. Compare clips only with similar camera angles.'+(method?.formula?` Score: ${method.formula}.`:'');}
  const made=result.shots.filter(s=>s.outcome==='made'||s.outcome==='likely made').length;
  const movingView=game&&['broadcast','elevated'].includes(result.camera_profile);
  const scored=result.shots.filter(s=>s.game?.score!=null);
  const diagnostics=result.diagnostics||{};
  const performance=result.performance||{};
  const performanceText=performance.elapsed_seconds!=null?`Analysis: ${performance.elapsed_seconds}s · Peak app memory: ${performance.peak_process_memory_mb} MB.`:'';
  const requested=result.vision?.pose_model_requested??result.vision?.pose_model_choice;
  const actual=result.vision?.pose_model;
  const modelText=requested?`Pose model requested: ${poseModelLabels[requested]||requested}; loaded: ${actual||'not reported'}. `:'';
  $('#visionSummary').textContent=game?`${modelText}Tracking: up to ${diagnostics.max_players_visible??0} player candidates per frame; ${diagnostics.ball_detections??0} ball observations; ${diagnostics.scene_cuts??0} camera cuts. ${performanceText} Review player IDs in the video to confirm the matchup.`:performanceText;
  $('#summary').innerHTML=`<div><small>Releases found</small><strong>${result.shots.length}</strong></div><div><small>${movingView?'Make / miss':'Made / likely made'}</small><strong>${movingView?'Unavailable':made}</strong></div><div><small>Camera view</small><strong>${esc(result.camera_view||'Unknown')}</strong></div><div><small>${game?'Scored releases':'Pose coverage'}</small><strong>${game?`${scored.length} / ${result.shots.length}`:`${result.diagnostics?.pose_frames??0} frames`}</strong></div>`;
  $('#annotated').src=apiUrl(`/api/jobs/${id}/video`);
  $('#shots').innerHTML=result.shots.length ? result.shots.map(s=>`<article class="shot"><div class="shot-top"><h3>Shot ${esc(s.number)}</h3><button class="ghost seek" data-time="${Number(s.release_s)}" aria-label="Review shot ${Number(s.number)} release">Review ${esc(s.release_s)}s</button><span class="pill ${s.outcome.includes('miss')?'missed':s.outcome==='unknown'?'unknown':''}">${esc(s.outcome)} · ${Math.round(s.outcome_confidence*100)}%</span></div>${game?gameMarkup(s.game):''}${Object.keys(s.metrics||{}).length?`<h4 class="section-label">Shooting mechanics</h4><div class="metrics">${metricsMarkup(s.metrics)}</div>`:''}${s.cues?.length?`<p class="cue">${s.cues.map(esc).join(' ')}</p>`:''}<div class="evidence">Evidence: ${(s.evidence||[]).map(esc).join(' · ')}</div></article>`).join('') : '<article class="shot"><p class="cue">No complete shot arc was detected. Try a clip where the ball, shooting wrist, and basket stay visible from gather through landing.</p></article>';
  $('#shots').querySelectorAll('.seek').forEach(button=>button.onclick=()=>{const video=$('#annotated');video.currentTime=Math.max(0,Number(button.dataset.time)-.5);video.pause();video.scrollIntoView({behavior:'smooth',block:'center'});});
  $('#limitations').innerHTML=result.limitations.map(x=>`<li>${esc(x)}</li>`).join('');window.scrollTo({top:$('#results').offsetTop-30,behavior:'smooth'});
}
$('#download').onclick=()=>{if(!report)return;const url=URL.createObjectURL(new Blob([JSON.stringify(report,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download=`ballform-${report.mode||'form'}-report.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
$('#again').onclick=()=>location.reload();

const existingJob = localStorage.getItem(jobKey);
if(existingJob){
  drop.classList.add('hidden');
  $('#recordLabel').classList.add('hidden');
  $('#progressStep').classList.remove('hidden');
  $('#progressText').textContent='Reconnecting to analysis on the Mac…';
  poll(existingJob);
}
