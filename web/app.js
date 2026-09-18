const $ = (s) => document.querySelector(s);
const fileInput = $('#file'), drop = $('#drop'), preview = $('#preview'), canvas = $('#rimCanvas');
const ctx = canvas.getContext('2d');
const token = new URLSearchParams(location.search).get('token') || '';
const transport = new URLSearchParams(location.search).get('transport') || '';
const jobKey = `ballform-job:${token || 'local'}`;
let file, dragStart, rimBox;

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
  file = chosen; preview.src = URL.createObjectURL(file); preview.load();
  drop.classList.add('hidden'); $('#recordLabel').classList.add('hidden'); $('#markStep').classList.remove('hidden');
  preview.onloadedmetadata = () => { preview.currentTime = Math.min(preview.duration * .25, preview.duration - .05); resizeCanvas(); };
}
fileInput.onchange = () => loadFile(fileInput.files[0]);
$('#cameraFile').onchange = () => loadFile($('#cameraFile').files[0]);
['dragenter','dragover'].forEach(e => drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.add('over'); }));
['dragleave','drop'].forEach(e => drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.remove('over'); }));
drop.addEventListener('drop', e => loadFile(e.dataTransfer.files[0]));
window.addEventListener('resize', resizeCanvas);
function resizeCanvas(){ const r=preview.getBoundingClientRect(); canvas.width=r.width*devicePixelRatio; canvas.height=r.height*devicePixelRatio; canvas.style.height=r.height+'px'; drawBox(); }
$('#scrubber').oninput = e => preview.currentTime = preview.duration * e.target.value / 100;
function point(e){ const r=canvas.getBoundingClientRect(); return {x:(e.clientX-r.left)/r.width,y:(e.clientY-r.top)/r.height}; }
canvas.onpointerdown = e => { dragStart=point(e); rimBox=null; canvas.setPointerCapture(e.pointerId); };
canvas.onpointermove = e => { if(!dragStart)return; const p=point(e); rimBox=[Math.min(p.x,dragStart.x),Math.min(p.y,dragStart.y),Math.abs(p.x-dragStart.x),Math.abs(p.y-dragStart.y)]; drawBox(); };
canvas.onpointerup = () => dragStart=null;
function drawBox(){ ctx.clearRect(0,0,canvas.width,canvas.height); if(!rimBox)return; const [x,y,w,h]=rimBox,d=devicePixelRatio;ctx.strokeStyle='#ff6a32';ctx.lineWidth=3*d;ctx.setLineDash([8*d,5*d]);ctx.strokeRect(x*canvas.width,y*canvas.height,w*canvas.width,h*canvas.height);ctx.fillStyle='#ff6a32';ctx.font=`${12*d}px DM Mono`;ctx.fillText('RIM',x*canvas.width,(y*canvas.height)-7*d); }
$('#clear').onclick = () => {rimBox=null;drawBox();};

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
  const form=new FormData(); form.append('video',file); if(rimBox)form.append('rim',JSON.stringify(rimBox));
  try {
    const {job_id}=await upload(form);
    localStorage.setItem(jobKey, job_id);
    $('#progressText').textContent='Upload complete. Waiting for the Mac…';
    poll(job_id);
  } catch(e){ showError(e.message); }
};
async function poll(id){
  try{
    const response=await fetch(apiUrl(`/api/jobs/${id}`)), state=await response.json();
    if(!response.ok) throw new Error(state.detail || 'Could not read the analysis job');
    $('#meter').style.width=`${35+Math.max(0,state.progress)*65}%`;$('#progressText').textContent=state.message;
    if(state.status==='complete') return render(id,state.result);
    if(state.status==='failed'){localStorage.removeItem(jobKey);return showError(state.message);}
    setTimeout(()=>poll(id),1000);
  }catch(e){showError(e.message);}
}
function showError(message){ $('#progressText').classList.add('error');$('#progressText').textContent=message; }
const labels={elbow_angle_at_release_deg:'Elbow at release',set_point_elbow_angle_deg:'Set-point elbow',upper_arm_elevation_deg:'Upper-arm elevation',wrist_over_elbow_pct_shoulder_width:'Wrist / elbow offset',release_height_body_ratio:'Release height ratio',follow_through_extension_deg:'Follow-through',release_angle_2d_deg:'2D launch angle'};
function displayMetric(k,v){if(v==null)return '—';if(k.includes('angle')||k.includes('_deg'))return `${v}°`;if(k.includes('pct'))return `${v}%`;return v;}
function esc(s){const d=document.createElement('div');d.textContent=s;return d.innerHTML;}
function render(id,result){
  localStorage.removeItem(jobKey);
  $('#progressStep').classList.add('hidden');$('#workspace').classList.add('hidden');$('#results').classList.remove('hidden');
  const made=result.shots.filter(s=>s.outcome.includes('made')).length;
  $('#summary').innerHTML=`<div><small>Shots found</small><strong>${result.shots.length}</strong></div><div><small>Made</small><strong>${made}</strong></div><div><small>Camera view</small><strong>${esc(result.camera_view)}</strong></div><div><small>Pose coverage</small><strong>${result.diagnostics.pose_frames} frames</strong></div>`;
  $('#annotated').src=apiUrl(`/api/jobs/${id}/video`);
  $('#shots').innerHTML=result.shots.length ? result.shots.map(s=>`<article class="shot"><div class="shot-top"><h3>Shot ${s.number} <small>@ ${s.release_s}s</small></h3><span class="pill ${s.outcome.includes('miss')?'missed':s.outcome==='unknown'?'unknown':''}">${esc(s.outcome)} · ${Math.round(s.outcome_confidence*100)}%</span></div><div class="metrics">${Object.entries(s.metrics).map(([k,v])=>`<div class="metric"><small>${labels[k]||k}</small><strong>${displayMetric(k,v)}</strong></div>`).join('')}</div><p class="cue">${s.cues.map(esc).join(' ')}</p><div class="evidence">Evidence: ${s.evidence.map(esc).join(' · ')}</div></article>`).join('') : '<article class="shot"><p class="cue">No complete shot arc was detected. Try a clip where the ball, shooting wrist, and basket stay visible from gather through landing.</p></article>';
  $('#limitations').innerHTML=result.limitations.map(x=>`<li>${esc(x)}</li>`).join('');window.scrollTo({top:$('#results').offsetTop-30,behavior:'smooth'});
}
$('#again').onclick=()=>location.reload();

const existingJob = localStorage.getItem(jobKey);
if(existingJob){
  drop.classList.add('hidden');
  $('#recordLabel').classList.add('hidden');
  $('#progressStep').classList.remove('hidden');
  $('#progressText').textContent='Reconnecting to analysis on the Mac…';
  poll(existingJob);
}
