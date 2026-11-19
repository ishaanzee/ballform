// Player match panel for form_match reports: closest players, what to change, measurements
// side by side, the two skeletons played through the release (side by side or overlaid), and the
// chosen player's skeleton drawn over the review video at each of your free throws.
(() => {
  const BONES = [['left_shoulder', 'right_shoulder'], ['left_shoulder', 'left_elbow'], ['left_elbow', 'left_wrist'],
    ['right_shoulder', 'right_elbow'], ['right_elbow', 'right_wrist'], ['left_shoulder', 'left_hip'],
    ['right_shoulder', 'right_hip'], ['left_hip', 'right_hip'], ['left_hip', 'left_knee'], ['left_knee', 'left_ankle'],
    ['right_hip', 'right_knee'], ['right_knee', 'right_ankle']];
  const esc = s => { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; };
  const value = (v, unit) => v == null ? '—' : unit === '°' ? `${Math.round(v)}°` : unit === 's' ? `${v.toFixed(2)} s` : `${v.toFixed(2)} torso`;
  let data = null, chosen = 0, frame = 0, timer = null, overlaid = false, onVideo = true, raf = null;

  function draw() {
    const canvas = document.querySelector('#matchCanvas');
    if (!canvas || !data) return;
    const ctx = canvas.getContext('2d'), w = canvas.width, h = canvas.height, times = data.skeleton.times;
    ctx.clearRect(0, 0, w, h);
    const style = getComputedStyle(document.documentElement);
    const ink = style.getPropertyValue('--ink') || '#171a19', orange = style.getPropertyValue('--orange') || '#fa5a24';
    const scale = h / 5.4;   // torso lengths: standing reach is about 4.6
    const figures = [[data.skeleton.yours, w * (overlaid ? .5 : .27), orange, 'You', w * (overlaid ? .38 : .27)], [data.matches[chosen].skeleton, w * (overlaid ? .5 : .73), ink, data.matches[chosen].name, w * (overlaid ? .62 : .73)]];
    for (const [frames, cx, color, name, labelX] of figures) {
      ctx.fillStyle = color; ctx.font = '700 12px DM Mono, monospace'; ctx.textAlign = 'center';
      ctx.fillText(name.toUpperCase(), labelX, 18);
      ctx.strokeStyle = style.getPropertyValue('--line') || '#c9c5bb'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(cx - w * .2, h - 14); ctx.lineTo(cx + w * .2, h - 14); ctx.stroke();
      const joints = frames && frames[frame];
      if (!joints) { ctx.fillStyle = style.getPropertyValue('--muted') || '#696c68'; ctx.fillText('not in view', labelX, h / 2); continue; }
      const at = name => [cx + joints[name][0] * scale, h - 14 - joints[name][1] * scale];
      ctx.strokeStyle = color; ctx.lineWidth = 4; ctx.lineCap = 'round'; ctx.globalAlpha = overlaid ? .8 : 1;
      for (const [a, b] of BONES) { ctx.beginPath(); ctx.moveTo(...at(a)); ctx.lineTo(...at(b)); ctx.stroke(); }
      const [nx, ny] = at('nose');
      ctx.beginPath(); ctx.arc(nx, ny, scale * .22, 0, Math.PI * 2); ctx.stroke(); ctx.globalAlpha = 1;
    }
    const t = times[frame];
    document.querySelector('#matchTime').textContent = Math.abs(t) < .001 ? 'release' : `${t > 0 ? '+' : '−'}${Math.abs(t).toFixed(2)} s`;
    document.querySelector('#matchScrub').value = frame;
  }

  // The player's joints at ``dt`` seconds from release, interpolated between stored frames.
  function jointsAt(frames, dt) {
    const times = data.skeleton.times;
    for (let i = 0; i + 1 < times.length; i++) {
      if (dt < times[i] || dt > times[i + 1] || !frames[i] || !frames[i + 1]) continue;
      const k = (dt - times[i]) / (times[i + 1] - times[i]), out = {};
      for (const name in frames[i]) out[name] = [0, 1].map(j => frames[i][name][j] + k * (frames[i + 1][name][j] - frames[i][name][j]));
      return out;
    }
    return null;
  }

  // Draws the chosen player over your footage: feet on your feet, scaled to your torso, facing your way.
  function drawOnVideo() {
    raf = requestAnimationFrame(drawOnVideo);
    const video = document.querySelector('#annotated'), canvas = document.querySelector('#reviewOverlay');
    if (!video || !canvas) return;
    const box = video.getBoundingClientRect(), vw = video.videoWidth, vh = video.videoHeight;
    const s = vw && vh ? Math.min(box.width / vw, box.height / vh) : 0, w = vw * s, h = vh * s, ratio = devicePixelRatio || 1;
    Object.assign(canvas.style, { left: `${(box.width - w) / 2}px`, top: `${(box.height - h) / 2}px`, width: `${w}px`, height: `${h}px` });
    if (canvas.width !== Math.round(w * ratio) || canvas.height !== Math.round(h * ratio)) { canvas.width = Math.round(w * ratio); canvas.height = Math.round(h * ratio); }
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!data || !onVideo || !w) return;
    const frames = data.matches[chosen].skeleton, times = data.skeleton.times, now = video.currentTime;
    const rep = data.release_times_s.findIndex((t, i) => data.placements[i] && now - t >= times[0] && now - t <= times[times.length - 1]);
    const joints = rep < 0 ? null : jointsAt(frames, now - data.release_times_s[rep]);
    if (!joints) return;
    // Your feet in this frame, so the player stays on them when the camera moves.
    const place = data.placements[rep], feet = jointsAt(place.track.map(p => p && { at: p }), now - data.release_times_s[rep]);
    if (!feet) return;
    const W = canvas.width, H = canvas.height, unit = place.torso * H;
    // The player's pose is upright; turn it by the tilt of your camera so it stands as your footage does.
    const turn = (place.tilt_deg || 0) * Math.PI / 180, cos = Math.cos(turn), sin = Math.sin(turn);
    const at = name => {
      const dx = joints[name][0] * place.facing * unit, dy = -joints[name][1] * unit;
      return [feet.at[0] * W + dx * cos - dy * sin, feet.at[1] * H + dx * sin + dy * cos];
    };
    for (const [color, width] of [['rgba(0,0,0,.55)', 7], ['#ffffff', 3.5]]) {
      ctx.strokeStyle = color; ctx.lineWidth = width * ratio; ctx.lineCap = 'round';
      for (const [a, b] of BONES) { ctx.beginPath(); ctx.moveTo(...at(a)); ctx.lineTo(...at(b)); ctx.stroke(); }
      const [nx, ny] = at('nose');
      ctx.beginPath(); ctx.arc(nx, ny, unit * .22, 0, Math.PI * 2); ctx.stroke();
    }
    ctx.font = `700 ${13 * ratio}px DM Mono, monospace`; ctx.textAlign = 'center'; ctx.lineWidth = 4 * ratio;
    const [lx, ly] = at('nose'), label = data.matches[chosen].name.toUpperCase();
    ctx.strokeStyle = 'rgba(0,0,0,.7)'; ctx.strokeText(label, lx, ly - unit * .9); ctx.fillStyle = '#fff'; ctx.fillText(label, lx, ly - unit * .9);
  }

  function videoControls() {
    const bar = document.querySelector('#matchOnVideo');
    if (!bar) return;
    const usable = data ? data.release_times_s.map((t, i) => [t, i]).filter(([, i]) => (data.placements || [])[i]) : [];
    bar.classList.toggle('hidden', !usable.length);
    if (!usable.length) { bar.innerHTML = ''; return; }
    const name = esc(data.matches[chosen].name);
    bar.innerHTML = `<label><input id="matchOnVideoToggle" type="checkbox" ${onVideo ? 'checked' : ''}> Draw ${name}'s pose over my video</label>` +
      usable.map(([t], n) => `<button type="button" class="ghost" data-time="${t}">Free throw ${n + 1}</button>`).join('');
    bar.querySelector('#matchOnVideoToggle').onchange = e => { onVideo = e.target.checked; };
    bar.querySelectorAll('button').forEach(b => b.onclick = () => {
      const video = document.querySelector('#annotated');
      video.currentTime = Math.max(0, Number(b.dataset.time) + data.skeleton.times[0]); video.playbackRate = .5; video.play();
    });
  }

  function play(on) {
    clearInterval(timer); timer = null;
    const button = document.querySelector('#matchPlay');
    if (on) timer = setInterval(() => { frame = (frame + 1) % data.skeleton.times.length; draw(); }, 110);
    if (button) { button.textContent = on ? 'Pause' : 'Play'; button.setAttribute('aria-pressed', String(on)); }
  }

  function detail() {
    const m = data.matches[chosen];
    const notes = (m.feedback || []).map(n => `<li>${esc(n.text)}</li>`).join('');
    const rows = (m.features || []).map(f => `<tr><th scope="row">${esc(f.label)}</th><td>${value(f.yours, f.unit)}</td><td>${value(f.theirs, f.unit)}</td><td>${Math.round(f.weight * 100)}%</td></tr>`).join('');
    document.querySelector('#matchDetail').innerHTML =
      `<h4 class="section-label">To look more like ${esc(m.name)}</h4>` +
      (notes ? `<ol class="match-notes">${notes}</ol>` : `<p class="cue">Nothing measured here is outside ${esc(m.name)}'s own shot-to-shot variation.</p>`) +
      `<details><summary>All measurements</summary><table class="match-table"><thead><tr><th scope="col">Measurement</th><th scope="col">You</th><th scope="col">${esc(m.name)}</th><th scope="col">Weight</th></tr></thead><tbody>${rows}</tbody></table><p class="report-note">Weight is how much of a measurement's variation is between players rather than between one player's own free throws. Based on ${m.reps} of ${esc(m.name)}'s free throws.</p></details>`;
    document.querySelectorAll('#matchList button').forEach((b, i) => b.setAttribute('aria-pressed', String(i === chosen)));
    videoControls();
    draw();
  }

  function render(result) {
    const root = document.querySelector('#formMatch'), match = result.form_match;
    play(false);
    root.classList.toggle('hidden', !match);
    if (!match) { data = null; videoControls(); return; }
    const notes = (match.notes || []).map(n => `<p class="cue">${esc(n)}</p>`).join('');
    if (!match.matches.length) { data = null; videoControls(); root.innerHTML = `<h3>Player match</h3>${notes}`; return; }
    data = match; chosen = 0; frame = match.skeleton.times.findIndex(t => t >= 0);
    const detailed = match.matches.filter(m => m.skeleton);
    const list = detailed.map((m, i) => `<li><button type="button" class="ghost" data-index="${i}" aria-pressed="false"><span>${esc(m.name)}</span><i style="width:${m.similarity}%"></i><b>${m.similarity}</b></button></li>`).join('');
    const steadier = (match.consistency || []).filter(c => c.yours > 1.5 * c.players).sort((a, b) => b.yours / b.players - a.yours / a.players).slice(0, 3);
    const consistency = match.reps < 2 ? '' : steadier.length
      ? `<p class="cue">Least repeatable from shot to shot: ${steadier.map(c => `${esc(c.label.toLowerCase())} (varies ${value(c.yours, c.unit)}, players ${value(c.players, c.unit)})`).join('; ')}.</p>`
      : '<p class="cue">Your free throws repeat about as closely as the players\' do on every measurement.</p>';
    root.innerHTML = `<h3>Player match</h3>
      <p class="report-note">Closest free-throw forms among ${match.matches.length} players, from your ${match.reps} free throw${match.reps === 1 ? '' : 's'} in this clip. Similarity is 0–100 on measured set point, release, timing and legs; it is resemblance, not quality.</p>
      ${notes}
      <div class="match-grid"><ol id="matchList" class="match-list">${list}</ol>
      <div class="match-stage"><canvas id="matchCanvas" width="560" height="380" role="img" aria-label="Your skeleton and the player's skeleton through the release"></canvas>
      <div class="match-controls"><button id="matchPlay" class="ghost" type="button" aria-pressed="false">Play</button><input id="matchScrub" type="range" min="0" max="${match.skeleton.times.length - 1}" aria-label="Time from release"><span id="matchTime"></span><button id="matchOverlay" class="ghost" type="button" aria-pressed="${overlaid}">Overlay</button></div></div></div>
      <div id="matchDetail"></div>${consistency}`;
    root.querySelectorAll('#matchList button').forEach(b => b.onclick = () => { chosen = Number(b.dataset.index); detail(); });
    root.querySelector('#matchPlay').onclick = () => play(!timer);
    root.querySelector('#matchOverlay').onclick = e => { overlaid = !overlaid; e.target.setAttribute('aria-pressed', String(overlaid)); draw(); };
    if (!raf) drawOnVideo();
    root.querySelector('#matchScrub').oninput = e => { play(false); frame = Number(e.target.value); draw(); };
    detail();
  }
  window.formMatch = { render };
})();
