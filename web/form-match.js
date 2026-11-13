// Player match panel for form_match reports: closest players, what to change, measurements
// side by side and the two skeletons played through the release.
(() => {
  const BONES = [['left_shoulder', 'right_shoulder'], ['left_shoulder', 'left_elbow'], ['left_elbow', 'left_wrist'],
    ['right_shoulder', 'right_elbow'], ['right_elbow', 'right_wrist'], ['left_shoulder', 'left_hip'],
    ['right_shoulder', 'right_hip'], ['left_hip', 'right_hip'], ['left_hip', 'left_knee'], ['left_knee', 'left_ankle'],
    ['right_hip', 'right_knee'], ['right_knee', 'right_ankle']];
  const esc = s => { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; };
  const value = (v, unit) => v == null ? '—' : unit === '°' ? `${Math.round(v)}°` : unit === 's' ? `${v.toFixed(2)} s` : `${v.toFixed(2)} torso`;
  let data = null, chosen = 0, frame = 0, timer = null;

  function draw() {
    const canvas = document.querySelector('#matchCanvas');
    if (!canvas || !data) return;
    const ctx = canvas.getContext('2d'), w = canvas.width, h = canvas.height, times = data.skeleton.times;
    ctx.clearRect(0, 0, w, h);
    const style = getComputedStyle(document.documentElement);
    const ink = style.getPropertyValue('--ink') || '#171a19', orange = style.getPropertyValue('--orange') || '#fa5a24';
    const scale = h / 5.4;   // torso lengths: standing reach is about 4.6
    const figures = [[data.skeleton.yours, w * .27, orange, 'You'], [data.matches[chosen].skeleton, w * .73, ink, data.matches[chosen].name]];
    for (const [frames, cx, color, name] of figures) {
      ctx.fillStyle = ink; ctx.font = '700 12px DM Mono, monospace'; ctx.textAlign = 'center';
      ctx.fillText(name.toUpperCase(), cx, 18);
      ctx.strokeStyle = style.getPropertyValue('--line') || '#c9c5bb'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(cx - w * .2, h - 14); ctx.lineTo(cx + w * .2, h - 14); ctx.stroke();
      const joints = frames && frames[frame];
      if (!joints) { ctx.fillStyle = style.getPropertyValue('--muted') || '#696c68'; ctx.fillText('not in view', cx, h / 2); continue; }
      const at = name => [cx + joints[name][0] * scale, h - 14 - joints[name][1] * scale];
      ctx.strokeStyle = color; ctx.lineWidth = 4; ctx.lineCap = 'round';
      for (const [a, b] of BONES) { ctx.beginPath(); ctx.moveTo(...at(a)); ctx.lineTo(...at(b)); ctx.stroke(); }
      const [nx, ny] = at('nose');
      ctx.beginPath(); ctx.arc(nx, ny, scale * .22, 0, Math.PI * 2); ctx.stroke();
    }
    const t = times[frame];
    document.querySelector('#matchTime').textContent = Math.abs(t) < .001 ? 'release' : `${t > 0 ? '+' : '−'}${Math.abs(t).toFixed(2)} s`;
    document.querySelector('#matchScrub').value = frame;
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
    draw();
  }

  function render(result) {
    const root = document.querySelector('#formMatch'), match = result.form_match;
    play(false);
    root.classList.toggle('hidden', !match);
    if (!match) { data = null; return; }
    const notes = (match.notes || []).map(n => `<p class="cue">${esc(n)}</p>`).join('');
    if (!match.matches.length) { data = null; root.innerHTML = `<h3>Player match</h3>${notes}`; return; }
    data = match; chosen = 0; frame = match.skeleton.times.findIndex(t => t >= 0);
    const detailed = match.matches.filter(m => m.skeleton);
    const list = detailed.map((m, i) => `<li><button type="button" class="ghost" data-index="${i}" aria-pressed="false"><span>${esc(m.name)}</span><i style="width:${m.similarity}%"></i><b>${m.similarity}</b></button></li>`).join('');
    const steadier = (match.consistency || []).filter(c => c.yours > 1.5 * c.players).sort((a, b) => b.yours / b.players - a.yours / a.players).slice(0, 3);
    const consistency = match.reps < 2 ? '' : steadier.length
      ? `<p class="cue">Least repeatable from shot to shot: ${steadier.map(c => `${esc(c.label.toLowerCase())} (varies ${value(c.yours, c.unit)}, players ${value(c.players, c.unit)})`).join('; ')}.</p>`
      : '<p class="cue">Your free throws repeat about as closely as the players\' do on every measurement.</p>';
    root.innerHTML = `<h3>Player match</h3>
      <p class="report-note">Closest free-throw forms among ${match.matches.length} players, from ${match.reps} of your free throw${match.reps === 1 ? '' : 's'}. Similarity is 0–100 on measured set point, release, timing and legs; it is resemblance, not quality.</p>
      ${notes}
      <div class="match-grid"><ol id="matchList" class="match-list">${list}</ol>
      <div class="match-stage"><canvas id="matchCanvas" width="560" height="380" role="img" aria-label="Your skeleton and the player's skeleton through the release"></canvas>
      <div class="match-controls"><button id="matchPlay" class="ghost" type="button" aria-pressed="false">Play</button><input id="matchScrub" type="range" min="0" max="${match.skeleton.times.length - 1}" aria-label="Time from release"><span id="matchTime"></span></div></div></div>
      <div id="matchDetail"></div>${consistency}`;
    root.querySelectorAll('#matchList button').forEach(b => b.onclick = () => { chosen = Number(b.dataset.index); detail(); });
    root.querySelector('#matchPlay').onclick = () => play(!timer);
    root.querySelector('#matchScrub').oninput = e => { play(false); frame = Number(e.target.value); draw(); };
    detail();
  }
  window.formMatch = { render };
})();
