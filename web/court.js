// Court calibration: pair landmarks on a half-court diagram with clicks on one video frame,
// preview the fitted court over the frame, and send the pairs with the analysis.
window.courtCalibration = (() => {
  const q = (s) => document.querySelector(s);
  const svgNS = 'http://www.w3.org/2000/svg';
  const scale = 6, pad = 14;  // diagram pixels per foot, margin
  let templates = null, standard = 'nba', marks = {}, selected = null, markedTime = null, fitted = null;

  fetch('/assets/court-template.json').then(r => r.json()).then(data => { templates = data; renderDiagram(); update(); })
    .catch(() => { q('#courtStatus').textContent = 'The court diagram could not be loaded; court calibration is unavailable.'; });

  const court = () => templates && templates[standard];
  const game = () => q('#analysisMode').value === 'one_on_one';
  const active = () => marking === 'landmarks';
  const onFrame = () => markedTime == null || Math.abs(preview.currentTime - markedTime) < .02;

  // The video is letterboxed inside its element (object-fit: contain); map through the shown picture.
  function picture() {
    const r = canvas.getBoundingClientRect(), vw = preview.videoWidth || 16, vh = preview.videoHeight || 9;
    const s = Math.min(r.width / vw, r.height / vh);
    return {r, vw, vh, w: vw * s, h: vh * s, x: (r.width - vw * s) / 2, y: (r.height - vh * s) / 2};
  }

  // Least-squares homography (h33 = 1) on Hartley-normalized points: court feet -> video pixels.
  function normalizer(points) {
    const cx = points.reduce((a, p) => a + p[0], 0) / points.length, cy = points.reduce((a, p) => a + p[1], 0) / points.length;
    const d = points.reduce((a, p) => a + Math.hypot(p[0] - cx, p[1] - cy), 0) / points.length || 1;
    const s = Math.SQRT2 / d;
    return [[s, 0, -s * cx], [0, s, -s * cy], [0, 0, 1]];
  }
  const mul = (a, b) => a.map((row, i) => b[0].map((_, j) => row.reduce((acc, _, k) => acc + a[i][k] * b[k][j], 0)));
  const apply = (h, p) => { const w = h[2][0] * p[0] + h[2][1] * p[1] + h[2][2];
    return w > 1e-9 ? [(h[0][0] * p[0] + h[0][1] * p[1] + h[0][2]) / w, (h[1][0] * p[0] + h[1][1] * p[1] + h[1][2]) / w] : null; };
  function inverse3(m) {
    const [a, b, c] = m[0], [d, e, f] = m[1], [g, h, i] = m[2];
    const A = e * i - f * h, B = f * g - d * i, C = d * h - e * g, det = a * A + b * B + c * C;
    if (Math.abs(det) < 1e-12) return null;
    return [[A / det, (c * h - b * i) / det, (b * f - c * e) / det], [B / det, (a * i - c * g) / det, (c * d - a * f) / det],
            [C / det, (b * g - a * h) / det, (a * e - b * d) / det]];
  }
  function solve(a, b) {  // Gaussian elimination with partial pivoting
    const n = b.length, m = a.map((row, i) => [...row, b[i]]);
    for (let c = 0; c < n; c++) {
      let p = c;
      for (let r = c + 1; r < n; r++) if (Math.abs(m[r][c]) > Math.abs(m[p][c])) p = r;
      if (Math.abs(m[p][c]) < 1e-12) return null;
      [m[c], m[p]] = [m[p], m[c]];
      for (let r = 0; r < n; r++) if (r !== c) { const f = m[r][c] / m[c][c]; for (let k = c; k <= n; k++) m[r][k] -= f * m[c][k]; }
    }
    return m.map((row, i) => row[n] / row[i]);
  }
  function homography(src, dst) {
    if (src.length < 4) return null;
    const ts = normalizer(src), td = normalizer(dst);
    const s = src.map(p => apply(ts, p)), d = dst.map(p => apply(td, p));
    const ata = Array.from({length: 8}, () => Array(8).fill(0)), atb = Array(8).fill(0);
    s.forEach(([x, y], i) => {
      const [u, v] = d[i];
      for (const [row, rhs] of [[[x, y, 1, 0, 0, 0, -u * x, -u * y], u], [[0, 0, 0, x, y, 1, -v * x, -v * y], v]]) {
        for (let r = 0; r < 8; r++) { atb[r] += row[r] * rhs; for (let c = 0; c < 8; c++) ata[r][c] += row[r] * row[c]; }
      }
    });
    const h = solve(ata, atb), tdInv = h && inverse3(td);
    if (!h || !tdInv) return null;
    return mul(mul(tdInv, [[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1]]), ts);
  }
  function generalPosition(points) {  // some 4 points with no 3 on one line (1 sq ft)
    const area = (a, b, c) => Math.abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) / 2;
    const n = points.length;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) for (let k = j + 1; k < n; k++) for (let l = k + 1; l < n; l++) {
      const quad = [points[i], points[j], points[k], points[l]];
      if ([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]].every(([a, b, c]) => area(quad[a], quad[b], quad[c]) > 1)) return true;
    }
    return false;
  }

  function computeFit() {
    fitted = null;
    const ids = Object.keys(marks), c = court();
    if (!c || ids.length < 4) return;
    const {vw, vh} = picture();
    const src = ids.map(id => c.landmarks[id].court), dst = ids.map(id => [marks[id][0] * vw, marks[id][1] * vh]);
    if (!generalPosition(src)) { fitted = {error: 'degenerate'}; return; }
    const h = homography(src, dst);
    if (!h) { fitted = {error: 'degenerate'}; return; }
    const errors = src.map((p, i) => { const q2 = apply(h, p); return q2 ? Math.hypot(q2[0] - dst[i][0], q2[1] - dst[i][1]) : Infinity; });
    let loo = null;
    if (ids.length >= 5) {
      loo = src.map((p, i) => {
        const keep = src.filter((_, j) => j !== i);
        if (!generalPosition(keep)) return null;
        const held = homography(keep, dst.filter((_, j) => j !== i)), q2 = held && apply(held, p);
        return q2 ? Math.hypot(q2[0] - dst[i][0], q2[1] - dst[i][1]) : null;
      });
    }
    fitted = {h, ids, errors, loo, far: Math.max(...src.map(p => p[1]))};
  }

  function renderDiagram() {
    const svg = q('#courtDiagram'), c = court();
    if (!svg || !c) return;
    const half = c.length / 2, w = c.width * scale + 2 * pad, h = half * scale + 2 * pad;
    const sx = x => (x + c.width / 2) * scale + pad, sy = y => y * scale + pad;
    svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
    svg.replaceChildren();
    const floor = document.createElementNS(svgNS, 'rect');
    Object.entries({x: pad, y: pad, width: c.width * scale, height: half * scale, class: 'court-floor'}).forEach(([k, v]) => floor.setAttribute(k, v));
    svg.append(floor);
    for (const line of c.lines) {
      const pts = line.filter(([, y]) => y <= half + .01);
      if (pts.length < 2) continue;
      const path = document.createElementNS(svgNS, 'polyline');
      path.setAttribute('points', pts.map(([x, y]) => `${sx(x)},${sy(y)}`).join(' '));
      path.setAttribute('class', 'court-line');
      svg.append(path);
    }
    const rim = document.createElementNS(svgNS, 'circle');
    Object.entries({cx: sx(c.rim[0]), cy: sy(c.rim[1]), r: 3, class: 'court-rim'}).forEach(([k, v]) => rim.setAttribute(k, v));
    svg.append(rim);
    for (const [id, mark] of Object.entries(c.landmarks)) {
      const g = document.createElementNS(svgNS, 'g');
      g.setAttribute('class', 'court-landmark');
      g.setAttribute('tabindex', '0');
      g.setAttribute('role', 'button');
      g.dataset.id = id;
      const title = document.createElementNS(svgNS, 'title');
      title.textContent = mark.label;
      const hit = document.createElementNS(svgNS, 'circle');
      Object.entries({cx: sx(mark.court[0]), cy: sy(mark.court[1]), r: 11, class: 'hit'}).forEach(([k, v]) => hit.setAttribute(k, v));
      const dot = document.createElementNS(svgNS, 'circle');
      Object.entries({cx: sx(mark.court[0]), cy: sy(mark.court[1]), r: 5, class: 'dot'}).forEach(([k, v]) => dot.setAttribute(k, v));
      g.append(title, hit, dot);
      const choose = () => { selected = id; marking = 'landmarks'; update(); };
      g.addEventListener('click', choose);
      g.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); choose(); } });
      svg.append(g);
    }
    update();
  }

  function update() {
    const panel = q('#courtPanel');
    q('#markLandmarks').classList.toggle('hidden', !game());
    if (!game()) { panel.classList.add('hidden'); if (active()) marking = 'rim'; }
    q('#courtDiagram')?.querySelectorAll('.court-landmark').forEach(g => {
      g.classList.toggle('selected', g.dataset.id === selected);
      g.classList.toggle('placed', g.dataset.id in marks);
      g.setAttribute('aria-label', `${court()?.landmarks[g.dataset.id].label}${g.dataset.id in marks ? ' (marked)' : ''}`);
    });
    computeFit();
    q('#courtStatus').textContent = statusText();
    q('#courtReturn').classList.toggle('hidden', onFrame());
    if (typeof drawBox === 'function') drawBox();
  }

  function statusText() {
    const c = court(), n = Object.keys(marks).length;
    if (!c) return '';
    const pick = selected ? `Now click “${c.landmarks[selected].label}” on the video.` : 'Pick a landmark on the diagram, then click the same spot on the video.';
    if (!onFrame()) return `Court marks belong to the frame at ${markedTime.toFixed(2)} s. Return to it to see or add to them.`;
    if (n < 4) return `${pick} ${n} of at least 4 marked (5 or more lets Ballform check the fit).`;
    if (fitted?.error) return `${pick} These marks cannot define the floor yet: 4 of them must not lie on one line.`;
    const rms = Math.sqrt(fitted.errors.reduce((a, e) => a + e * e, 0) / n);
    let text = `${pick} Error on the clicked points: ${rms.toFixed(1)} px RMS, ${Math.max(...fitted.errors).toFixed(1)} px max.`;
    if (!fitted.loo) text += ' With exactly 4 points the fit always passes through them, so its error cannot be checked; add a fifth.';
    else {
      const valid = fitted.loo.map((e, i) => [e, fitted.ids[i]]).filter(([e]) => e != null);
      if (valid.length) {
        const [worst, id] = valid.reduce((a, b) => (b[0] > a[0] ? b : a));
        text += ` Leaving each point out moves it by up to ${worst.toFixed(0)} px (${c.landmarks[id].label}); `
          + 'a large value is expected for a lone far landmark, but next to other marks it suggests a misclick.';
      }
    }
    if (fitted.far < c.length / 4) text += ' All marks are near the basket, so far-court distances are extrapolated; add a half-court or far-sideline landmark if one is visible.';
    return text + ' Check that the drawn lines sit on the court.';
  }

  function place(e) {
    if (!court() || !selected) { q('#courtStatus').textContent = 'Pick a landmark on the diagram first.'; return; }
    if (!onFrame()) return;
    const {r, w, h, x, y} = picture();
    const nx = (e.clientX - r.left - x) / w, ny = (e.clientY - r.top - y) / h;
    if (nx < 0 || nx > 1 || ny < 0 || ny > 1) return;
    markedTime ??= preview.currentTime;
    marks[selected] = [nx, ny];
    // Move on to the next unmarked landmark so pairs can be placed quickly.
    const ids = Object.keys(court().landmarks);
    selected = ids.slice(ids.indexOf(selected) + 1).find(id => !(id in marks)) || null;
    update();
  }

  function draw(ctx) {
    if (!court() || !Object.keys(marks).length || !onFrame()) return;
    const {vw, vh, w, h, x, y} = picture(), d = devicePixelRatio;
    const toCanvas = ([px, py]) => [(x + px / vw * w) * d, (y + py / vh * h) * d];
    ctx.save();
    if (fitted?.h) {
      ctx.strokeStyle = 'rgba(80,212,237,.9)'; ctx.lineWidth = 1.5 * d; ctx.setLineDash([]);
      for (const line of court().lines) {
        ctx.beginPath();
        let open = false;
        for (let i = 0; i + 1 < line.length; i++) {
          const [a, b] = [line[i], line[i + 1]], steps = Math.max(1, Math.ceil(Math.hypot(b[0] - a[0], b[1] - a[1])));
          for (let s = 0; s <= steps; s++) {
            const p = apply(fitted.h, [a[0] + (b[0] - a[0]) * s / steps, a[1] + (b[1] - a[1]) * s / steps]);
            if (!p || Math.abs(p[0] - vw / 2) > 2 * vw || Math.abs(p[1] - vh / 2) > 2 * vh) { open = false; continue; }
            const [cx, cy] = toCanvas(p);
            if (open) ctx.lineTo(cx, cy); else ctx.moveTo(cx, cy);
            open = true;
          }
        }
        ctx.stroke();
      }
    }
    ctx.font = `${11 * d}px DM Mono, monospace`;
    Object.entries(marks).forEach(([id, [nx, ny]], i) => {
      const [cx, cy] = toCanvas([nx * vw, ny * vh]);
      ctx.fillStyle = '#fa5a24'; ctx.beginPath(); ctx.arc(cx, cy, 4 * d, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = '#fff'; ctx.fillText(String(i + 1), cx + 6 * d, cy - 6 * d);
    });
    ctx.restore();
  }

  // Result page: name the new metrics and say how the calibration held up.
  Object.assign(labels, {shot_distance_ft: 'Shot distance (floor)', shot_zone: 'Shot zone',
    shooter_court_x_ft: 'Shooter across court (from lane centre)', shooter_court_y_ft: 'Shooter from baseline',
    separation_ft: 'Floor separation at release', contest_clearance_ft: 'Contest clearance (approx., feet)',
    visible_hand_clearance_ft: 'Visible hand clearance (approx., feet)'});
  function report(result) {
    const summary = result.court_calibration, note = q('#courtNote');
    note.classList.toggle('hidden', !summary);
    if (!summary) return;
    const error = summary.clicked_error_px;
    note.textContent = `Court calibration: ${summary.standard.toUpperCase()} lines, ${summary.points} landmarks, `
      + `${error.rms} px RMS error on the clicked points; the floor mapping held on ${summary.reliable_frames} of `
      + `${summary.frames} analyzed frames. Metrics in feet are measured on the floor; the torso-length metrics are `
      + 'unchanged. Check the court lines drawn in the video.';
  }

  function clear() { marks = {}; markedTime = null; selected = null; update(); }

  function append(form) {
    if (!game() || !court() || !fitted?.h) return;
    form.append('court_landmarks', JSON.stringify({standard, time_s: markedTime,
      points: Object.entries(marks).map(([id, image]) => ({id, image}))}));
  }

  q('#markLandmarks').addEventListener('click', () => { marking = 'landmarks'; q('#courtPanel').classList.remove('hidden'); update(); });
  q('#courtStandard').addEventListener('change', e => { standard = e.target.value; marks = {}; markedTime = null; selected = null; renderDiagram(); });
  q('#courtClear').addEventListener('click', clear);
  q('#courtReturn').addEventListener('click', () => { if (markedTime != null) preview.currentTime = markedTime; });
  q('#analysisMode').addEventListener('change', update);
  preview.addEventListener('seeked', update);
  preview.addEventListener('loadedmetadata', clear);
  return {place, draw, clear, append, report};
})();
