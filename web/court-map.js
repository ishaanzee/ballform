// Top-down court map beside the review video: where each tracked player stands on the calibrated
// floor at the video's current time. Positions come from /api/jobs/<id>/court-map (app/court_map.py).
window.courtMap = (() => {
  const svgNS = 'http://www.w3.org/2000/svg', q = s => document.querySelector(s);
  const UNIT = 10, PAD = 12, TRAIL_S = 1.5;   // svg units per foot
  let templates = null, data = null, players = [], lastTime = -1, token = 0;
  const loadTemplates = () => templates ? Promise.resolve(templates)
    : fetch('/assets/court-template.json').then(r => r.json()).then(t => (templates = t));
  const el = (name, attrs = {}, parent) => {
    const node = document.createElementNS(svgNS, name);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    if (parent) parent.append(node);
    return node;
  };

  function hide() {
    data = null; players = [];
    q('#courtMap').classList.add('hidden');
    q('#reviewRow').classList.remove('with-map');
  }

  async function render(id, result) {
    const mine = ++token;
    hide();
    if (result.mode !== 'one_on_one' || !result.court_calibration) return;
    let payload, courts;
    try {
      const response = await fetch(apiUrl(`/api/jobs/${id}/court-map`));
      if (!response.ok) return;
      [payload, courts] = [await response.json(), await loadTemplates()];
    } catch { return; }
    const court = courts[payload.standard];
    if (mine !== token || !court || !payload.tracks.length) return;
    data = payload;
    build(court);
    q('#courtMap').classList.remove('hidden');
    q('#reviewRow').classList.add('with-map');
    q('#courtMapNote').textContent = `Feet on the calibrated floor, smoothed over time, turned to face the way the camera does. Dots take the jersey colour seen `
      + `in the video. The floor mapping held on ${data.mapped_frames} of ${data.frames} frames; a player is not drawn `
      + `where it did not, where the feet were not tracked, or beyond half court.`;
    lastTime = -1;
  }

  // Court feet -> svg. The half court is turned by quarter turns so that the side nearest the camera is at
  // the bottom, as in the video; without a camera direction the baseline is at the top, as on the diagram.
  let turn = 0;
  const xy = (x, y, court) => {
    const w = court.width, half = court.length / 2;
    const [u, v] = turn === 0 ? [x + w / 2, y] : turn === 1 ? [half - y, x + w / 2] : turn === 2 ? [w / 2 - x, half - y] : [y, w / 2 - x];
    return [u * UNIT + PAD, v * UNIT + PAD];
  };
  const pair = (x, y, court) => xy(x, y, court).join(',');

  function build(court) {
    const svg = q('#courtMapSvg'), half = court.length / 2, [cx, cy] = data.toward_camera || [0, 1];
    turn = Math.abs(cy) >= Math.abs(cx) ? (cy >= 0 ? 0 : 2) : (cx >= 0 ? 1 : 3);
    const [wide, tall] = turn % 2 ? [half, court.width] : [court.width, half];
    svg.setAttribute('viewBox', `0 0 ${wide * UNIT + 2 * PAD} ${tall * UNIT + 2 * PAD}`);
    svg.replaceChildren();
    el('rect', {x: PAD, y: PAD, width: wide * UNIT, height: tall * UNIT, class: 'court-floor'}, svg);
    for (const line of court.lines) {
      const points = line.filter(([, y]) => y <= half + .01);
      if (points.length >= 2) el('polyline', {points: points.map(([x, y]) => pair(x, y, court)).join(' '), class: 'court-line'}, svg);
    }
    const [rx, ry] = xy(court.rim[0], court.rim[1], court);
    el('circle', {cx: rx, cy: ry, r: 7, class: 'court-rim'}, svg);
    const trails = el('g', {class: 'map-trails'}, svg), dots = el('g', {}, svg);
    players = data.tracks.map(track => {
      const group = el('g', {class: 'map-player'}, dots);
      const trail = el('polyline', {class: 'map-trail', stroke: track.colour || '#777'}, trails);
      const ring = el('circle', {r: 17, class: 'map-ring'}, group);
      el('circle', {r: 10, class: 'map-dot', fill: track.colour || '#777'}, group);
      const label = el('text', {class: 'map-label', x: 14, y: -12}, group);
      label.textContent = `P${track.id}`;
      return {track, group, trail, ring, court};
    });
  }

  // Position of a track at a fractional frame, or null when it has no point there.
  function at(track, frame) {
    const f = track.frames;
    if (frame < f[0] || frame > f[f.length - 1]) return null;
    let lo = 0, hi = f.length - 1;
    while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (f[mid] <= frame) lo = mid; else hi = mid; }
    if (f[hi] - f[lo] > 1 && frame !== f[lo]) return frame === f[hi] ? [track.x[hi], track.y[hi]] : null;   // never across a break
    const u = f[hi] === f[lo] ? 0 : (frame - f[lo]) / (f[hi] - f[lo]);
    return [track.x[lo] + (track.x[hi] - track.x[lo]) * u, track.y[lo] + (track.y[hi] - track.y[lo]) * u];
  }

  function draw() {
    requestAnimationFrame(draw);
    const video = q('#annotated');
    if (!data || !video || video.currentTime === lastTime) return;
    const now = lastTime = video.currentTime, frame = now * data.fps, trails = q('#courtMapTrails').checked;
    const shot = data.shots.find(s => s.release_s != null && now >= (s.start_s ?? s.release_s) - .5 && now <= (s.end_s ?? s.release_s + 1));
    for (const {track, group, trail, ring, court} of players) {
      const half = court.length / 2, inside = p => p && p[1] <= half && Math.abs(p[0]) <= court.width / 2 + 3 && p[1] >= -3;
      const point = at(track, frame);
      group.style.display = inside(point) ? '' : 'none';
      trail.style.display = inside(point) && trails ? '' : 'none';
      if (!inside(point)) continue;
      group.setAttribute('transform', `translate(${pair(point[0], point[1], court)})`);
      const role = !shot ? '' : track.id === shot.shooter ? 'shooter' : track.id === shot.defender ? 'defender' : '';
      ring.setAttribute('class', `map-ring ${role}`);
      if (trails) {
        const path = [];
        for (let back = Math.round(TRAIL_S * data.fps); back >= 0; back -= 2) {
          const past = at(track, frame - back);
          if (inside(past)) path.push(pair(past[0], past[1], court)); else path.length = 0;   // only the unbroken stretch up to now
        }
        trail.setAttribute('points', path.join(' '));
      }
    }
    const name = id => id == null ? 'not identified' : `P${id}`;
    q('#courtMapShot').textContent = shot
      ? `Shot ${shot.number}: shooter ${name(shot.shooter)} (orange ring), defender ${name(shot.defender)} (dark ring).` : '';
  }
  requestAnimationFrame(draw);
  return {render};
})();
