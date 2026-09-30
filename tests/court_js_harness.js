// Runs web/court.js against stub DOM globals: clicks landmarks, prints the status and the form field.
// Usage: node court_js_harness.js <repo root> '<json [[id, x_px, y_px], ...]>' [video width] [video height] [proposal json]
// Actions may also be ["drag", x0, y0, x1, y1] or ["remove", id]; a proposal (the /api/court/detect
// response) is applied before them.
const fs = require('fs'), path = require('path'), vm = require('vm');
const [root, clicksJson, width = '1920', height = '1080', proposalJson = ''] = process.argv.slice(2);
const created = [], elements = {};
function element(name) {
  const listeners = {};
  return {name, value: name === '#analysisMode' ? 'one_on_one' : '', textContent: '', dataset: {}, children: [],
    classList: {toggle() {}, add() {}, remove() {}}, setAttribute() {}, replaceChildren() { this.children = []; },
    append(...kids) { this.children.push(...kids); }, querySelectorAll: () => created.filter(e => e.dataset.id),
    addEventListener(type, fn) { (listeners[type] ||= []).push(fn); }, fire(type, event = {}) { (listeners[type] || []).forEach(fn => fn(event)); }};
}
const document = {
  querySelector: s => (elements[s] ||= element(s)),
  createElementNS: (_, tag) => { const e = element(tag); created.push(e); return e; },
};
const template = JSON.parse(fs.readFileSync(path.join(root, 'web', 'court-template.json'), 'utf8'));
const context = vm.createContext({
  document, window: {}, console, Math, JSON, Object, Array, Number, String, Promise, devicePixelRatio: 1,
  fetch: () => Promise.resolve({json: () => template}),
  preview: {currentTime: 0, videoWidth: Number(width), videoHeight: Number(height), addEventListener() {}},
  canvas: {getBoundingClientRect: () => ({left: 0, top: 0, width: Number(width), height: Number(height)})},
});
vm.runInContext('var marking = "rim"; var labels = {}; function drawBox() {}', context);
vm.runInContext(fs.readFileSync(path.join(root, 'web', 'court.js'), 'utf8'), context);
setTimeout(() => {
  const calibration = context.window.courtCalibration;
  if (proposalJson) calibration.applyProposal(JSON.parse(proposalJson), 0);
  for (const [id, ...args] of JSON.parse(clicksJson)) {
    if (id === 'drag') {
      const [x0, y0, x1, y1] = args;
      calibration.place({clientX: x0, clientY: y0});
      calibration.move({clientX: x1, clientY: y1});
      calibration.release();
    } else if (id === 'remove') {
      created.find(e => e.dataset.id === args[0]).fire('click');
      calibration.remove();
    } else {
      created.find(e => e.dataset.id === id).fire('click');
      calibration.place({clientX: args[0], clientY: args[1]});
    }
  }
  const form = {fields: {}, append(k, v) { this.fields[k] = v; }};
  calibration.append(form);
  console.log(JSON.stringify({status: elements['#courtStatus'].textContent, marking: context.marking, source: calibration.source(),
    field: form.fields.court_landmarks ? JSON.parse(form.fields.court_landmarks) : null}));
}, 0);
