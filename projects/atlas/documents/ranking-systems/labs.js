'use strict';
const number = (id, min, max) => {
  const node = document.getElementById(id);
  if (node.value.trim() === '') return NaN;
  const value = Number(node.value);
  return Number.isFinite(value) && value >= min && value <= max ? value : NaN;
};
function connect(id, update) {
  const lab = document.getElementById(id);
  if (lab) { lab.addEventListener('input', update); lab.addEventListener('change', update); update(); }
}
connect('budget-lab', () => {
  const n = number('candidates',100,5000), cost = number('cost',0.001,1);
  document.getElementById('candidate-value').textContent = n;
  const time = 90 + n*cost;
  document.getElementById('budget-result').textContent = Number.isFinite(time) ? `${time.toFixed(0)} ms modeled total · ${time<=200 ? 'within' : 'above'} the 200 ms target` : 'Enter a cost between 0.001 and 1 ms.';
});
connect('vote-lab', () => {
  const up = number('upvotes',0,100000), down = number('downvotes',0,100000), n = up+down;
  let result = 'Enter whole nonnegative vote counts.';
  if (Number.isInteger(up) && Number.isInteger(down)) {
    const p = n ? up/n : 0, z = 1.96;
    const lower = n ? (p+z*z/(2*n)-z*Math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n) : 0;
    result = `${n ? (p*100).toFixed(1)+'% approval' : 'No votes'} · Wilson lower bound ${lower.toFixed(3)}`;
  }
  document.getElementById('vote-result').textContent = result;
});
connect('ads-lab', () => {
  const ctr = number('ctr',0,1), cvr = number('cvr',0,1), bid = number('bid',0,10000);
  const cpa = document.getElementById('bid-basis').value === 'cpa';
  const val = 1000*ctr*(cpa ? cvr : 1)*bid;
  document.getElementById('cvr').disabled = !cpa;
  document.getElementById('ads-result').textContent = Number.isFinite(val) ? `${val.toFixed(2)} currency / 1,000 impressions` : 'Enter valid probabilities and a nonnegative bid.';
});
