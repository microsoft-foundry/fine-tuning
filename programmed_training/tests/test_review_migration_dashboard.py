"""Compile and exercise the actual dashboard JavaScript offline, using Node."""

from pathlib import Path
import shutil
import subprocess

import pytest

from programmed_training.utils.ml_log import JsonLogger

ROOT = Path(__file__).resolve().parents[1]

# No browser, HTTP server, CDN, npm package, or paid resource is used. The tiny
# DOM supplies only the surface touched by initialization, clearRunUI and KPIs.
NODE_CHECK = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync(process.argv[2], 'utf8');
const scripts = [...html.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/gi)]
  .filter(match => !/\bsrc\s*=/.test(match[1])).map(match => match[2]);
assert.equal(scripts.length, 1, 'Check the complete inline application script');
const elements = new Map();
const classes = new Map();
const element = id => {
  if (!elements.has(id)) {
    const value = {textContent: ''};
    const details = [{textContent: ''}, {textContent: ''}];
    classes.set(id, new Set());
    elements.set(id, {
      textContent: '', innerHTML: '', value: '0', style: {}, dataset: {},
      classList: {
        add(name) { classes.get(id).add(name); },
        remove(name) { classes.get(id).delete(name); },
        toggle(name, enabled) {
          if (enabled ?? !classes.get(id).has(name)) this.add(name);
          else this.remove(name);
        },
      },
      addEventListener() {},
      querySelector(selector) {
        assert.ok(selector === '.v' || selector === '.d');
        return selector === '.v' ? value : details[0];
      },
      querySelectorAll(selector) { assert.equal(selector, '.d'); return details; },
    });
  }
  return elements.get(id);
};
const cards = ['plotReward', 'plotOptim', 'plotKL'].map(id => {
  const card = {id};
  element(id).parentElement = card;
  return card;
});
const grid = {
  children: [...cards],
  insertBefore(card, reference) {
    assert.ok(this.children.includes(card));
    assert.ok(reference == null || this.children.includes(reference));
    if (card === reference) return;
    this.children.splice(this.children.indexOf(card), 1);
    this.children.splice(reference == null ? this.children.length : this.children.indexOf(reference), 0, card);
  },
};
for (const card of cards) {
  card.parentElement = grid;
  Object.defineProperty(card, 'nextSibling', {
    get() { return grid.children[grid.children.indexOf(card) + 1] ?? null; },
  });
}
const context = vm.createContext({
  document: {
    getElementById: element,
    querySelector: element,
    body: {classList: {toggle() {}}},
  },
  location: {search: '', hash: ''},
  URL, URLSearchParams, window: {},
  fetch: async url => {
    assert.match(url, /^\/api\/runs\?/);
    return {ok: true, json: async () => ({root: 'offline', roots: [], runs: []})};
  },
  setInterval() { throw new Error('Tests must not start refresh timers'); },
  clearInterval() {},
});
// vm.Script parses the ENTIRE inline script, not a copied KPI implementation.
new vm.Script(scripts[0], {filename: 'dashboard-inline.js'}).runInContext(context);
const kpi = id => {
  const node = element(id);
  return [node.querySelector('.v').textContent, node.querySelectorAll('.d')[1].textContent];
};
async function render(mode, rows, partialTail = false, meta = null, run = 'offline/run') {
  vm.runInContext(
    `CURRENT_RUN = ${JSON.stringify(run)}; RUN_META = ${JSON.stringify(meta)}; applyMode('${mode}');`,
    context,
  );
  const text = rows.map(row => JSON.stringify(row)).join('\n') + '\n' +
    (partialTail ? '{"step":' : '');
  context.fetch = async url => {
    assert.ok(url.startsWith('/api/run/' + run + '/file/metrics.jsonl?'));
    return {ok: true, status: 200, text: async () => text};
  };
  const parsed = await context.loadMetrics();
  assert.deepEqual(JSON.parse(JSON.stringify(parsed)), rows);
  context.renderKPIs(parsed);
}
async function checkNumerics() {
  for (const [meta, run] of [
    [{recipe: 'rl_numerics_check'}, 'offline/run'],
    [{component: 'rl_numerics_check'}, 'offline/run'],
    [null, 'offline/rl_numerics_check'],
  ]) {
    const renderNumerics = rows => render('rl', rows, false, meta, run);
    await renderNumerics([
      {step: 0, 'optim/kl_sample_train_v1': -0.02, 'optim/kl_sample_train_v2': 0.001},
      {step: 1, 'optim/kl_sample_train_v1': 0, 'optim/kl_sample_train_v2': 0},
      {step: 1, 'optim/lr': 0.00004},
    ]);
    assert.deepEqual(kpi('kpiTrainAcc'), ['0', 'step 1 · diagnostic only']);
    assert.deepEqual(kpi('kpiTestAcc'), ['0', 'step 1 · diagnostic only']);
    assert.equal(element('kpiTrainAcc').querySelector('.d').textContent, 'Sampler/trainer KL v1 (primary)');
    assert.equal(element('kpiTestAcc').querySelector('.d').textContent, 'Sampler/trainer KL v2');
    assert.equal(classes.get('kpiTrainAcc').has('good'), false);
    assert.deepEqual(grid.children.map(card => card.id), ['plotKL', 'plotReward', 'plotOptim']);

    await renderNumerics([{step: 0, 'optim/kl_sample_train_v1': -0.02}]);
    assert.deepEqual(kpi('kpiTrainAcc'), ['-0.020000', 'step 0 · diagnostic only']);
    assert.deepEqual(kpi('kpiTestAcc'), ['–', 'step 0 · not recorded']);
    for (const [current, state] of [
      [{}, 'not recorded'],
      [{'optim/kl_sample_train_v1': null}, 'not recorded'],
      [{'optim/kl_sample_train_v1': 'invalid'}, 'invalid measurement'],
    ]) {
      await renderNumerics([
        {step: 0, 'optim/kl_sample_train_v1': 0.02},
        {step: 1, ...current},
      ]);
      assert.deepEqual(kpi('kpiTrainAcc'), ['–', 'step 1 · ' + state]);
    }

    await render('rl', [{step: 0, 'env/all/correct': 0.5}]);
    assert.equal(kpi('kpiTrainAcc')[0], '50.0%');
    assert.equal(element('kpiTrainAcc').querySelector('.d').textContent, 'Train correct');
    assert.equal(element('kpiTestAcc').querySelector('.d').textContent, 'Eval correct');
    assert.equal(classes.get('kpiTrainAcc').has('good'), true);
    assert.deepEqual(grid.children.map(card => card.id), ['plotReward', 'plotOptim', 'plotKL']);
  }
}
async function check() {
  const loggerText = fs.readFileSync(process.argv[4], 'utf8');
  context.fetch = async url => {
    assert.ok(url.startsWith('/api/run/offline/run/file/metrics.jsonl?'));
    return {ok: true, status: 200, text: async () => loggerText};
  };
  vm.runInContext("CURRENT_RUN = 'offline/run'; RUN_META = {recipe: 'rl_numerics_check'}; applyMode('rl');", context);
  const loggedRows = await context.loadMetrics();
  assert.equal(loggedRows.length, 2, 'The actual invalid logger row must survive ingestion');
  assert.equal(loggedRows[1].step, 1);
  for (const key of ['optim/kl_sample_train_v1', 'optim/kl_sample_train_v2']) {
    assert.ok(['NaN', 'Infinity', '-Infinity'].includes(loggedRows[1][key].__nonfinite__));
    assert.deepEqual(JSON.parse(JSON.stringify(context.pick(loggedRows, key))), [0.02, null]);
  }
  context.renderKPIs(loggedRows);
  assert.deepEqual(kpi('kpiTrainAcc'), ['–', 'step 1 · invalid measurement']);
  assert.deepEqual(kpi('kpiTestAcc'), ['–', 'step 1 · invalid measurement']);
  if (process.argv[5]) {
    const payload = JSON.parse(fs.readFileSync(process.argv[5], 'utf8'));
    await render('rl', [
      {step: 25, 'optim/kl_sample_train_v1': 0.02, 'optim/kl_sample_train_v2': 0.01},
    ], false, payload.meta, 'offline/rl_numerics_check');
    assert.equal(kpi('kpiTrainAcc')[0], '0.020000');
    context.fetch = async () => ({ok: true, status: 200, text: async () => payload.metrics_jsonl});
    const rows = await context.loadMetrics();
    if (rows.length === 0) {
      context.loadRuns = async () => {};
      context.loadLogTail = async () => {};
      await context.refresh();
      assert.deepEqual(kpi('kpiTrainAcc'), ['–', 'final attempt · no updates recorded']);
      assert.deepEqual(kpi('kpiTestAcc'), ['–', 'final attempt · no updates recorded']);
      assert.equal(kpi('kpiStep')[0], '–');
      assert.equal(element('kpiProg').style.width, '0%');
    } else {
      context.renderKPIs(rows);
      assert.deepEqual(kpi('kpiTrainAcc'), payload.expected_v1);
      assert.deepEqual(kpi('kpiTestAcc'), payload.expected_v2);
      assert.equal(kpi('kpiStep')[0], String(rows.at(-1).step));
    }
  }
  await checkNumerics();
  if (process.argv[3] === 'numerics') {
    console.log('dashboard inline syntax and offline KPI fixtures passed');
    return;
  }
  // Evaluation-only final rows, duplicate steps, setup rows, nulls, and a
  // partially-written JSONL tail must not erase the latest training metric.
  await render('rl', [
    {step: 0, 'env/all/correct': 0.1, 'env/all/reward/total': 0.2},
    {step: 0, 'test/env/all/correct': 0.25, 'test/env/all/reward/total': 0.3},
    {step: 1, 'env/all/correct': 0.6, 'env/all/reward/total': 0.65},
    {step: 1, 'optim/lr': 0.00002},
    {step: 2, 'test/env/all/correct': 0.5, 'test/env/all/reward/total': 0.55},
    {step: 2, 'env/all/correct': null, 'test/env/all/correct': null},
  ], true);
  assert.deepEqual(kpi('kpiTrainAcc'), ['60.0%', '+50.00 pp vs prev train']);
  assert.deepEqual(kpi('kpiTestAcc'), ['50.0%', '+25.00 pp vs prev eval']);

  // The reverse event order works too; zero is a real value, not missing.
  await render('rl', [
    {step: 0, 'env/all/correct': 0.8, 'test/env/all/correct': 0.6},
    {step: 1, 'test/env/all/correct': 0},
    {step: 1, 'env/all/correct': 0},
    {step: 1, 'time/total': 1},
  ]);
  assert.deepEqual(kpi('kpiTrainAcc'), ['0.0%', '-80.00 pp vs prev train']);
  assert.deepEqual(kpi('kpiTestAcc'), ['0.0%', '-60.00 pp vs prev eval']);

  // Reward fallback has its own latest row; it need not share correctness.
  await render('rl', [
    {step: 0, 'env/all/reward/total': 0.3},
    {step: 1, 'env/all/correct': 0.2},
    {step: 2, 'test/env/all/reward/total': 0.4},
    {step: 3, 'test/env/all/correct': 0.5},
  ]);
  assert.deepEqual(kpi('kpiTrainAcc'), ['20.0%', 'reward 0.300']);
  assert.deepEqual(kpi('kpiTestAcc'), ['50.0%', 'reward 0.400']);

  // Absent train data remains absent, never borrowed from eval or shown as 0.
  await render('rl', [
    {step: 0, 'test/env/all/correct': 0, 'test/env/all/reward/total': 0},
    {step: 1, 'env/all/correct': null},
  ]);
  assert.deepEqual(kpi('kpiTrainAcc'), ['–', 'reward –']);
  assert.deepEqual(kpi('kpiTestAcc'), ['0.0%', 'reward 0']);

  // Preserve the existing independent SFT train/eval selection and deltas.
  await render('sft', [
    {step: 0, train_mean_nll: 2},
    {step: 0, 'test/nll': 3},
    {step: 1, train_mean_nll: 1.5},
    {step: 2, 'test/nll': 2.25, train_mean_nll: null},
  ]);
  assert.deepEqual(kpi('kpiTrainNLL'), ['1.5000', '-0.5000 from start']);
  assert.deepEqual(kpi('kpiTestNLL'), ['2.2500', '-0.7500 vs prev eval']);
  context.clearRunUI();
  context.renderKPIs([]);
  assert.deepEqual(kpi('kpiTrainAcc'), ['–', '']);
  assert.deepEqual(kpi('kpiTrainNLL'), ['–', '']);
  console.log('dashboard inline syntax and offline KPI fixtures passed');
}
check().catch(error => { console.error(error); process.exitCode = 1; });
"""


@pytest.mark.parametrize("dashboard,checks", [
    (ROOT / "dashboard.html", "all"),
    (ROOT.parent / "loom-bench" / "single_run.html", "numerics"),
], ids=["cookbook", "benchmark"])
@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")], ids=["nan", "inf", "negative-inf"])
def test_actual_dashboard_javascript_with_interleaved_metric_fixtures(dashboard, checks, nonfinite, tmp_path):
    if not dashboard.is_file():
        pytest.skip("The benchmark dashboard is available only in the source checkout")
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for offline dashboard JavaScript validation")
    logger = JsonLogger(tmp_path)
    logger.log_metrics({
        "optim/kl_sample_train_v1": 0.02, "optim/kl_sample_train_v2": 0.02,
    }, step=0)
    logger.log_metrics({
        "optim/kl_sample_train_v1": nonfinite, "optim/kl_sample_train_v2": nonfinite,
    }, step=1)
    result = subprocess.run(
        [node, "-", str(dashboard), checks, str(logger.metrics_file)],
        input=NODE_CHECK,
        text=True,
        encoding="utf-8",
        capture_output=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "offline KPI fixtures passed" in result.stdout