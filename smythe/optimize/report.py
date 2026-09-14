"""Pure HTML inspection of retained Autotune evidence; no experiment is rerun.

The collector owns ledger validation. This renderer checks its detached v1
payload and hash, preserves recorded values, and draws only saved comparisons.
"""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from html import escape
import json
import math
import re


_CSP = "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'"
_CSS = """
:root { color-scheme: light; color: #000000; background: #ffffff; }
* { box-sizing: border-box; }
body { margin: 0; font: 16px/1.55 'Avenir Next', 'Helvetica Neue', Arial, sans-serif; }
main { max-width: 1180px; margin: auto; padding: 48px 40px 72px; }
header { border-top: 3px solid #000000; padding-top: 22px; margin-bottom: 30px; }
nav { display: flex; flex-wrap: wrap; gap: 0 20px; margin-top: 12px; }
nav a { color: #000000; display: inline-flex; align-items: center; min-height: 44px;
  font-size: .875rem; text-underline-offset: .2em; }
nav a:hover { text-decoration-thickness: 2px; }
h1, h2, h3, h4 { font-family: Georgia, 'Times New Roman', serif; font-weight: 400;
  line-height: 1.2; overflow-wrap: anywhere; text-wrap: balance; }
h1 { margin: 0 0 12px; font-size: clamp(2rem, 4vw, 3.5rem); letter-spacing: -.025em; }
h2 { font-size: 1.85rem; margin: 0 0 12px; }
h3 { font-size: 1.4rem; margin: 30px 0 10px; }
h4 { font-size: 1.2rem; margin: 18px 0 8px; }
p { max-width: 72ch; margin: 0 0 12px; overflow-wrap: anywhere; }
section { margin-top: 42px; padding-top: 20px; border-top: 1px solid #000000; }
.overview { display: grid; grid-template-columns: 1fr 1fr; gap: 40px; }
.overview > *, .cell-value { min-width: 0; }
.note { font-size: .875rem; }
.state { font-weight: 700; }
.attention { text-decoration: underline; text-underline-offset: .2em; }
.table-region { max-width: 100%; overflow: auto; margin: 14px 0 20px; }
table { width: 100%; border-collapse: collapse; text-align: left; }
caption { text-align: left; font-weight: 600; padding-bottom: 8px; }
th, td { padding: 10px 12px 10px 0; border-bottom: 1px solid #000000;
  vertical-align: top; overflow-wrap: anywhere; }
th { font-weight: 600; } tbody th { font-weight: 400; }
thead th { border-bottom: 2px solid #000000; font-size: .8rem; }
th:last-child, td:last-child { padding-right: 0; }
.amount { display: block; text-align: right; white-space: nowrap; }
code, pre, .amount, .number, time { font-family: ui-monospace, 'SFMono-Regular', Consolas,
  'Liberation Mono', monospace; font-size: .83rem; overflow-wrap: anywhere; }
code { white-space: pre-wrap; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; max-height: 32rem; overflow: auto;
  line-height: 1.55; margin: 10px 0 18px; }
summary { cursor: pointer; min-height: 44px; padding: 10px 2px; font-size: .875rem; font-weight: 600; }
summary:hover { text-decoration: underline; text-underline-offset: .2em; }
nav a:focus-visible, summary:focus-visible, pre:focus-visible, .table-region:focus-visible {
  outline: 2px solid #000000; outline-offset: 3px; }
dl { margin: 0 0 18px; } dt { font-size: .8rem; font-weight: 600; margin-top: 12px; }
dd { margin: 3px 0 0; overflow-wrap: anywhere; }
figure { margin: 14px 0 24px; } figure svg { display: block; width: 100%; height: auto; }
figcaption { font-size: .8rem; max-width: 72ch; }
.plot-direction { display: flex; justify-content: space-between; gap: 16px;
  font-size: .8rem; margin: 0 5.7143% 8px; }
.metric { border-top: 1px solid #000000; margin-top: 24px; padding-top: 1px; }
ul { padding-left: 22px; } li { overflow-wrap: anywhere; }
footer { border-top: 3px solid #000000; margin-top: 40px; padding-top: 16px; font-size: .8rem; }
::selection { color: #ffffff; background: #000000; }
* { scrollbar-color: #000000 #ffffff; scrollbar-width: thin; }
@media (max-width: 700px) {
  main { padding: 24px 20px 48px; } .overview { grid-template-columns: 1fr; gap: 20px; }
  h2 { font-size: 1.55rem; } section { margin-top: 32px; }
  .records, .records tbody, .records tbody tr { display: block; width: 100%; }
  .records thead { position: absolute; width: 1px; height: 1px; overflow: hidden;
    clip-path: inset(50%); white-space: nowrap; }
  .records caption { display: block; }
  .records tbody tr { border-top: 1px solid #000000; padding: 10px 0; margin-top: 10px; }
  .records tbody th, .records tbody td { display: grid; grid-template-columns: 6.5rem minmax(0, 1fr);
    gap: 12px; padding: 7px 0; border: 0; }
  .records [data-label]::before { content: attr(data-label); font-size: .75rem; font-weight: 600; }
  .records .record-cell { display: block; } .records .amount { text-align: left; white-space: normal; }
  .records .record-cell::before { display: block; }
}
@media print {
  main { max-width: none; padding: 0; } pre, .table-region { max-height: none; overflow: visible; }
  h2, h3, h4, caption { break-after: avoid; } figure { break-inside: avoid; }
  th, td { font-size: 9pt; }
}
"""


def _json(value: object, *, pretty: bool = False) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      **({'indent': 2} if pretty else {'separators': (',', ':')}))


def _json_shape(value: object, depth: int = 0) -> None:
    if depth > 64:
        raise ValueError("Report JSON is nested too deeply")
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list:
        for item in value:
            _json_shape(item, depth + 1)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for item in value.values():
            _json_shape(item, depth + 1)
        return
    raise ValueError("Report must contain finite JSON values with string object keys")


def _text(value: object, missing: str = "Not recorded") -> str:
    text = missing if value is None else str(value)
    # Keep control characters visible instead of letting HTML parsing hide them.
    text = ''.join(f"\\u{ord(c):04x}" if (ord(c) < 32 and c not in '\n\t') or
                   127 <= ord(c) < 160 or ord(c) in (0x202a, 0x202b, 0x202c, 0x202d, 0x202e,
                                                   0x2066, 0x2067, 0x2068, 0x2069) else c
                   for c in text)
    return escape(text, quote=True)


def _code(value: object) -> str:
    return f'<code>{_text(value)}</code>'


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(value: object, name: str) -> int | float:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite number")
    try:
        valid = math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{name} must be a finite number")
    return value


def _number(value: object) -> str:
    return 'Not recorded' if value is None else f'<span class="number">{_text(_json(_finite(value, "Metric")))}</span>'


def _money(value: object) -> str:
    if value is None:
        return 'Not recorded'
    whole, fraction = divmod(_integer(value, 'Money'), 1_000_000)
    return f'<span class="amount">${whole:,}.{fraction:06d}</span>'


def _flag(value: object) -> str:
    if value is None:
        return 'Not recorded'
    if type(value) is not bool:
        raise ValueError('Recorded result flag must be boolean')
    return 'Passed' if value else 'Failed'


def _object(value: object, name: str) -> dict:
    if type(value) is not dict:
        raise ValueError(f'{name} must be an object')
    return value


def _rows(value: object, name: str) -> list[dict]:
    if type(value) is not list or any(type(row) is not dict for row in value):
        raise ValueError(f'{name} must be a list of objects')
    return value


def _time(value: object) -> str:
    if value is None:
        return 'Not recorded'
    seconds, nanos = divmod(_integer(value, 'Timestamp'), 1_000_000_000)
    try:
        stamp = datetime.fromtimestamp(seconds, timezone.utc)
    except (OverflowError, OSError, ValueError):
        return _code(f'{value} ns (outside calendar display range)')
    iso = stamp.strftime('%Y-%m-%dT%H:%M:%S') + f'.{nanos:09d}Z'
    return f'<time datetime="{iso}">{stamp:%Y-%m-%d %H:%M:%S} UTC</time>'


def _details(label: str, value: object) -> str:
    return (f'<details><summary>{_text(label)}</summary>'
            f'<pre tabindex="0">{_text(_json(value, pretty=True))}</pre></details>')


def _definitions(items: list[tuple[str, str]]) -> str:
    return '<dl>' + ''.join(f'<dt>{_text(label)}</dt><dd>{value}</dd>' for label, value in items) + '</dl>'


def _table(caption: str, headers: list[str], rows: list[list[str]], *, records: bool = True) -> str:
    head = ''.join(f'<th scope="col" role="columnheader">{_text(h)}</th>' for h in headers)
    body = []
    for row in rows:
        cells = []
        for i, value in enumerate(row):
            tag = 'th' if i == 0 else 'td'
            role = 'scope="row" role="rowheader"' if i == 0 else 'role="cell"'
            cls = ' class="record-cell"' if value.startswith('<details>') else ''
            cells.append(f'<{tag} {role} data-label="{_text(headers[i])}"{cls}>'
                         f'<div class="cell-value">{value}</div></{tag}>')
        body.append('<tr role="row">' + ''.join(cells) + '</tr>')
    return (f'<div class="table-region" tabindex="0" role="region" aria-label="{_text(caption)}">'
            f'<table class="{"records" if records else "balances"}" role="table">'
            f'<caption>{_text(caption)}</caption><thead role="rowgroup"><tr role="row">{head}</tr></thead>'
            f'<tbody role="rowgroup">{"".join(body)}</tbody></table></div>')


def _flags(value: object, name: str) -> str:
    mapping = _object(value, name)
    if not mapping:
        return '<p class="note">No individual flags recorded.</p>'
    return _table(name, ['Check', 'Recorded result'], [[_text(k), _flag(v)] for k, v in mapping.items()])


def _reasons(value: object) -> str:
    if type(value) is not list or any(type(v) is not str for v in value):
        raise ValueError('Recorded reasons must be a string list')
    return '<ul>' + ''.join(f'<li>{_text(v)}</li>' for v in value) + '</ul>' if value else ''


def _comparison(value: object) -> dict | None:
    """Recognize saved engine statistics without making up missing fields."""
    if value is None:
        return None
    item = _object(value, 'Comparison')
    for key in ('baseline_mean', 'candidate_mean', 'mean_improvement', 'confidence_level',
                'lower_confidence_bound'):
        if key in item:
            _finite(item[key], key)
    for key in ('sample_count', 'bootstrap_resamples'):
        if key in item:
            _integer(item[key], key, minimum=1)
    for key in ('hard_bounds_passed', 'non_regression_passed'):
        if key in item:
            _flag(item[key])
    if 'direction' in item and item['direction'] not in ('maximize', 'minimize'):
        raise ValueError('Comparison direction is invalid')
    if 'objective_name' in item and type(item['objective_name']) is not str:
        raise ValueError('Comparison objective name must be a string')
    if 'confidence_level' in item and not 0 < item['confidence_level'] < 1:
        raise ValueError('Comparison confidence must be between zero and one')
    if 'confidence_interval' in item:
        interval = item['confidence_interval']
        if type(interval) is not list or len(interval) != 2:
            raise ValueError('Comparison interval needs two finite endpoints')
        for endpoint in interval:
            _finite(endpoint, 'Interval endpoint')
        if interval[0] > interval[1]:
            raise ValueError('Comparison interval endpoints are reversed')
        if 'lower_confidence_bound' in item and item['lower_confidence_bound'] != interval[0]:
            raise ValueError('Saved lower bound disagrees with its interval')
    required = {'objective_name', 'direction', 'baseline_mean', 'candidate_mean',
                'mean_improvement', 'confidence_level', 'confidence_interval', 'sample_count'}
    return item if required <= item.keys() else None


def _interval_plot(item: dict, threshold: int | float | None, identity: str) -> str:
    low, high = item['confidence_interval']
    mean = item['mean_improvement']
    values = [0, low, high, mean] + ([] if threshold is None else [threshold])
    # Scaling first prevents max - min overflowing for opposite finite extremes.
    magnitude = max(abs(v) for v in values) or 1
    scaled = [v / magnitude for v in values]
    left, right = min(scaled), max(scaled)
    if left == right:
        left, right = -1, 1
    def x(value):
        coordinate = 32 + 496 * ((value / magnitude - left) / (right - left))
        if not math.isfinite(coordinate):
            raise ValueError('Non-finite plot coordinate')
        return f'{coordinate:.3f}'
    title = f"{item['objective_name']}: saved paired improvement interval"
    parts = [f'<figure><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 560 100" '
             f'role="img" aria-labelledby="{identity}"><title id="{identity}">{_text(title)}</title>',
             '<rect width="560" height="100" fill="#ffffff"/>',
             '<path d="M32 70H528" fill="none" stroke="#000000"/>',
             f'<path d="M{x(0)} 22V76" stroke="#000000" stroke-dasharray="2 3"/>',
             f'<path d="M{x(low)} 42H{x(high)}M{x(low)} 34V50M{x(high)} 34V50" '
             'fill="none" stroke="#000000" stroke-width="2"/>',
             f'<circle cx="{x(mean)}" cy="42" r="4" fill="#000000"/>']
    if threshold is not None:
        parts.append(f'<path d="M{x(threshold)} 16V64" fill="none" stroke="#000000" stroke-dasharray="6 3"/>')
    parts.extend(['</svg><div class="plot-direction"><span>Less improvement</span>'
                  '<span>More improvement</span></div>',
                  '<figcaption>Saved interval and mean (solid point); positive means improvement. '
                  'Dotted line: zero. '])
    if threshold is not None:
        parts.append(f'Dashed line: primary minimum {_number(threshold)}; the saved lower bound must exceed it. ')
    parts.append('Each metric has its own scale. Exact values appear in the table.</figcaption></figure>')
    return ''.join(parts)


def _development(assessment: dict) -> str:
    scores = []
    if 'development' in assessment and assessment['development'] is not None:
        scores = [_object(assessment['development'], 'Development score')]
    elif 'development_scores' in assessment:
        scores = _rows(assessment['development_scores'], 'Development scores')
    parts = ['<h3>Development</h3>']
    if not scores:
        return ''.join(parts) + '<p>Detailed development scores not recorded.</p>'
    parts.append('<p class="note">Only retained development scores are shown. No confidence interval or new ranking is calculated.</p>')
    for score in scores:
        parts.append(f'<h4>Candidate {_code(score.get("candidate_id"))}</h4>')
        means = _object(score.get('metric_means', {}), 'Development means')
        baseline = _object(score.get('incumbent_metric_means', {}), 'Development baseline')
        improvement = _object(score.get('mean_improvements', {}), 'Development improvements')
        names = dict.fromkeys([*means, *baseline, *improvement])
        if names:
            parts.append(_table('Saved development values · recorded metric units',
                                ['Metric', 'Incumbent mean', 'Candidate mean', 'Improvement'], [
                [_text(name), _number(baseline.get(name)), _number(means.get(name)), _number(improvement.get(name))]
                for name in names]))
        else:
            parts.append('<p>Detailed development values not recorded.</p>')
        parts.append(_definitions([('Development viable', _flag(score.get('viable'))),
                                   ('Primary improvement passed', _flag(score.get('primary_improvement_passed')))]))
        for key, label in [('gates', 'Development gates'), ('hard_bounds', 'Development hard bounds'),
                           ('secondary_non_regression', 'Development secondary non-regression')]:
            if key in score:
                parts.append(_flags(score[key], label))
        if 'reasons' in score:
            parts.append(_reasons(score['reasons']))
    return ''.join(parts)


def _assessment(value: object, stage: str, objectives: dict[str, dict]) -> str:
    parts = [f'<h3>{_text(stage)}</h3>']
    if value is None:
        return ''.join(parts) + '<p>No assessment recorded for this stage.</p>'
    assessment = _object(value, stage)
    parts.append(_definitions([(f'{stage} promotion passed', _flag(assessment.get('promote')))]))
    raw = [('Primary', assessment.get('primary'))]
    if 'secondary' in assessment:
        raw.extend(('Secondary', item) for item in _rows(assessment['secondary'], 'Secondary comparisons'))
    for index, (role, candidate) in enumerate(raw):
        item = _comparison(candidate)
        if item is None:
            parts.append(f'<p>{role} detailed comparison not recorded.</p>')
            continue
        objective = objectives.get(item['objective_name'], {})
        if objective and item['direction'] != objective.get('direction'):
            raise ValueError('Comparison direction differs from recorded objective')
        minimum = assessment.get('min_improvement') if role == 'Primary' else None
        if minimum is not None:
            _finite(minimum, 'Minimum improvement')
            if minimum < 0:
                raise ValueError('Minimum improvement must be nonnegative')
        parts.append(f'<div class="metric"><h4>{_text(item["objective_name"])} · {role.lower()}</h4>')
        direction = 'Higher' if item['direction'] == 'maximize' else 'Lower'
        parts.append(f'<p class="note">{direction} raw values are better. Units: recorded metric units. '
                     f'Paired samples: {item["sample_count"]:,}.</p>')
        parts.append(_table(f'{stage}: saved values', ['Measure', 'Recorded value'], [
            ['Incumbent mean', _number(item['baseline_mean'])], ['Candidate mean', _number(item['candidate_mean'])],
            ['Mean improvement', _number(item['mean_improvement'])],
            ['Confidence level', _number(item['confidence_level'])],
            ['Improvement interval', _number(item['confidence_interval'][0]) + ' to ' + _number(item['confidence_interval'][1])],
            ['Raw candidate hard minimum', _number(objective.get('hard_min'))],
            ['Raw candidate hard maximum', _number(objective.get('hard_max'))],
            ['Hard bounds passed', _flag(item.get('hard_bounds_passed'))],
        ]))
        if role == 'Secondary':
            parts.append(_definitions([('Allowed regression of mean improvement', _number(objective.get('max_regression'))),
                                       ('Mean non-regression passed', _flag(item.get('non_regression_passed')))]))
        parts.append(_interval_plot(item, minimum, f'{stage.lower()}-metric-{index}'))
        parts.append('</div>')
    for key, label in [('all_gates_passed', 'All gates passed'), ('hard_bounds_passed', 'All hard bounds passed'),
                       ('secondary_non_regression_passed', 'Secondary non-regression passed')]:
        if key in assessment:
            parts.append(_definitions([(label, _flag(assessment[key]))]))
    if 'gates' in assessment:
        parts.append(_flags(assessment['gates'], f'{stage} gates'))
    if 'reasons' in assessment:
        parts.append(_reasons(assessment['reasons']))
    return ''.join(parts)


def render_optimization_report(payload: dict) -> str:
    """Render the hash-bound v1 payload as inert standalone HTML.

    This is a view of saved evidence, not an independent statistical assessment.
    Integer balances are never converted to floating point for display.
    """
    _json_shape(payload)
    payload = _object(payload, 'Report payload')
    if type(payload.get('report_version')) is not int or payload['report_version'] != 1:
        raise ValueError('Unsupported optimization report version')
    digest = payload.get('evidence_sha256')
    if type(digest) is not str or not re.fullmatch('[0-9a-f]{64}', digest):
        raise ValueError('Report evidence SHA-256 is invalid')
    bound = {key: value for key, value in payload.items() if key != 'evidence_sha256'}
    if sha256(_json(bound).encode('utf-8')).hexdigest() != digest:
        raise ValueError('Report evidence SHA-256 does not match payload')
    snapshot = _object(payload.get('ledger_snapshot'), 'Ledger snapshot')
    campaign = _object(snapshot.get('campaign'), 'Campaign')
    if type(payload.get('campaign_id')) is not str or payload['campaign_id'] != campaign.get('campaign_id'):
        raise ValueError('Campaign identity does not match')
    contract = _object(campaign.get('contract'), 'Contract')
    objectives = {row['name']: row for row in _rows(contract.get('objectives', []), 'Objectives')}
    cost = _object(snapshot.get('cost'), 'Cost')
    counts = _object(snapshot.get('trial_counts'), 'Trial counts')
    for name, count in counts.items():
        _integer(count, f'{name} count')
    total = _integer(snapshot.get('trial_count'), 'Trial count')
    if sum(counts.values()) != total:
        raise ValueError('Trial counts do not reconcile')
    trials = _rows(payload.get('trials'), 'Trials')
    page = _object(payload.get('trial_detail'), 'Trial detail')
    limit = _integer(page.get('limit'), 'Detail limit', minimum=1)
    if limit > 1000 or page.get('total') != total or type(page.get('total')) is not int:
        raise ValueError('Trial detail bounds do not match')
    if type(page.get('returned')) is not int or page['returned'] != len(trials) or len(trials) > min(limit, total):
        raise ValueError('Trial detail count does not match')
    if type(page.get('truncated')) is not bool or page['truncated'] != (len(trials) < total):
        raise ValueError('Trial detail truncation does not match')
    candidates = _rows(snapshot.get('candidates'), 'Candidates')
    decisions = _rows(snapshot.get('decisions'), 'Decisions')
    if _integer(snapshot.get('candidate_count'), 'Candidate count') != len(candidates):
        raise ValueError('Candidate count does not match')
    order = campaign.get('ordered_candidate_ids')
    if (type(order) is not list or any(type(v) is not str for v in order)
            or len(set(order)) != len(order)):
        raise ValueError('Recorded candidate order is invalid')
    identifiers = [row.get('candidate_id') for row in candidates]
    if (any(type(v) is not str for v in identifiers) or len(set(identifiers)) != len(identifiers)
            or set(order) != set(identifiers)):
        raise ValueError('Recorded candidate order differs from inventory')
    by_id = {row['candidate_id']: row for row in candidates}
    candidates = [by_id[identifier] for identifier in order]
    if _integer(snapshot.get('decision_count'), 'Decision count') != len(decisions) or len(decisions) > 1:
        raise ValueError('Expected zero or one recorded decision')
    for name in ('confirmed_microusd', 'reserved_microusd', 'unknown_exposure_microusd', 'total_exposure_microusd'):
        _integer(cost.get(name), name)
    if cost['total_exposure_microusd'] != sum(cost[k] for k in ('confirmed_microusd', 'reserved_microusd', 'unknown_exposure_microusd')):
        raise ValueError('Cost balances do not reconcile')
    if type(snapshot.get('spent_microusd')) is not int or snapshot['spent_microusd'] != cost['confirmed_microusd']:
        raise ValueError('Confirmed cost disagrees with recorded spend')
    hashes = payload.get('evaluator_hashes')
    if type(hashes) is not list or any(type(v) is not str for v in hashes):
        raise ValueError('Evaluator fingerprints must be strings')
    decision = decisions[0] if decisions else None
    if decision is not None and type(decision.get('promoted')) is not bool:
        raise ValueError('Recorded decision must have a boolean promoted flag')
    state = 'No terminal decision' if decision is None else 'Promoted' if decision['promoted'] else 'Rejected'
    name = _text(contract.get('name'), 'Autotune campaign')
    parts = ['<!doctype html><!-- Existing operator-report direction: black/white editorial; '
             'first viewport shows recorded decision and exact balances; read-only mode; '
             'desktop/mobile/print finish review required. --><html lang="en"><head><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width, initial-scale=1">',
             f'<meta http-equiv="Content-Security-Policy" content="{escape(_CSP, quote=True)}">',
             f'<title>{name} — Smythe Autotune report</title><style>{_CSS}</style></head><body><main>',
             f'<header><h1>{name}</h1><p>Autotune report · <strong class="state">{state}</strong></p>',
             f'<p class="note">Campaign {_code(payload["campaign_id"])}</p>',
             '<nav aria-label="Report sections"><a href="#comparisons">Comparisons</a>'
             '<a href="#policies">Policies</a><a href="#trials">Trials</a>'
             '<a href="#provenance">Evidence</a></nav></header>',
             '<div class="overview"><div><h2>Recorded decision</h2>',
             f'<p>{_text(decision.get("reason")) if decision else "No terminal decision is recorded. Retained trials may be incomplete."}</p>',
             f'<p>{total:,} trials · {counts.get("completed", 0):,} completed · '
             f'<strong>{counts.get("unknown", 0):,} unknown</strong> · '
             f'{counts.get("prepared", 0) + counts.get("dispatched", 0):,} prepared or dispatched.</p>',
             '<p class="note">Statuses describe retained records, not whether a process is currently running. '
             'Unknown trial outcomes remain unresolved even when their ceiling is zero.</p>',
             '</div><div><h2>Cost balances</h2>',
             _table('Whole-campaign accounting · USD', ['Account', 'Amount'], [
                 ['Recorded budget limit', _money(contract.get('max_budget_microusd'))],
                 ['Confirmed completed-trial cost', _money(cost['confirmed_microusd'])],
                 ['Held prepared/dispatched ceilings', _money(cost['reserved_microusd'])],
                 ['Unknown exposure', _money(cost['unknown_exposure_microusd'])],
                 ['Total exposure', _money(cost['total_exposure_microusd'])]], records=False),
             '<p class="note">Total exposure includes confirmed cost and held ceilings. No missing cost is estimated.</p></div></div>',
             '<section aria-labelledby="comparisons"><h2 id="comparisons">Saved comparisons</h2>',
             '<p>These are retained evaluator observations and decision results. Statistics are not rerun, '
             'and the report does not establish live-provider performance or statistical calibration.</p>']
    if decision is None:
        parts.append('<p>No decision assessment is available.</p>')
    else:
        assessment = _object(decision.get('assessment', {}), 'Decision assessment')
        parts.append(_development(assessment))
        parts.append(_assessment(assessment.get('confirmation'), 'Confirmation', objectives))
        parts.append(_assessment(assessment.get('holdout'), 'Holdout', objectives))
    parts.append('</section><section aria-labelledby="policies"><h2 id="policies">Candidate policies</h2>'
                 '<p>Recorded inventory order is preserved. This is not a new performance ranking.</p>')
    for row in candidates:
        candidate = _object(row.get('candidate'), 'Candidate')
        identifier = row.get('candidate_id')
        roles = []
        if identifier == campaign.get('incumbent_candidate_id'):
            roles.append('Incumbent')
        if decision and identifier == decision.get('candidate_id'):
            roles.append('Decision candidate')
        parts.append(f'<h3>{_code(identifier)}</h3><p class="state">{_text(" · ".join(roles) or "Recorded candidate")}</p>')
        parts.append(f'<p>{_text(candidate.get("hypothesis"))}</p>')
        parts.append(_definitions([('Parent', _code(candidate.get('parent'))),
                                   ('Policy hash', _code(row.get('policy_hash')))]))
        parts.append(_details('Inspect policy and candidate', row))
    parts.append('</section><section aria-labelledby="trials"><h2 id="trials">Trial detail</h2>')
    parts.append(f'<p>Showing {len(trials):,} of {total:,} trials (detail limit {limit:,}). '
                 'Whole-campaign balances and decision evidence above remain complete.</p>')
    if page['truncated']:
        parts.append('<p class="attention">Trial detail is truncated; additional trial rows are not included in this report.</p>')
    trial_rows = []
    for trial in trials:
        duration = trial.get('duration_ms')
        if duration is not None and _finite(duration, 'Trial duration') < 0:
            raise ValueError('Trial duration must be nonnegative')
        trial_rows.append([_code(trial.get('trial_key')), _code(trial.get('candidate_id')),
                           _text(trial.get('split')) + '<br>' + _text(trial.get('status')),
                           _money(trial.get('ceiling_microusd')), _money(trial.get('actual_cost_microusd')),
                           _number(duration), _details('Inspect trial', trial)])
    parts.append(_table('Retained trial records', ['Trial', 'Candidate', 'Split / state', 'Ceiling USD',
                                                  'Confirmed USD', 'Duration ms', 'Record'], trial_rows)
                 if trial_rows else '<p>No trial rows recorded.</p>')
    parts.append('</section><section aria-labelledby="provenance"><h2 id="provenance">Evidence and provenance</h2>')
    parts.append(_definitions([('Report version', _code(payload['report_version'])),
                               ('Evidence SHA-256', _code(digest)),
                               ('Contract hash', _code(campaign.get('contract_hash'))),
                               ('Plan hash', _code(campaign.get('plan_hash'))),
                               ('Holdout seed commitment', _code(campaign.get('holdout_seed_commitment'))),
                               ('Created', _time(campaign.get('created_at_ns')))]))
    parts.append('<p class="note">The evidence hash identifies canonical report JSON, not database bytes. '
                 'Evaluator fingerprints identify retained evaluators; they do not attest a simulation or provider type.</p>')
    parts.append(_details('Evaluator fingerprints', hashes))
    parts.append(_details('Complete recorded decision evidence', decisions))
    parts.append(_details('Recorded campaign and contract', campaign))
    parts.append('</section><footer>Smythe Autotune · Static, read-only report. '
                 'Artifact hashes and paths are text; no artifact is loaded.</footer></main></body></html>')
    result = ''.join(parts)
    if len(result.encode('utf-8')) > 32 * 1024 * 1024:
        raise ValueError('Report exceeds the 32 MiB publication limit')
    return result
