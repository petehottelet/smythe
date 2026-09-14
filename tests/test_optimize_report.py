"""Pure renderer checks using synthetic, explicitly saved assessment fixtures."""

from copy import deepcopy
from hashlib import sha256
from html import unescape
from html.parser import HTMLParser
import json
import math
import re
import xml.etree.ElementTree as ET

import pytest

from smythe.optimize.report import render_optimization_report


def bind(payload):
    value = {key: item for key, item in payload.items() if key != 'evidence_sha256'}
    payload['evidence_sha256'] = sha256(json.dumps(
        value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False,
    ).encode()).hexdigest()
    return payload


def comparison(name='quality', direction='maximize'):
    return {'objective_name': name, 'direction': direction, 'sample_count': 3,
            'sample_seeds': [1, 2, 3], 'baseline_mean': 11.123456789012345,
            'candidate_mean': 13.25, 'mean_improvement': 2.126543210987655,
            'confidence_level': .95, 'bootstrap_resamples': 100,
            'bootstrap_seed': 42, 'confidence_interval': [1.75, 2.5],
            'lower_confidence_bound': 1.75, 'hard_bounds_passed': True,
            'non_regression_passed': True}


def assessment():
    return {'promote': True, 'primary': comparison(),
            'secondary': [comparison('latency', 'minimize')],
            'gates': {'format': True}, 'all_gates_passed': True,
            'hard_bounds_passed': True, 'secondary_non_regression_passed': True,
            'min_improvement': .125, 'reasons': []}


def development():
    return {'candidate_id': 'candidate', 'policy_hash': 'sha256:' + 'b' * 64,
            'primary_objective': 'quality', 'primary_direction': 'maximize',
            'primary_mean': 12.75, 'metric_means': {'quality': 12.75},
            'incumbent_metric_means': {'quality': 10.125},
            'mean_improvements': {'quality': 2.625}, 'gates': {'format': True},
            'hard_bounds': {'quality': True}, 'primary_improvement_passed': True,
            'secondary_non_regression': {}, 'viable': True, 'reasons': [], 'trial_keys': ['t1']}


def payload():
    candidates = [{'candidate_id': name, 'policy_hash': 'sha256:' + c * 64,
                   'candidate': {'policy': {'concurrency': i + 1}, 'hypothesis': 'Synthetic fixture policy',
                                 'parent': None if not i else 'incumbent'}, 'created_at_ns': 0}
                  for i, (name, c) in enumerate([('incumbent', 'a'), ('candidate', 'b')])]
    decision = {'decision_id': 'decision-fixture', 'campaign_id': 'fixture', 'candidate_id': 'candidate',
                'promoted': True, 'reason': 'Recorded acceptance reason', 'trial_keys': ['t1'],
                'assessment': {'development': development(), 'confirmation': assessment(),
                               'holdout': assessment()}, 'created_at_ns': 0}
    trial = {'trial_key': 't1', 'candidate_id': 'candidate', 'split': 'development',
             'phase': 'challenger.plan_fixture', 'status': 'completed', 'metrics': {'quality': 12.75},
             'gates': {'format': True}, 'ceiling_microusd': 1_000_000, 'actual_cost_microusd': 123_456,
             'duration_ms': 2.125, 'artifact_hashes': [], 'error': None,
             'prepared_at_ns': 0, 'dispatched_at_ns': 1, 'terminal_at_ns': 2}
    snapshot = {'campaign': {'campaign_id': 'fixture', 'contract_hash': 'sha256:' + 'c' * 64,
                            'contract': {'name': 'Synthetic report fixture', 'max_budget_microusd': 5_000_000,
                                         'objectives': [{'name': 'quality', 'direction': 'maximize', 'primary': True,
                                                         'hard_min': 1.25, 'hard_max': None, 'max_regression': None},
                                                        {'name': 'latency', 'direction': 'minimize', 'primary': False,
                                                         'hard_min': None, 'hard_max': 900.5, 'max_regression': 3.75}]},
                            'ordered_candidate_ids': ['incumbent', 'candidate'],
                            'incumbent_candidate_id': 'incumbent', 'plan_hash': 'sha256:' + 'd' * 64,
                            'holdout_seed_commitment': 'sha256:' + 'e' * 64, 'created_at_ns': 0},
                'spent_microusd': 123_456, 'cost': {'confirmed_microusd': 123_456,
                                                 'reserved_microusd': 0, 'unknown_exposure_microusd': 0,
                                                 'total_exposure_microusd': 123_456},
                'trial_count': 1, 'trial_counts': {'completed': 1}, 'candidate_count': 2,
                'candidates': candidates, 'decision_count': 1, 'decisions': [decision]}
    return bind({'report_version': 1, 'campaign_id': 'fixture', 'evaluator_hashes': ['sha256:' + 'f' * 64],
                 'ledger_snapshot': snapshot, 'trials': [trial],
                 'trial_detail': {'limit': 500, 'total': 1, 'returned': 1, 'truncated': False}})


def visible(html):
    return unescape(re.sub('<[^>]+>', '', html))


def test_saved_values_roles_thresholds_and_decision_are_preserved():
    data = payload()
    output = render_optimization_report(data)
    text = visible(output)
    for value in (str(comparison()['baseline_mean']), '13.25', '2.126543210987655', '1.75', '2.5', '12.75', '10.125', '2.625'):
        assert value in text
    assert 'Paired samples: 3' in text
    assert 'recorded metric units' in text
    assert output.count('<svg ') == 4
    assert output.count('Dashed line: primary minimum') == 2
    assert output.count('Allowed regression of mean improvement') == 2
    assert '900.5' in text and '3.75' in text
    assert 'Recorded acceptance reason' in text and 'Promoted' in text
    assert text.index('Recorded decision') < text.index('Saved comparisons')
    assert text.index('Cost balances') < text.index('Saved comparisons')
    assert 'Incumbent' in text and 'Decision candidate' in text


@pytest.mark.parametrize('stage', ['development', 'confirmation', 'holdout'])
def test_rejection_uses_recorded_decision_and_available_stage(stage):
    data = payload()
    record = data['ledger_snapshot']['decisions'][0]
    record.update(promoted=False, reason=f'Recorded {stage} rejection')
    saved = record['assessment']
    if stage == 'development':
        score = development()
        score.update(viable=False, reasons=['Development gate failed'])
        record['assessment'] = {'stage': 'development', 'development_scores': [score]}
    else:
        saved[stage].update(promote=False, reasons=['Recorded rejection detail'])
        if stage == 'confirmation':
            saved['holdout'] = None
    output = render_optimization_report(bind(data))
    text = visible(output)
    assert f'Recorded {stage} rejection' in text
    assert 'Autotune report · Rejected' in text
    assert output.count('<svg ') == {'development': 0, 'confirmation': 2, 'holdout': 4}[stage]


def test_no_decision_unknown_zero_ceiling_and_reserved_are_not_success_or_live():
    data = payload()
    snapshot = data['ledger_snapshot']
    snapshot.update(decisions=[], decision_count=0, trial_count=2, trial_counts={'unknown': 1, 'dispatched': 1},
                    spent_microusd=0)
    snapshot['cost'] = {'confirmed_microusd': 0, 'reserved_microusd': 17,
                        'unknown_exposure_microusd': 0, 'total_exposure_microusd': 17}
    data['trials'][0].update(status='unknown', ceiling_microusd=0, actual_cost_microusd=None,
                             duration_ms=None, metrics={}, gates={}, error='Outcome unknown')
    data['trial_detail'].update(total=2, truncated=True)
    output = render_optimization_report(bind(data))
    text = visible(output)
    assert 'No terminal decision' in text and '1 unknown' in text
    assert 'Not recorded' in text and '$0.000017' in text
    assert 'Trial detail is truncated' in text
    assert '0 confirmed' not in text and '<svg ' not in output
    assert 'whether a process is currently running' in text


@pytest.mark.parametrize('stats', [{}, {'custom': {'notes': 'Saved custom interpretation'}},
                                   {'stage': 'development', 'development_scores': []},
                                   {'confirmation': {'promote': False, 'primary': {'baseline_mean': 4}}}])
def test_absent_custom_or_partial_statistics_remain_unplotted(stats):
    data = payload()
    data['ledger_snapshot']['decisions'][0]['assessment'] = stats
    output = render_optimization_report(bind(data))
    assert '<svg ' not in output
    assert 'Complete recorded decision evidence' in output
    assert 'not recorded' in output or 'No assessment' in output
    assert 'Autotune report · <strong class="state">Promoted' in output


@pytest.mark.parametrize('field,value', [
    ('mean_improvement', True), ('baseline_mean', '10'), ('candidate_mean', None),
    ('confidence_interval', [1]), ('confidence_interval', [True, 2]),
    ('confidence_interval', [2, 1]), ('sample_count', False), ('sample_count', 3.0),
    ('sample_count', 0), ('confidence_level', 1), ('confidence_level', 0),
    ('lower_confidence_bound', -100), ('hard_bounds_passed', 'true'),
    ('direction', 'up'), ('objective_name', []),
])
def test_malformed_present_comparison_is_refused(field, value):
    data = payload()
    data['ledger_snapshot']['decisions'][0]['assessment']['confirmation']['primary'][field] = value
    with pytest.raises(ValueError):
        render_optimization_report(bind(data))


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_anywhere_is_rejected_even_outside_plot(value):
    data = payload()
    data['ledger_snapshot']['decisions'][0]['assessment']['custom'] = value
    with pytest.raises(ValueError, match='finite JSON'):
        render_optimization_report(data)


@pytest.mark.parametrize('low,mean,high,minimum', [(-1e308, 0.0, 1e308, 1e308),
                                                 (0.0, 0.0, 0.0, 0.0),
                                                 (-5e-324, 0.0, 5e-324, 0.0),
                                                 (1, 4, 2, .1)])
def test_svg_coordinates_remain_finite_for_extreme_and_degenerate_values(low, mean, high, minimum):
    data = payload()
    saved = data['ledger_snapshot']['decisions'][0]['assessment']['confirmation']
    saved['primary'].update(confidence_interval=[low, high], lower_confidence_bound=low,
                            mean_improvement=mean)
    saved['min_improvement'] = minimum
    output = render_optimization_report(bind(data))
    svgs = re.findall(r'<svg .*?</svg>', output)
    assert len(svgs) == 4
    for svg in svgs:
        tree = ET.fromstring(svg)
        for elem in tree.iter():
            for key, value in elem.attrib.items():
                if key in ('cx', 'cy', 'x', 'y', 'r', 'width', 'height'):
                    assert math.isfinite(float(value))
                elif key == 'd':
                    numbers = [float(n) for n in re.findall(r'-?\d+(?:\.\d+)?', value)]
                    assert numbers and all(math.isfinite(n) for n in numbers)
                    assert all(0 <= n <= 560 for n in numbers)
    assert str(low) in visible(output) and str(high) in visible(output)


@pytest.mark.parametrize('amount', [0, 1, 1_234_567, (1 << 63) - 1, 3 * ((1 << 63) - 1)])
def test_money_keeps_large_integers_and_exact_six_decimals(amount):
    data = payload()
    data['ledger_snapshot']['cost'].update(confirmed_microusd=amount, total_exposure_microusd=amount)
    data['ledger_snapshot']['spent_microusd'] = amount
    data['trials'][0]['actual_cost_microusd'] = amount
    expected = f'${amount // 1_000_000:,}.{amount % 1_000_000:06d}'
    assert expected in render_optimization_report(bind(data))


@pytest.mark.parametrize('value', [True, -1, '1', .1])
def test_malformed_integer_money_is_refused(value):
    data = payload()
    data['trials'][0]['actual_cost_microusd'] = value
    with pytest.raises(ValueError, match='Money'):
        render_optimization_report(bind(data))


class Markup(HTMLParser):
    def __init__(self):
        super().__init__()
        self.elements = []

    def handle_starttag(self, tag, attrs):
        self.elements.append((tag, dict(attrs)))


def test_untrusted_markup_controls_paths_are_inert_and_csp_is_static():
    data = payload()
    attack = '</title><script>alert(1)</script><img src="https://evil.test/a" onerror="x">'
    data['ledger_snapshot']['campaign']['contract']['name'] = attack
    data['ledger_snapshot']['candidates'][0]['candidate']['hypothesis'] = attack + '\x00\u202e'
    data['ledger_snapshot']['decisions'][0]['reason'] = attack
    data['trials'][0]['artifact_hashes'] = [attack, 'file:///local/private.png']
    output = render_optimization_report(bind(data))
    parser = Markup()
    parser.feed(output)
    assert not any(tag in {'script', 'img', 'iframe', 'link', 'object', 'embed', 'form'} for tag, _ in parser.elements)
    assert [attrs.get('href') for tag, attrs in parser.elements if tag == 'a'] == [
        '#comparisons', '#policies', '#trials', '#provenance']
    assert not any(key.lower().startswith('on') or key == 'src'
                   for _, attrs in parser.elements for key in attrs)
    assert '&lt;script&gt;' in output and '\x00' not in output and '\u202e' not in output
    assert '\\u0000' in output and '\\u202e' in output
    csp = next(attrs['content'] for tag, attrs in parser.elements
               if tag == 'meta' and attrs.get('http-equiv') == 'Content-Security-Policy')
    assert csp == "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'"
    assert set(re.findall(r'#[0-9a-fA-F]{6}', output)) <= {'#000000', '#ffffff'}


def test_hash_binding_determinism_no_input_mutation_or_external_work(monkeypatch):
    import builtins
    import socket
    import sqlite3
    from smythe.optimize import statistics
    data = payload()
    original = deepcopy(data)
    def forbidden(*args, **kwargs):
        raise AssertionError('Pure renderer accessed an external subsystem')
    monkeypatch.setattr(builtins, 'open', forbidden)
    monkeypatch.setattr(socket, 'socket', forbidden)
    monkeypatch.setattr(sqlite3, 'connect', forbidden)
    monkeypatch.setattr(statistics, 'assess_promotion', forbidden)
    monkeypatch.setattr(statistics, 'bootstrap_confidence_interval', forbidden)
    first = render_optimization_report(data)
    assert render_optimization_report(data) == first
    assert data == original
    data['ledger_snapshot']['decisions'][0]['reason'] = 'Changed retained data'
    with pytest.raises(ValueError, match='does not match'):
        render_optimization_report(data)


@pytest.mark.parametrize('change', ['version', 'count', 'truncated', 'returned', 'balance', 'decision', 'campaign'])
def test_inconsistent_payload_is_refused(change):
    data = payload()
    if change == 'version':
        data['report_version'] = True
    elif change == 'count':
        data['ledger_snapshot']['trial_counts']['completed'] = True
    elif change == 'truncated':
        data['trial_detail']['truncated'] = 0
    elif change == 'returned':
        data['trial_detail']['returned'] = False
    elif change == 'balance':
        data['ledger_snapshot']['cost']['total_exposure_microusd'] += 1
    elif change == 'decision':
        data['ledger_snapshot']['decisions'][0]['promoted'] = 1
    else:
        data['campaign_id'] = 'other'
    with pytest.raises(ValueError):
        render_optimization_report(bind(data))


def test_saved_candidate_order_wins_over_snapshot_id_sort_without_mutation():
    data = payload()
    data['ledger_snapshot']['candidates'].reverse()
    assert data['ledger_snapshot']['candidates'][0]['candidate_id'] == 'candidate'
    data = bind(data)
    original = deepcopy(data)
    output = render_optimization_report(data)
    policies = output.split('<h2 id="policies">', 1)[1].split('<h2 id="trials">', 1)[0]
    assert policies.index('<h3><code>incumbent</code>') < policies.index('<h3><code>candidate</code>')
    assert data == original


@pytest.mark.parametrize('order', [['incumbent'], ['incumbent', 'incumbent'], ['other', 'candidate'], [True, 'candidate']])
def test_invalid_saved_candidate_order_is_refused(order):
    data = payload()
    data['ledger_snapshot']['campaign']['ordered_candidate_ids'] = order
    with pytest.raises(ValueError, match='candidate order'):
        render_optimization_report(bind(data))


def test_trial_detail_cannot_exceed_total_even_with_consistent_returned_flag():
    data = payload()
    data['trials'].append(deepcopy(data['trials'][0]))
    data['trial_detail']['returned'] = 2
    with pytest.raises(ValueError, match='detail count'):
        render_optimization_report(bind(data))


def test_section_navigation_targets_unique_headings_and_remains_in_incomplete_report():
    data = payload()
    data['ledger_snapshot'].update(decisions=[], decision_count=0)
    output = render_optimization_report(bind(data))
    parser = Markup()
    parser.feed(output)
    links = [attrs['href'][1:] for tag, attrs in parser.elements if tag == 'a']
    headings = [attrs['id'] for tag, attrs in parser.elements if tag == 'h2' and 'id' in attrs]
    assert links == headings == ['comparisons', 'policies', 'trials', 'provenance']
    assert len(set(links)) == len(links)
    assert '<nav aria-label="Report sections">' in output


def test_direction_labels_are_html_text_outside_scaled_svg_for_each_saved_interval():
    output = render_optimization_report(payload())
    svgs = re.findall(r'<svg .*?</svg>', output)
    assert len(svgs) == 4
    assert all('Less improvement' not in svg and 'More improvement' not in svg for svg in svgs)
    assert output.count('<div class="plot-direction"><span>Less improvement</span>') == len(svgs)
    assert output.count('<span>More improvement</span></div>') == len(svgs)
