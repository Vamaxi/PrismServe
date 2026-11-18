"""Compiled resource accounting, phase-aware feedback and conservative changes.

The resource formulas match the v0.3 measured controller. Cached reservations
never cache feasibility: current device/host limits are checked on every call.
Uncertainty margins are engineering heuristics, not statistical confidence bounds.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, replace
import math
import time

from .measured_policy import MeasuredController
from .types import ControlState, Decision, Estimate, Plan


class EfficientController(MeasuredController):
    def __init__(self, *args, conservative_gate=True, minimum_gain=.10, uncertainty_floor=.10, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0 <= minimum_gain < 1 or not 0 <= uncertainty_floor < 1:
            raise ValueError("gain and uncertainty margins must be within[0,1)")
        self.conservative_gate = conservative_gate
        self.minimum_gain, self.uncertainty_floor = minimum_gain, uncertainty_floor
        self._families = {}
        self._sample_dicts = {}
        self._bounds = {}
        self._qstats = {}
        self._lookup_cache = OrderedDict()
        self._reservation_cache = OrderedDict()
        self._phase_feedback = {}
        self.lookup_hits = self.reservation_hits = self.searches = self.resource_checks = 0
        m = self.model
        self._kv = m.kv_bytes_per_token
        self._backups = m.fp_weight_bytes + sum(m.int4_layer_bytes_for(i) for i in range(m.layers))
        for row in self.calibration.samples:
            key = (row.quantized_layers, bool(row.cpu_requests))
            shape = (row.batch_size, row.prefill_tokens, row.context_tokens, row.cpu_requests)
            self._families.setdefault(key, []).append((row, tuple(math.log(v + 1) for v in shape)))
            self._sample_dicts[row] = asdict(row)
        for q in set(self.quantization_candidates) | {r.quantized_layers for r in self.calibration.samples}:
            family = [row for row in self.calibration.samples if row.quantized_layers == q]
            residuals = [(r.batch_size, max(0, r.workspace_bytes - (r.batch_size - r.cpu_requests) * r.context_tokens * self._kv)) for r in family]
            smallest = min((b for b, _ in residuals), default=1)
            largest = max((b for b, _ in residuals), default=1)
            fixed = max((v for b, v in residuals if b == smallest), default=0)
            growth = max((max(0, v - fixed) / (b - smallest) for b, v in residuals if b > smallest), default=0)
            self._bounds[q] = (smallest, largest, fixed, growth)

    @staticmethod
    def _bounded_put(cache, key, value, size=1024):
        cache[key] = value
        cache.move_to_end(key)
        if len(cache) > size:
            cache.popitem(last=False)

    @staticmethod
    def _features(items, cpu_count):
        features = MeasuredController._features(items, cpu_count)
        prefill = sum(bool(item.prefill_tokens) for item in items)
        features['phase'] = 'prefill' if prefill == len(items) and items else ('mixed' if prefill else 'decode')
        return features

    def _lookup(self, layers, features):
        shape = tuple(features[name] for name in ('batch_size', 'prefill_tokens', 'context_tokens', 'cpu_requests'))
        key = (layers, shape)
        if key in self._lookup_cache:
            self.lookup_hits += 1
            return self._lookup_cache[key]
        family = self._families.get((layers, bool(features['cpu_requests'])), ())
        if not family:
            return None, 0., 0.
        logs = tuple(math.log(v + 1) for v in shape)
        def distance(pair):
            row, vector = pair
            return (sum(abs(a - b) for a, b in zip(vector, logs)), row.batch_size,
                    row.context_tokens, row.prefill_tokens, row.cpu_requests)
        row = min(family, key=distance)[0]
        work = max(features['batch_size'], features['prefill_tokens'])
        scale = max(1., features['batch_size'] / row.batch_size,
                    work / max(row.batch_size, row.prefill_tokens),
                    features['batch_size'] * max(1, features['context_tokens']) / (row.batch_size * max(1, row.context_tokens)),
                    features['cpu_requests'] / max(1, row.cpu_requests))
        result = row, row.elapsed_ms * scale, scale
        self._bounded_put(self._lookup_cache, key, result)
        return result

    def _weights(self, q):
        if q not in self._qstats:
            if len(set(q)) != len(q) or any(not 0 <= i < self.model.layers for i in q):
                raise ValueError('invalid quantized layer indices')
            self._qstats[q] = (max(0, self.model.fp_weight_bytes - sum(self.model.layer_saving_bytes_for(i) for i in q)),
                               sum(self.model.layer_sensitivities[i] for i in q))
        return self._qstats[q]

    def _resource(self, items, plan, state):
        self.resource_checks += 1
        m, kv = self.model, self._kv
        q, old_q = plan.quantized_layers, tuple(sorted(set(state.quantized_layers)))
        cpu, old_cpu = set(plan.cpu_request_ids), set(state.cpu_request_ids)
        ids = {item.request_id for item in items}
        if len(ids) != len(items) or not cpu <= ids or len(cpu) != len(plan.cpu_request_ids):
            raise ValueError('invalid or duplicate request IDs in resource plan')
        if cpu and (self.profile.device_kind == 'cpu' or any(item.prefill_tokens and item.request_id in cpu for item in items)):
            raise ValueError('cannot offload CPU execution or prefilling attention')
        weights, quality = self._weights(q)
        gpu_kv = sum(item.context_tokens * kv for item in items if item.request_id not in cpu)
        host_kv = sum(item.context_tokens * kv for item in items if item.request_id in cpu)
        compute_tokens = sum(item.prefill_tokens or 1 for item in items)
        base_workspace = max(1 << 20, 8 * m.hidden_size * max(1, compute_tokens))
        smallest, largest, fixed, growth = self._bounds.get(q, (1, 1, 0, 0))
        scratch = fixed + growth * max(0, len(items) - smallest) + fixed * max(0, len(items) - largest) / smallest
        workspace = max(base_workspace, math.ceil((scratch + gpu_kv) * self.workspace_safety_factor))
        stable = q == old_q and cpu == old_cpu
        cache_key = (q, gpu_kv, host_kv, len(items), base_workspace)
        if stable and cache_key in self._reservation_cache:
            self.reservation_hits += 1
            device_peak, host_peak = self._reservation_cache[cache_key]
            staging = migration = 0
        else:
            old_gpu = old_host = migration = 0
            for item in items:
                existing = item.existing_context_tokens
                if existing is None:
                    existing = 0 if item.prefill_tokens else item.context_tokens
                size = existing * kv
                if item.request_id in old_cpu:
                    old_host += size
                else:
                    old_gpu += size
                if (item.request_id in cpu) != (item.request_id in old_cpu):
                    migration += size
            changed = set(q) ^ set(old_q)
            staging = max((m.int4_layer_bytes_for(i) if i in q else m.fp_layer_bytes_for(i) for i in changed), default=0)
            device_peak = max(weights + gpu_kv + workspace,
                              self._weights(old_q)[0] + old_gpu + workspace + staging)
            host_peak = self._backups + max(host_kv, old_host) + migration + host_kv
            if stable:
                self._bounded_put(self._reservation_cache, cache_key, (device_peak, host_peak))
        p, cfg = self.profile, self.config
        memory_ok = ((device_peak + host_peak <= int(p.host_memory_bytes * cfg.memory_fraction)) if p.device_kind == 'cpu'
                     else (device_peak <= int(p.device_memory_bytes * cfg.memory_fraction)
                           and host_peak <= int(p.host_memory_bytes * cfg.memory_fraction)))
        to_q = len(set(q) - set(old_q)) * self.calibration.to_int4_layer_ms
        to_fp = len(set(old_q) - set(q)) * self.calibration.to_fp_layer_ms
        transport = p.host_bandwidth_gbps if p.device_kind == 'cpu' else p.link_bandwidth_gbps
        transition = to_q + to_fp + migration / (transport * 1e6)
        quality_ok = quality <= cfg.quality_budget + 1e-12
        return device_peak, host_peak, transition, quality, memory_ok, quality_ok, {
            'memory_feasible': memory_ok, 'quality_proxy_feasible': quality_ok,
            'weight_bytes': weights, 'gpu_kv_bytes': gpu_kv, 'cpu_kv_bytes': host_kv,
            'host_backup_bytes': self._backups, 'workspace_bytes': workspace,
            'reserved_device_kv_copy_bytes': gpu_kv, 'reserved_host_kv_copy_bytes': host_kv,
            'transition_staging_bytes': staging, 'migration_bytes': migration,
            'dequantization_scratch_bytes': 0,
            'workspace_source': 'compiled_monotone_reserved_kv_bound',
        }

    def _estimate(self, items, plan, state):
        resource = self._resource(items, plan, state)
        device, host, transition, risk, memory_ok, quality_ok, components = resource
        features = self._features(items, len(plan.cpu_request_ids))
        if memory_ok and quality_ok:
            sample, baseline, scale = self._lookup(plan.quantized_layers, features)
        else:
            sample, baseline, scale = None, 0., 1.
        key = (plan.quantized_layers, bool(plan.cpu_request_ids), features['phase'])
        stats = self._phase_feedback.get(key, (1., 0., 0))
        predicted = baseline * stats[0] if items else 0.
        known = bool(self._families.get((plan.quantized_layers, bool(plan.cpu_request_ids)))) or not items
        relative_margin = min(.35, self.uncertainty_floor + .03 * max(0, scale - 1))
        uncertainty = predicted * relative_margin + stats[1]
        reason = 'compiled reservations and phase-aware measured timing'
        if not memory_ok:
            reason = 'memory capacity or transition peak exceeded'
        elif not quality_ok:
            reason = 'quality proxy budget exceeded'
        elif not known:
            reason = 'missing measured timing family'
        return Estimate(device, host, predicted, transition, risk, memory_ok and quality_ok and known, reason,
                        {**components, 'model': 'compiled_phase_aware_measured_controller',
                         'timing_source': self.calibration.source, 'timing_features': features,
                         'timing_family_available': known, 'lookup_base_ms': baseline,
                         'timing_shape_scale': scale, 'ewma_correction': stats[0],
                         'feedback_samples': stats[2], 'uncertainty_ms': uncertainty,
                         'uncertainty_note': 'heuristic margin, not a statistical confidence interval',
                         'nearest_sample': None if sample is None else dict(self._sample_dicts[sample]),
                         'latency_proxy_feasible': predicted + transition <= self.config.tbt_slo_ms})

    def evaluate_plan(self, items, plan, state, reason='reuse stable plan with fresh resource check'):
        start = time.perf_counter_ns()
        estimate = self._estimate(items, plan, state)
        return Decision(plan, estimate, self.policy, (time.perf_counter_ns() - start) / 1000, reason)

    def supports_change(self, alternative, alternative_batch, reference, reference_batch):
        if not self.conservative_gate or not reference.estimate.feasible:
            return True
        a, r = alternative.estimate, reference.estimate
        horizon = self.config.transition_horizon
        upper = (a.iteration_ms + a.components['uncertainty_ms'] + a.transition_ms / horizon) / alternative_batch
        lower = max(0., r.iteration_ms - r.components['uncertainty_ms'] + r.transition_ms / horizon) / reference_batch
        return upper < lower * (1 - self.minimum_gain)

    def choose(self, items, state=None, now_s=0.):
        start = time.perf_counter_ns()
        self.searches += 1
        state = state or ControlState()
        decision = super().choose(items, state, now_s)
        adapted = bool(decision.plan.quantized_layers or decision.plan.cpu_request_ids)
        if items and adapted and decision.estimate.feasible and self.policy != 'naive' and self.conservative_gate:
            reference = self.evaluate_plan(items, Plan(), state, 'conservative full-precision fallback')
            if reference.estimate.feasible and not self.supports_change(decision, len(items), reference, len(items)):
                decision = reference
        return replace(decision, search_us=(time.perf_counter_ns() - start) / 1000)

    def observe(self, decision, elapsed_ms):
        if not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
            raise ValueError('execution observation must be finite and positive')
        fields = decision.estimate.components
        baseline = fields.get('lookup_base_ms', 0)
        if not decision.estimate.feasible or baseline <= 0:
            return
        phase = fields['timing_features']['phase']
        key = (decision.plan.quantized_layers, bool(decision.plan.cpu_request_ids), phase)
        ratio, error, count = self._phase_feedback.get(key, (1., 0., 0))
        observed = min(4., max(.25, elapsed_ms / baseline))
        alpha = self.ewma_alpha
        prediction_error = min(abs(elapsed_ms - decision.estimate.iteration_ms), 4 * max(1., baseline))
        self._phase_feedback[key] = ((1-alpha)*ratio + alpha*observed,
                                     (1-alpha)*error + alpha*prediction_error, count+1)
