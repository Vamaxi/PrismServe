"""Read-only experiment instrumentation; unsupported driver fields stay null."""
from __future__ import annotations

import csv
from datetime import datetime
import json
from pathlib import Path
import platform
import shutil
import subprocess
import threading
import time

GPU_FIELDS = (
    'timestamp', 'index', 'uuid', 'temperature.gpu', 'clocks.current.sm',
    'clocks.current.memory', 'power.draw', 'utilization.gpu', 'memory.used', 'pstate',
    'clocks_event_reasons.sw_power_cap', 'clocks_event_reasons.hw_thermal_slowdown',
    'clocks_event_reasons.sw_thermal_slowdown',
)


def parse_gpu_csv(text):
    rows = []
    for raw in csv.reader(text.splitlines()):
        if len(raw) != len(GPU_FIELDS):
            continue
        values = [v.strip() for v in raw]
        try:
            # nvidia-smi timestamps use the operating system's local timezone.
            stamp = datetime.strptime(values[0], '%Y/%m/%d %H:%M:%S.%f').timestamp()
        except ValueError:
            continue
        row = {'unix_s': stamp}
        for field, value in zip(GPU_FIELDS[1:], values[1:]):
            if value in ('N/A', '[N/A]', '[Not Supported]', 'Not Supported', ''):
                row[field] = None
            elif value in ('Active', 'Not Active'):
                row[field] = value == 'Active'
            else:
                try:
                    row[field] = float(value)
                except ValueError:
                    row[field] = value
        rows.append(row)
    return rows


def summarize_gpu(rows, start, end):
    selected = [r for r in rows if start <= r['unix_s'] <= end]
    result = {'samples': len(selected), 'window_start_unix_s': start, 'window_end_unix_s': end}
    for field in GPU_FIELDS[3:]:
        values = [r[field] for r in selected if isinstance(r.get(field), (int, float))]
        if values:
            result[field] = {'min': min(values), 'max': max(values), 'mean': sum(values)/len(values)}
        else:
            result[field] = None
    return result


def machine_conditions():
    import psutil
    battery = psutil.sensors_battery()
    memory = psutil.virtual_memory()
    return {'unix_s': time.time(), 'platform': platform.platform(), 'cpu_name': platform.processor(),
            'logical_cpus': psutil.cpu_count(), 'host_available_bytes': memory.available,
            'host_total_bytes': memory.total,
            'battery_percent': None if battery is None else battery.percent,
            'power_plugged': None if battery is None else battery.power_plugged}


class RunMonitor:
    """One persistent driver query process plus a low-rate host sampler per run.

    No power/clock settings are changed. Instrumentation also runs for baselines.
    Host and GPU samples are asynchronous observations, not allocator guarantees.
    """
    def __init__(self, destination: Path, cuda: bool, interval_s=.5):
        self.destination, self.cuda, self.interval = Path(destination), cuda, interval_s
        self.process = self.handle = self.thread = None
        self.stop_event = threading.Event()
        self.host = []
        self.error = None

    def __enter__(self):
        import os
        import psutil
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self.before = machine_conditions()
        self.owner = psutil.Process()
        if self.cuda:
            program = shutil.which('nvidia-smi')
            if program:
                self.handle = self.destination.with_suffix('.gpu.csv').open('w', encoding='utf-8')
                command = [program, '--id=0', '--query-gpu='+','.join(GPU_FIELDS), '--format=csv,noheader,nounits',
                           '--loop-ms='+str(round(self.interval*1000))]
                options = {'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {}
                self.process = subprocess.Popen(command, stdout=self.handle, stderr=subprocess.STDOUT, **options)
            else:
                self.error = 'nvidia-smi unavailable; no clock or thermal claims can be made'
        def sample():
            while not self.stop_event.is_set():
                memory = psutil.virtual_memory()
                battery = psutil.sensors_battery()
                self.host.append({'unix_s': time.time(), 'process_rss_bytes': self.owner.memory_info().rss,
                                  'host_available_bytes': memory.available, 'cpu_utilization_pct': psutil.cpu_percent(),
                                  'power_plugged': None if battery is None else battery.power_plugged,
                                  'battery_percent': None if battery is None else battery.percent})
                self.stop_event.wait(self.interval)
        self.thread = threading.Thread(target=sample, name='study-resource-sampler', daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop_event.set()
        self.thread.join(timeout=5)
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
            self.handle.close()
        self.after = machine_conditions()
        gpu = [] if self.handle is None else parse_gpu_csv(self.destination.with_suffix('.gpu.csv').read_text())
        if self.cuda and not gpu and not self.error:
            self.error = 'driver query returned no usable samples; raw output retained'
        self.gpu = gpu
        self.destination.with_suffix('.json').write_text(json.dumps({
            'before': self.before, 'after': self.after, 'gpu_fields': GPU_FIELDS,
            'gpu_samples': gpu, 'host_samples': self.host, 'error': self.error,
            'notes': ['Driver event flags describe reported clock-limit reasons; correlation does not establish causality.',
                      'Read-only sampling at 500 ms. Clock settings and laptop power source are not controlled.']}, indent=2), encoding='utf-8')

    def summary(self, start, end):
        return {'gpu': summarize_gpu(self.gpu, start, end), 'power_plugged_before': self.before['power_plugged'],
                'power_plugged_after': self.after['power_plugged'], 'monitor_error': self.error,
                'sampled_peak_process_rss_bytes': max((r['process_rss_bytes'] for r in self.host if start <= r['unix_s'] <= end), default=None)}
