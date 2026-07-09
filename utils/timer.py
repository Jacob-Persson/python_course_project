import time
from collections import defaultdict

_timers = defaultdict(lambda: {'total': 0.0, 'count': 0})
_stack = {}
_enabled = False

def enable():
    global _enabled
    _enabled = True

def disable():
    global _enabled
    _enabled = False

def tic(name):
    if not _enabled:
        return
    _stack[name] = time.perf_counter()

def toc(name):
    if not _enabled or name not in _stack:
        return
    elapsed = time.perf_counter() - _stack.pop(name)
    _timers[name]['total'] += elapsed
    _timers[name]['count'] += 1

def clear():
    _timers.clear()
    _stack.clear()

def report():
    if not _timers:
        return
    print("\n============= TIMING REPORT =============")
    print(f"{'Name':<45s} {'Total (s)':>10s} {'Calls':>8s} {'Avg (ms)':>10s}")
    print("-" * 75)
    for name in sorted(_timers):
        d = _timers[name]
        avg_ms = d['total'] / d['count'] * 1e3 if d['count'] else 0.0
        print(f"{name:<45s} {d['total']:>10.3f} {d['count']:>8d} {avg_ms:>10.3f}")
    print("=========================================\n")
