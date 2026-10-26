"""Decoupled (asynchronous) Dual-System inference.

System 2 runs in a background thread (own CUDA stream) on the newest submitted observation and publishes a
KeyframePlan stamped with that observation's timestamp. System 1 (`act`) never waits for an in-flight System 2
call: it uses the latest COMPLETED plan, places it in time with delay = t_now - plan.obs_time (temporal RoPE), and
returns the next 1 s of actions. The only blocking wait is for the very first plan (wait_for_first_plan=True).

The C-RADIOv4 features of the frame System 1 encodes are handed to System 2 with the submission, so a frame is
encoded by the shared backbone once and reused by both systems.

Model interface (QwenGR00T_DualSystem implements it):
    model.encode_visual(examples) -> VisualFeatures                          (current frame, frozen C-RADIOv4)
    model.plan_keyframes(examples, radio=None) -> keyframes [B, K, D_a]      (System 2, normalized actions)
    model.act_from_plan(examples, keyframes, delay_s [B], radio=None) -> actions [B, N_a, D_a]   (System 1)

Usage:
    runner = AsyncDualSystemRunner(model)
    with runner:                                   # starts / stops the System 2 thread
        while robot.running():
            obs = robot.observe()                  # {"image": [PIL], "lang": str, "state": [1, D_st]}
            actions, info = runner.act([obs])      # info: delay_s, plan_seq, plan_latency_s, ...

AsyncVLMRunner (framework dual_system.stale_input vlm): only the VLM runs in the background thread
(model.encode_vlm_cache on the newest observation -> VLMCache stamped with its time, kept for a short history); every
act runs C-RADIO, System 2's DiT (fresh C-RADIO / state + the cached VLM tokens closest to the trained VLM lag, e.g. the
0.5 s old ones for a model trained with a fixed lag) and System 1 (plan age 0).
"""

import threading
import time
import warnings
from dataclasses import dataclass, field
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional

import torch


@dataclass
class KeyframePlan:
    keyframes: torch.Tensor  # [B, K, D_a] normalized actions, System 2 reference time (t_i relative to obs_time)
    obs_time: float  # clock time (s) of the observation System 2 ran on
    seq: int  # 1, 2, ... in publication order
    latency_s: float  # System 2 compute time
    published_at: float = 0.0


@dataclass
class VLMCache:
    tokens: torch.Tensor  # [B, L_v, D_v] VLM last hidden states of the observation at obs_time
    mask: torch.Tensor  # [B, L_v] bool
    obs_time: float
    seq: int
    latency_s: float  # VLM compute time
    published_at: float = 0.0


@dataclass
class _Job:
    examples: List[dict]
    t_obs: float
    radio: Any = None
    radio_event: Optional[torch.cuda.Event] = None
    submitted_at: float = field(default_factory=time.monotonic)


class AsyncDualSystemRunner:
    def __init__(self, model, clock: Callable[[], float] = time.monotonic, wait_for_first_plan: bool = True,
                 first_plan_timeout_s: float = 120.0, max_plan_age_s: Optional[float] = None,
                 train_delay_range_s: Optional[tuple] = None):
        """max_plan_age_s: warn when a plan older than this is used (default: keyframe horizon - System 1 horizon,
        i.e. the plan no longer covers the next 1 s). train_delay_range_s: (lo, hi) plan ages seen in training,
        reported in info["delay_in_training_range"]."""
        self.model, self.clock = model, clock
        self.wait_for_first_plan, self.first_plan_timeout_s = wait_for_first_plan, float(first_plan_timeout_s)
        if max_plan_age_s is None and hasattr(model, "max_plan_age_s"):
            max_plan_age_s = model.max_plan_age_s
        self.max_plan_age_s = max_plan_age_s
        if train_delay_range_s is None and hasattr(model, "train_delay_range_s"):
            train_delay_range_s = model.train_delay_range_s
        self.train_delay_range_s = train_delay_range_s
        self._cond = threading.Condition()
        self._pending: Optional[_Job] = None
        self._plan: Optional[KeyframePlan] = None
        self._error: Optional[BaseException] = None
        self._stop = False
        self._thread: Optional[threading.Thread] = None
        self._seq = 0
        self.dropped = 0  # submissions replaced by a newer one before System 2 picked them up

    # ------------------------------------------------------------------ lifecycle
    def start(self):
        if self._thread is not None:
            return self
        self._stop = False
        self._thread = threading.Thread(target=self._loop, name="dual-system-s2", daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout: float = 30.0):
        with self._cond:
            self._stop = True
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout)
        self._thread = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    def reset(self):
        """Forget the current plan (new episode); the next act waits for a fresh plan if wait_for_first_plan."""
        with self._cond:
            self._pending, self._plan = None, None

    @property
    def latest_plan(self) -> Optional[KeyframePlan]:
        with self._cond:
            return self._plan

    # ------------------------------------------------------------------ System 2 side
    def submit(self, examples: List[dict], t_obs: Optional[float] = None, radio=None):
        """Non-blocking: queue an observation for System 2 (an older, not yet started one is dropped)."""
        if self._thread is None:
            raise RuntimeError("AsyncDualSystemRunner.start() first")
        ev = None
        if radio is not None and torch.cuda.is_available() and radio.tokens.is_cuda:
            ev = torch.cuda.Event()
            ev.record(torch.cuda.current_stream(radio.tokens.device))
        job = _Job(examples, self.clock() if t_obs is None else float(t_obs), radio, ev)
        with self._cond:
            if self._pending is not None:
                self.dropped += 1
            self._pending = job
            self._cond.notify_all()

    def _loop(self):
        stream = None
        while True:
            with self._cond:
                while self._pending is None and not self._stop:
                    self._cond.wait()
                if self._stop:
                    return
                job, self._pending = self._pending, None
            try:
                t0 = time.perf_counter()
                dev = job.radio.tokens.device if job.radio is not None else getattr(self.model, "_device", None)
                if dev is not None and torch.device(dev).type == "cuda":
                    if stream is None:
                        stream = torch.cuda.Stream(device=dev)
                    with torch.cuda.stream(stream):
                        if job.radio_event is not None:
                            stream.wait_event(job.radio_event)
                            job.radio.tokens.record_stream(stream)
                            job.radio.valid.record_stream(stream)
                        out = self._run_job(job)
                    stream.synchronize()  # the result is complete before it is published
                else:
                    out = self._run_job(job)
                with self._cond:
                    self._seq += 1
                    self._plan = self._make_result(out, job, self._seq, time.perf_counter() - t0)
                    self._cond.notify_all()
            except BaseException as e:  # surfaced in the caller's next act()
                with self._cond:
                    self._error = e
                    self._cond.notify_all()
                return

    def _run_job(self, job: _Job):
        return self.model.plan_keyframes(job.examples, radio=job.radio)

    def _make_result(self, out, job: _Job, seq: int, latency_s: float):
        return KeyframePlan(out, job.t_obs, seq, latency_s, self.clock())

    # ------------------------------------------------------------------ System 1 side
    def _get_plan(self) -> KeyframePlan:
        with self._cond:
            if self._error is not None:
                raise RuntimeError("System 2 thread failed") from self._error
            if self._plan is None:
                if not self.wait_for_first_plan:
                    raise RuntimeError("no System 2 plan yet (wait_for_first_plan=False)")
                deadline = time.monotonic() + self.first_plan_timeout_s
                while self._plan is None and self._error is None:
                    left = deadline - time.monotonic()
                    if left <= 0:
                        raise TimeoutError(f"no System 2 plan after {self.first_plan_timeout_s:.0f} s")
                    self._cond.wait(left)
                if self._error is not None:
                    raise RuntimeError("System 2 thread failed") from self._error
            return self._plan

    @torch.no_grad()
    def act(self, examples: List[dict], t_now: Optional[float] = None, submit: bool = True):
        """-> (actions [B, N_a, D_a] normalized tensor, info). Encodes the frame once (shared C-RADIOv4), submits it
        to System 2 (non-blocking) and runs System 1 on the latest completed plan."""
        t_now = self.clock() if t_now is None else float(t_now)
        radio = self.model.encode_visual(examples)
        if submit:
            self.submit(examples, t_now, radio=radio)
        plan = self._get_plan()
        if plan.keyframes.is_cuda:
            plan.keyframes.record_stream(torch.cuda.current_stream(plan.keyframes.device))
        delay = max(0.0, t_now - plan.obs_time)
        b = plan.keyframes.shape[0]
        if b != len(examples):
            raise ValueError(f"plan batch {b} != {len(examples)} examples")
        delay_t = torch.full((b,), delay, device=plan.keyframes.device)
        actions = self.model.act_from_plan(examples, plan.keyframes, delay_t, radio=radio)
        info: Dict[str, Any] = {"delay_s": delay, "plan_seq": plan.seq, "plan_obs_time": plan.obs_time,
                                "plan_latency_s": plan.latency_s, "dropped_submissions": self.dropped}
        if self.train_delay_range_s is not None:
            lo, hi = self.train_delay_range_s
            info["delay_in_training_range"] = bool(lo - 1e-6 <= delay <= hi + 1e-6)
        if self.max_plan_age_s is not None and delay > self.max_plan_age_s:
            warnings.warn(f"System 2 plan is {delay:.2f} s old (> {self.max_plan_age_s:.2f} s): it no longer covers "
                          "the System 1 horizon")
        return actions, info


class AsyncVLMRunner(AsyncDualSystemRunner):
    """The VLM in a background thread, System 2's DiT + System 1 on every act (framework dual_system.stale_input vlm).

    act never waits for an in-flight VLM call: System 2 plans on the current frame's C-RADIO tokens / state and cached
    VLM tokens (their age is given to system2.vlm_age_embed), and System 1 runs on that plan at once (plan age 0). Only
    the very first act waits for VLM tokens (wait_for_first_plan=True).

    target_age_s: the VLM tokens' age the model expects. Every completed VLM result is kept for history_s; act uses the
        cached tokens whose age is closest to target_age_s and labels them with exactly target_age_s, as in training
        with a fixed VLM lag (frames t - lag). Until tokens at least that old exist (episode start) it uses the newest
        tokens labelled age 0, the training fallback within the first lag frames. "auto" = model.fixed_vlm_age_s (the
        fixed training lag; None if the model was trained on several ages); None = always the newest tokens, labelled
        with their true age.
    max_vlm_age_s: warn when an older age label is used (default: the oldest VLM age seen in training).
    """

    def __init__(self, model, clock: Callable[[], float] = time.monotonic, wait_for_first_plan: bool = True,
                 first_plan_timeout_s: float = 120.0, max_vlm_age_s: Optional[float] = None, target_age_s="auto",
                 history_s: float = 2.0):
        if getattr(model, "stale_input", "vlm") != "vlm":
            raise ValueError("AsyncVLMRunner needs a model trained with dual_system.stale_input vlm; use AsyncDualSystemRunner")
        rng = getattr(model, "train_vlm_age_range_s", None)
        super().__init__(model, clock, wait_for_first_plan, first_plan_timeout_s, max_plan_age_s=float("inf"),
                         train_delay_range_s=rng)
        self.max_vlm_age_s = rng[1] if max_vlm_age_s is None and rng is not None else max_vlm_age_s
        self.target_age_s = getattr(model, "fixed_vlm_age_s", None) if target_age_s == "auto" else target_age_s
        self.age_tol_s = 0.5 / float(getattr(model, "rate", 30.0))  # half a training frame
        self.history_s = max(float(history_s), (self.target_age_s or 0.0) + 1.0)
        self._history: Deque[VLMCache] = deque()
        self._reset_at = -float("inf")  # time.monotonic() of the last reset (jobs submitted before it are dropped)

    def reset(self):
        """New episode: forget the cached VLM tokens; a VLM call still running on the old episode is discarded."""
        with self._cond:
            self._pending, self._plan = None, None
            self._history.clear()
            self._reset_at = time.monotonic()

    def _run_job(self, job: _Job):
        return self.model.encode_vlm_cache(job.examples)

    def _make_result(self, out, job: _Job, seq: int, latency_s: float):  # called with self._cond held
        if job.submitted_at < self._reset_at:
            return self._plan  # observation of the previous episode
        cache = VLMCache(out[0], out[1], job.t_obs, seq, latency_s, self.clock())
        self._history.append(cache)
        while len(self._history) > 1 and cache.obs_time - self._history[0].obs_time > self.history_s:
            self._history.popleft()
        return cache

    def _select(self, t_now: float):
        """-> (cache, age label): the cached tokens closest to target_age_s (labelled target_age_s), else the newest."""
        latest = self._get_plan()
        with self._cond:
            hist = list(self._history) or [latest]
        if self.target_age_s is None:
            return latest, max(0.0, t_now - latest.obs_time)
        tgt = float(self.target_age_s)
        if not any(t_now - c.obs_time >= tgt - self.age_tol_s for c in hist):
            return latest, 0.0  # episode start: no tokens that old yet
        return min(hist, key=lambda c: abs(t_now - c.obs_time - tgt)), tgt

    @torch.no_grad()
    def act(self, examples: List[dict], t_now: Optional[float] = None, submit: bool = True):
        """-> (actions [B, N_a, D_a] normalized tensor, info). Submits the frame to the VLM (non-blocking), then
        C-RADIO -> System 2 (cached VLM tokens, see target_age_s) -> System 1 on the current frame."""
        t_now = self.clock() if t_now is None else float(t_now)
        if submit:
            self.submit(examples, t_now)
        cache, label = self._select(t_now)
        if cache.tokens.is_cuda:
            stream = torch.cuda.current_stream(cache.tokens.device)
            cache.tokens.record_stream(stream)
            cache.mask.record_stream(stream)
        b = cache.tokens.shape[0]
        if b != len(examples):
            raise ValueError(f"VLM cache batch {b} != {len(examples)} examples")
        age = max(0.0, t_now - cache.obs_time)
        radio = self.model.encode_visual(examples)
        kf = self.model.plan_keyframes(examples, radio=radio, vlm=(cache.tokens, cache.mask),
                                       vlm_age_s=torch.full((b,), label, device=cache.tokens.device))
        actions = self.model.act_from_plan(examples, kf, 0.0, radio=radio)
        info: Dict[str, Any] = {"vlm_age_s": age, "vlm_age_label_s": label, "vlm_seq": cache.seq,
                                "vlm_obs_time": cache.obs_time, "vlm_latency_s": cache.latency_s,
                                "dropped_submissions": self.dropped, "keyframes": kf}
        if self.train_delay_range_s is not None:
            lo, hi = self.train_delay_range_s
            info["vlm_age_in_training_range"] = bool(lo - 1e-6 <= label <= hi + 1e-6)
        if self.max_vlm_age_s is not None and label > self.max_vlm_age_s + 1e-6:
            warnings.warn(f"VLM tokens are {label:.2f} s old (> {self.max_vlm_age_s:.2f} s seen in training)")
        if self.target_age_s is not None and label > 0 and abs(age - label) > max(0.1, 3 * self.age_tol_s):
            warnings.warn(f"no cached VLM tokens near {label:.2f} s old (closest {age:.2f} s): VLM calls too slow / irregular")
        return actions, info
