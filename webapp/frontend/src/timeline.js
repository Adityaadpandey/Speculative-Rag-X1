import { gsap } from "gsap";

export const STEP_NAMES = ["fast", "verify", "rerank", "synthesis"];

export function initTimeline(rootEl) {
  const steps = {};
  for (const name of STEP_NAMES) {
    const el = rootEl.querySelector(`.timeline-step[data-step="${name}"]`);
    steps[name] = {
      el,
      ring: el.querySelector(".ring-progress"),
      check: el.querySelector(".ring-check"),
      detailEl: el.querySelector('[data-role="step-detail"]'),
      timeEl: el.querySelector('[data-role="step-time"]'),
      activeTween: null,
    };
  }
  return { steps };
}

const RING_CIRCUMFERENCE = 100.53;
const CHECK_LENGTH = 24;

function resetRing(step) {
  gsap.set(step.ring, { strokeDashoffset: RING_CIRCUMFERENCE });
  gsap.set(step.check, { strokeDashoffset: CHECK_LENGTH });
}

export function setStepState(handle, name, { state, detail, time, hidden } = {}) {
  const step = handle.steps[name];
  if (!step) return;

  if (hidden !== undefined) step.el.hidden = hidden;
  if (detail !== undefined) step.detailEl.textContent = detail;
  if (time !== undefined) step.timeEl.textContent = time;

  if (!state) return;

  if (step.activeTween) {
    step.activeTween.kill();
    step.activeTween = null;
  }
  step.el.classList.remove("pending", "active", "done", "corrected");
  step.el.classList.add(state);

  if (state === "pending") {
    resetRing(step);
  } else if (state === "active") {
    resetRing(step);
    step.activeTween = gsap.to(step.ring, {
      strokeDashoffset: RING_CIRCUMFERENCE * 0.25,
      duration: 1.1,
      repeat: -1,
      yoyo: true,
      ease: "sine.inOut",
    });
  } else if (state === "done") {
    gsap.to(step.ring, { strokeDashoffset: 0, duration: 0.35, ease: "expo.out" });
    gsap.to(step.check, { strokeDashoffset: 0, duration: 0.35, delay: 0.1, ease: "expo.out" });
  } else if (state === "corrected") {
    gsap.to(step.ring, { strokeDashoffset: 0, duration: 0.3, ease: "expo.out" });
    const tl = gsap.timeline();
    tl.to(step.el, { x: -4, duration: 0.05 })
      .to(step.el, { x: 4, duration: 0.05 })
      .to(step.el, { x: -2, duration: 0.05 })
      .to(step.el, { x: 0, duration: 0.05 });
  }
}
