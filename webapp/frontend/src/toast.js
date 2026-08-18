import { gsap } from "gsap";

let containerEl = null;

export function initToastContainer(el) {
  containerEl = el;
}

export function clearToasts() {
  if (!containerEl) return;
  containerEl.innerHTML = "";
}

export function showToast(message, { type = "error", duration = 6000 } = {}) {
  if (!containerEl) {
    throw new Error("toast container not initialized — call initToastContainer first");
  }
  const el = document.createElement("div");
  el.className = `toast toast-${type}`;
  el.textContent = message;
  containerEl.appendChild(el);

  const tl = gsap.timeline();
  tl.fromTo(el, { x: 40, opacity: 0 }, { x: 0, opacity: 1, duration: 0.35, ease: "power3.out" })
    .to(el, { x: -6, duration: 0.06, ease: "power1.inOut" })
    .to(el, { x: 4, duration: 0.06, ease: "power1.inOut" })
    .to(el, { x: 0, duration: 0.06, ease: "power1.inOut" });

  if (duration > 0) {
    gsap.to(el, {
      opacity: 0,
      x: 40,
      duration: 0.3,
      ease: "power2.in",
      delay: duration / 1000,
      onComplete: () => el.remove(),
    });
  }

  return el;
}
