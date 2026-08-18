import { diffWords } from "diff";
import { gsap } from "gsap";

export function renderAnswerDiff(containerEl, oldText, newText) {
  if (oldText == null || oldText === newText) {
    containerEl.textContent = newText;
    return;
  }

  const parts = diffWords(oldText, newText);
  containerEl.textContent = "";

  const addedSpans = [];
  const removedSpans = [];

  for (const part of parts) {
    const span = document.createElement("span");
    span.className = "diff-word";
    span.textContent = part.value;
    if (part.added) {
      span.classList.add("added");
      addedSpans.push(span);
    } else if (part.removed) {
      span.classList.add("removed");
      removedSpans.push(span);
    }
    containerEl.appendChild(span);
  }

  gsap.set(addedSpans, { opacity: 0, x: -6 });

  const tl = gsap.timeline();
  if (removedSpans.length > 0) {
    tl.to(removedSpans, {
      opacity: 0,
      height: 0,
      duration: 0.25,
      ease: "power1.in",
      onComplete: () => removedSpans.forEach((s) => s.remove()),
    });
  }
  tl.to(
    addedSpans,
    { opacity: 1, x: 0, duration: 0.3, ease: "power2.out", stagger: 0.03 },
    removedSpans.length > 0 ? "-=0.05" : 0
  );

  return tl;
}
