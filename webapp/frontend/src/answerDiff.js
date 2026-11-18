import { diffWords } from "diff";
import { gsap } from "gsap";

// diffWords only counts Latin letters as word characters, so Hindi vowel
// signs and Urdu letters each became their own inline-block span, which
// broke letter joining ("اسلام" rendered as "اسل ام"). Those scripts get a
// plain swap instead of the word-by-word animation.
const NEEDS_SHAPING = /[\u0900-\u0DFF\u0600-\u06FF\u0750-\u077F\uFB50-\uFEFF]/;

export function renderAnswerDiff(containerEl, oldText, newText) {
  if (oldText == null || oldText === newText || NEEDS_SHAPING.test(oldText + newText)) {
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

  // Once the animation settles, swap the spans back for plain text so the
  // final answer is one normally shaped, selectable run of text.
  const tl = gsap.timeline({ onComplete: () => { containerEl.textContent = newText; } });
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
