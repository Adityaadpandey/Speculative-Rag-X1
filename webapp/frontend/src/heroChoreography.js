import { gsap } from "gsap";

export function collapseHero({ heroEl, compactBarEl }) {
  const heroInner = heroEl.querySelector(".hero-inner");
  const tl = gsap.timeline();

  // `hidden` is a plain boolean attribute, not a CSS property GSAP's CSSPlugin
  // understands -- gsap.set(el, { hidden: true }) would silently no-op, so
  // toggling it goes through tl.call() instead.
  tl.to(heroInner, { opacity: 0, y: -24, duration: 0.28, ease: "power2.in" })
    .call(() => { heroEl.hidden = true; })
    .call(() => { compactBarEl.hidden = false; })
    .fromTo(
      compactBarEl,
      { opacity: 0, y: -16 },
      { opacity: 1, y: 0, duration: 0.3, ease: "power2.out" }
    );

  return tl;
}
