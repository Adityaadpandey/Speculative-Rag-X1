import { gsap } from "gsap";

export function initGradientMesh(rootEl) {
  const blobs = rootEl.querySelectorAll(".blob");
  const tl = gsap.timeline({ repeat: -1, yoyo: true, defaults: { ease: "sine.inOut" } });

  blobs.forEach((blob, i) => {
    tl.to(
      blob,
      {
        x: () => gsap.utils.random(-80, 80),
        y: () => gsap.utils.random(-60, 60),
        scale: () => gsap.utils.random(0.85, 1.25),
        duration: 8 + i * 2,
      },
      0
    );
  });

  return tl;
}
