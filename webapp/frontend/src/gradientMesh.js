import gsap from "gsap";

export function initGradientMesh() {
  const mesh = document.getElementById("gradient-mesh");
  if (!mesh) return;

  const blobs = mesh.querySelectorAll(".blob");
  if (blobs.length === 0) return;

  const blob1 = blobs[0];
  const blob2 = blobs[1];
  const blob3 = blobs[2];

  const tl = gsap.timeline({ repeat: -1 });

  tl.to(
    blob1,
    {
      rotate: 360,
      x: "30vw",
      y: "20vh",
      duration: 12,
      ease: "sine.inOut",
    },
    0
  );

  tl.to(
    blob1,
    {
      x: "-10vw",
      y: "-15vh",
      duration: 8,
      ease: "sine.inOut",
    },
    2
  );

  tl.to(
    blob2,
    {
      rotate: -360,
      x: "-25vw",
      y: "-30vh",
      duration: 14,
      ease: "sine.inOut",
    },
    0
  );

  tl.to(
    blob2,
    {
      x: "20vw",
      y: "25vh",
      duration: 9,
      ease: "sine.inOut",
    },
    3
  );

  tl.to(
    blob3,
    {
      rotate: 360,
      x: "15vw",
      y: "35vh",
      duration: 13,
      ease: "sine.inOut",
    },
    0
  );

  tl.to(
    blob3,
    {
      x: "-20vw",
      y: "-10vh",
      duration: 10,
      ease: "sine.inOut",
    },
    1
  );

  return tl;
}
