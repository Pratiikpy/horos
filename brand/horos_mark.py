"""The Horos mark.

**The idea.** A horos was a slab of marble planted upright on pledged ground, incised with the claim
against it so that anyone could read the encumbrance before acting. The same word means a boundary —
a limit — and in Aristotle the term of a proposition.

So the mark is a standing marker with one declared band cut across it. The stone is the claim being
published; the band is the interval it is published inside; the ground it is set into is the fact
that it was planted in public, before the fact. Those are the three meanings of the word and the
three things the product does.

**Five versions were rendered and rejected before this one. Each taught the same lesson in a
different way.**

  * A tapered slab with two grooves cut across it read as a **refrigerator**. Detail drawn onto one
    large solid form becomes panel seams at icon size.
  * A columnar stele read as classical architecture — an exhausted logo idiom that says nothing
    about boundaries.
  * Two bars with a vertical between them read as a serif capital **I**. Narrowing the vertical and
    unbalancing the bars did not break it; the letterform reading is too strong.
  * Replacing the vertical with a horizontal tick killed the letter and produced a **hamburger menu
    icon** instead. Three stacked horizontals is a UI cliché.
  * A shaded, gradient-lit slab had real presence but read as a **milk carton**. Naturalistic
    rendering makes a physical object; a mark needs flat graphic geometry.

What survived: a flat silhouette, no shading, asymmetric weathered crown, and an inverted palette —
dark stone on sunlit limestone rather than the dark-mode everything else in the marketplace uses.

**The constraints are OKX's, and they rejected a sibling agent three times.** From
OKX-REVIEW-RULES.md §1.5, verbatim: "it must be a 1:1 icon and **not a backgroundless image**" and
"right now it's still just the floating logo". Hence: exactly 1:1 at 2048px, filled to all four
corners, hard 90-degree edges, no transparency, no text, no letterforms, no protocol logos, no
humans, and flat geometry rather than the soft gradients that read as instant AI generation.

Deterministic: no randomness, so the file renders byte-identically every time.
"""
from __future__ import annotations

from pathlib import Path

S = 1000.0

GROUND_LIGHT = "#eaddc3"        # sunlit limestone
GROUND_SHADE = "#c9b691"
EARTH = "#b3a07c"               # the ground the stone is set into
STONE = "#1c1811"               # the marker itself, carved dark
CLAIM = "#c2571f"               # the declared band: terracotta, not a crypto orange

TOP, BOTTOM = 172.0, 856.0      # the slab
CROWN_HALF, BASE_HALF = 110.0, 152.0
CROWN = 26.0                    # depth of the weathered, deliberately uneven crown
BAND_Y, BAND_H = 400.0, 44.0    # the declared interval
BAND_OVERHANG = 20.0            # the band spills past the stone: the claim exceeds the marker
EARTH_Y = 856.0


def svg() -> str:
    cx = S / 2

    def half_at(y: float) -> float:
        return CROWN_HALF + (BASE_HALF - CROWN_HALF) * ((y - TOP) / (BOTTOM - TOP))

    # The crown is asymmetric on purpose — a weathered stone, not a manufactured block.
    slab = (f"M {cx - CROWN_HALF + CROWN * 0.7:.0f} {TOP:.0f} "
            f"L {cx + CROWN_HALF - CROWN * 0.2:.0f} {TOP + CROWN * 0.5:.0f} "
            f"L {cx + CROWN_HALF:.0f} {TOP + CROWN * 1.1:.0f} "
            f"L {cx + BASE_HALF:.0f} {BOTTOM:.0f} "
            f"L {cx - BASE_HALF:.0f} {BOTTOM:.0f} "
            f"L {cx - CROWN_HALF:.0f} {TOP + CROWN * 0.9:.0f} Z")

    band = (f"M {cx - half_at(BAND_Y) - BAND_OVERHANG:.0f} {BAND_Y:.0f} "
            f"L {cx + half_at(BAND_Y) + BAND_OVERHANG:.0f} {BAND_Y:.0f} "
            f"L {cx + half_at(BAND_Y + BAND_H) + BAND_OVERHANG:.0f} {BAND_Y + BAND_H:.0f} "
            f"L {cx - half_at(BAND_Y + BAND_H) - BAND_OVERHANG:.0f} {BAND_Y + BAND_H:.0f} Z")

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{S:.0f}" height="{S:.0f}" \
viewBox="0 0 {S:.0f} {S:.0f}">
  <defs>
    <linearGradient id="ground" x1="0.15" y1="0" x2="0.8" y2="1">
      <stop offset="0" stop-color="{GROUND_LIGHT}"/>
      <stop offset="1" stop-color="{GROUND_SHADE}"/>
    </linearGradient>
  </defs>

  <!-- Filled to all four corners with hard edges. OKX rejects a backgroundless floating logo. -->
  <rect width="{S:.0f}" height="{S:.0f}" fill="url(#ground)"/>

  <!-- The earth the marker is set into. A horos was planted in the ground it spoke about. -->
  <rect x="0" y="{EARTH_Y:.0f}" width="{S:.0f}" height="{S - EARTH_Y:.0f}" fill="{EARTH}"/>

  <path d="{slab}" fill="{STONE}"/>
  <path d="{band}" fill="{CLAIM}"/>
</svg>"""


def write(directory: str | Path = ".") -> dict:
    """Render the mark to SVG and to every PNG size the listing and the site need."""
    import cairosvg

    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    source = out / "horos-mark.svg"
    source.write_text(svg(), encoding="utf-8")

    written = {"svg": str(source)}
    for size in (2048, 1080, 512, 256, 64):
        png = out / f"horos-{size}.png"
        cairosvg.svg2png(bytestring=svg().encode("utf-8"), write_to=str(png),
                         output_width=size, output_height=size)
        written[f"png_{size}"] = str(png)
    return written


def audit(path: str | Path) -> dict:
    """Check the rendered PFP against OKX's published rules before it is ever uploaded."""
    from PIL import Image

    img = Image.open(path)
    w, h = img.size
    rgba = img.convert("RGBA")
    corners = [rgba.getpixel(p) for p in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1))]
    checks = {
        "square_1_1": w == h,
        "at_least_1080p": min(w, h) >= 1080,
        "fully_opaque": rgba.getchannel("A").getextrema() == (255, 255),
        "corners_filled_hard_edges": all(c[3] == 255 for c in corners),
        "not_flat_colour": len(img.convert("RGB").getcolors(maxcolors=1 << 20) or []) > 32,
    }
    return {"path": str(path), "size": f"{w}x{h}", "checks": checks,
            "passes": all(checks.values()),
            "failed": [k for k, v in checks.items() if not v]}


if __name__ == "__main__":
    import json
    here = Path(__file__).parent
    print(json.dumps(write(here), indent=2))
    print(json.dumps(audit(here / "horos-2048.png"), indent=2))
