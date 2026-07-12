# House style

One consistent visual voice across the README, docs, diagrams, and
generated graph exports. Understated, warm, serif. If an element
wouldn't look at home on a well-set title page, it doesn't ship.

## Palette

| Role | Hex | Use |
|---|---|---|
| Ink | `#23221e` | Text, the deliverable node, badge labels |
| Paper | `#faf8f1` | Default node fill |
| Parchment | `#f1ecdf` | Secondary node fill (editors, joins) |
| Champagne tint | `#f5ead0` | Accent fill (adversarial/red-team, running) |
| Hairline | `#a89f8c` | Borders, edges, arrows |
| Gold | `#9a7b2d` | The accent: badge values, deliverable border, running stroke |
| Ivory ink | `#f5efe0` / `#ece7da` | Text on ink-filled surfaces |

Status colors (graph exports): done `#eef0e4`/`#5a7742`, failed
`#f3e0dd`/`#8c3b2e`, skipped `#eceae3`/`#8a8578`, running
`#f5ead0`/`#9a7b2d`. Muted, same warmth — never traffic-light saturation.

## Type

- **Wordmark**: Trajan-style inscriptional capitals (Cinzel, baked to
  SVG paths in `assets/wordmark.svg` so it renders identically
  everywhere — webfonts don't load inside GitHub-proxied images).
- **Diagrams**: `Georgia, 'Times New Roman', serif` — system serifs
  that Mermaid can actually use on a viewer's machine.
- Markdown body text is set by GitHub and can't be styled; don't fight
  it with HTML hacks.

## Mermaid

Every diagram carries this init header (kept as `MERMAID_THEME` in
`smythe/graph.py`; `ExecutionGraph.to_mermaid(theme=True)` emits it):

```text
%%{init: {"theme":"base","themeVariables":{"fontFamily":"Georgia, 'Times New Roman', serif","fontSize":"14px","primaryColor":"#faf8f1","primaryTextColor":"#23221e","primaryBorderColor":"#a89f8c","lineColor":"#a89f8c"},"flowchart":{"curve":"basis","nodeSpacing":48,"rankSpacing":58}}}%%
```

Rules:

- `curve: basis` — gentle curves, never `step` (elbow arrows read as
  circuit diagrams, and GitHub's renderer draws them awkwardly).
- Edges and borders stay hairline (`1px`–`1.25px`), colored `#a89f8c`.
- One emphasis node per diagram at most: ink fill, gold border
  (`fill:#23221e,stroke:#9a7b2d,color:#f5efe0`) — the deliverable.
- Node titles `<b>bold</b>`, detail on a second line, five words or
  fewer per line where possible.

## Badges

shields.io flat-square, ink label + gold value:
`?style=flat-square&labelColor=23221e&color=9a7b2d`.

## Regenerating the wordmark

`assets/wordmark.svg` is generated from Cinzel (OFL) at weight 560,
52px, +14 tracking, via fontTools' SVGPathPen — glyph outlines, no
text elements, dark-mode aware via `prefers-color-scheme`. If the
wording ever changes, re-bake rather than editing paths by hand.
