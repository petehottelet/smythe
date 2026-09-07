# House style

Smythe's visual system is editorial, inscriptional, and strictly monochrome.
Landing-page assets should feel like a well-set title page: decisive hierarchy,
precise rules, generous white space, and no decorative interface chrome.

The Glyph Rain screenshot, atlas, animation, and explorer controls are the
intentional color exception. They show the green product artifact itself. Every graph, chart,
diagram, landing-page wordmark, and callout uses black and white only.

## Palette

| Role | Hex | Use |
|---|---|---|
| Ink | `#000000` | Text, rules, arrows, solid data marks, emphasis surfaces |
| Paper | `#ffffff` | Canvas, open data marks, reversed text |

Do not use gray, color, transparency, gradients, glows, shadows, or tinted
fills. When a third visual category is required, use a black hatch, dots, a
double rule, or a dashed outline on white.

## Type

- **Wordmark**: inscriptional capitals baked into SVG paths so the mark renders
  identically without a webfont.
- **Callout numerals**: `Trajan Pro 3`, `Trajan Pro`, Trajan, Cinzel, Georgia,
  serif, in that order. Trajan is reserved for the large number; explanatory
  copy remains quiet and compact.
- **Chart and diagram titles**: Georgia or Times New Roman.
- **Labels and annotations**: Avenir Next or Helvetica Neue; monospace only for
  source records and exact machine values.
- Markdown body text is set by GitHub and should not be restyled with HTML.

## Charts and callouts

- Render charts from committed result records with
  `python benchmarks/render_readme_charts.py`.
- Prefer small multiples with direct values over decorative dashboard cards.
- Encode Smythe as solid black, the primary comparison as white with a black
  outline, and a third comparison with black hatching on white.
- Keep scales honest and state whether higher or lower is better.
- Put the result record and sample size on the asset itself.
- Normalize JSON source-record newlines to LF before embedding a chart's
  source hash. Windows and Linux checkouts must identify the same record.
- Use square rules and open space. Avoid rounded panels, pills, gradients,
  shadows, and glow effects.
- A callout pairs one Trajan numeral with one exact comparison. It never
  substitutes for the underlying chart.

## Diagrams

Use hand-authored SVG for landing-page diagrams whose routing and hierarchy
matter. Catalog-bound plates, such as the Glyph Rain specimens, should be
rendered deterministically from their committed source data. Use Mermaid for
detailed documentation diagrams that must track code; keep it off the landing
page when a polished example graph already tells the story.

Differentiate node states without color:

- completed: white fill, black outline;
- failed or final deliverable: black fill, white text;
- skipped: white fill, dashed black outline;
- running: white fill, heavy black outline.

Edges remain solid black hairlines. Use orthogonal routes for showpiece SVGs
and gentle Mermaid curves for generated graphs.

## Mermaid

Every public diagram carries the same pure black-and-white init header, kept as
`MERMAID_THEME` in `smythe/graph.py`:

```text
%%{init: {"theme":"base","themeVariables":{"fontFamily":"Georgia, 'Times New Roman', serif","fontSize":"14px","primaryColor":"#ffffff","primaryTextColor":"#000000","primaryBorderColor":"#000000","lineColor":"#000000","secondaryColor":"#ffffff","tertiaryColor":"#ffffff","background":"#ffffff","mainBkg":"#ffffff","clusterBkg":"#ffffff","clusterBorder":"#000000"},"flowchart":{"curve":"basis","nodeSpacing":48,"rankSpacing":58}}}%%
```

## Badges

Use the repository-native SVG badges in `assets/badges/`. Each uses the
conventional two-tone split—a black label field with white type and a white
value field with black type—inside a complete one-pixel black rectangular
border. Keep their values synchronized with the release and CI state.

## Explorer controls

Use VT323 pixel lettering for explorer headings, labels, numerals, buttons,
inputs, and help text. The separate SMYTHE logo uses outlined Trajan Pro Bold,
with generous black padding above and below it. Keep text controls borderless,
without interface glow, corner ornaments, or stepped edges.
Use primary `#37FF6E`, bright `#9CFFBC`, and black, following
[hottelet.com](https://www.hottelet.com/). Focus and selection use visible
outlines or solid fills. These UI colors do not change the rain's 137°
Matrix green body grade, `#A2FFD8` leading glyphs, or bloom.
This explorer treatment does not change the chart typography or the README's
strictly black-and-white charts.

## Verification

After changing graph assets, regenerate them, scan every active SVG and Mermaid
file for colors outside `#000000` and `#ffffff`, then render the SVGs to pixels
for visual review. Pattern fills and `none` are allowed; translucent marks are
not.

For raster reproduction, compare decoded mode, dimensions, and pixel hashes.
PNG compression can differ across platforms without changing the image.
Keep exact file hashes in download receipts and byte-level checks for source
artwork, licenses, SVG geometry, and structured catalog exports.
