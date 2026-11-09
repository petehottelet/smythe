use anyhow::{Result, bail, ensure};
use geo::{Area, BoundingRect, Contains, Intersects, LineString, Polygon, Validation};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const ORIGINALS: &str = include_str!("../../../benchmarks/noumenon/glyph_contours_v2.json");
pub const EXTRA: &str =
    include_str!("../../../benchmarks/partitions/glyph_svg_v2_256/contours-extra.json");
pub type Point = (f64, f64);

#[derive(Clone, Debug, Deserialize)]
pub struct Recipe {
    pub recipe: String,
    pub outline: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Glyph {
    pub glyph_id: String,
    pub recipe: String,
    pub version: String,
    pub svg: String,
    pub svg_sha256: String,
    pub paths: Vec<Vec<Vec<Point>>>,
}

pub fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
pub fn source_sha() -> String {
    sha(format!("{ORIGINALS}\n{EXTRA}").as_bytes())
}

pub fn implementation_sources() -> std::collections::BTreeMap<&'static str, String> {
    [
        (
            "crates/smythe/src/seams.rs",
            include_bytes!("../../smythe/src/seams.rs").as_slice(),
        ),
        (
            "crates/smythe-core/src/errors.rs",
            include_bytes!("../../smythe-core/src/errors.rs").as_slice(),
        ),
        (
            "clippy.toml",
            include_bytes!("../../../clippy.toml").as_slice(),
        ),
        (
            "crates/smythe-noumenon/src/measurements.rs",
            include_bytes!("measurements.rs").as_slice(),
        ),
        (
            "crates/smythe-core/src/money.rs",
            include_bytes!("../../smythe-core/src/money.rs").as_slice(),
        ),
        (
            "crates/smythe-core/src/journal.rs",
            include_bytes!("../../smythe-core/src/journal.rs").as_slice(),
        ),
        (
            "crates/smythe/Cargo.toml",
            include_bytes!("../../smythe/Cargo.toml").as_slice(),
        ),
        (
            "crates/smythe/src/lib.rs",
            include_bytes!("../../smythe/src/lib.rs").as_slice(),
        ),
        (
            "crates/smythe/src/runtime.rs",
            include_bytes!("../../smythe/src/runtime.rs").as_slice(),
        ),
        (
            "crates/smythe/src/store.rs",
            include_bytes!("../../smythe/src/store.rs").as_slice(),
        ),
        (
            "crates/smythe-cli/Cargo.toml",
            include_bytes!("../../smythe-cli/Cargo.toml").as_slice(),
        ),
        (
            "Cargo.toml",
            include_bytes!("../../../Cargo.toml").as_slice(),
        ),
        (
            "Cargo.lock",
            include_bytes!("../../../Cargo.lock").as_slice(),
        ),
        (
            "crates/smythe-core/Cargo.toml",
            include_bytes!("../../smythe-core/Cargo.toml").as_slice(),
        ),
        (
            "crates/smythe-core/src/lib.rs",
            include_bytes!("../../smythe-core/src/lib.rs").as_slice(),
        ),
        (
            "crates/smythe-noumenon/Cargo.toml",
            include_bytes!("../Cargo.toml").as_slice(),
        ),
        (
            "crates/smythe-noumenon/src/lib.rs",
            include_bytes!("lib.rs").as_slice(),
        ),
        (
            "crates/smythe-cli/src/main.rs",
            include_bytes!("../../smythe-cli/src/main.rs").as_slice(),
        ),
        (
            "crates/smythe-noumenon/src/geometry.rs",
            include_bytes!("geometry.rs").as_slice(),
        ),
        (
            "crates/smythe-noumenon/src/validation.rs",
            include_bytes!("validation.rs").as_slice(),
        ),
        (
            "crates/smythe-noumenon/src/pipeline.rs",
            include_bytes!("pipeline.rs").as_slice(),
        ),
    ]
    .into_iter()
    .map(|(name, bytes)| (name, sha(bytes)))
    .collect()
}
pub fn recipes() -> Result<Vec<Recipe>> {
    let mut result: Vec<Recipe> = serde_json::from_str(ORIGINALS)?;
    ensure!(
        result.len() == 192,
        "original catalog must contain 192 glyphs"
    );
    result.extend(serde_json::from_str::<Vec<Recipe>>(EXTRA)?);
    ensure!(
        result.len() == 256,
        "extended catalog must contain 256 glyphs"
    );
    Ok(result)
}

fn area(ring: &[Point]) -> f64 {
    ring.iter()
        .zip(ring.iter().cycle().skip(1))
        .take(ring.len())
        .map(|(&(x, y), &(a, b))| x * b - a * y)
        .sum::<f64>()
        / 2.0
}

/// Parse the authored M/L/C/Z grammar, with exactly 64 samples per cubic.
fn rings(outline: &str) -> Result<Vec<Vec<Point>>> {
    ensure!(
        outline.len() <= 262_144 && outline.is_ascii(),
        "invalid contour input"
    );
    let mut tokens = Vec::new();
    let mut start = 0;
    let b = outline.as_bytes();
    while start < b.len() {
        if b[start].is_ascii_whitespace() || b[start] == b',' {
            start += 1;
            continue;
        }
        let mut end = start + 1;
        if !b"MLCZ".contains(&b[start]) {
            ensure!(
                b[start].is_ascii_digit() || b[start] == b'-' || b[start] == b'.',
                "unsupported contour command"
            );
            while end < b.len() && (b[end].is_ascii_digit() || b[end] == b'.') {
                end += 1;
            }
        }
        tokens.push(&outline[start..end]);
        start = end;
    }
    let mut result = vec![];
    let mut ring = vec![];
    let mut position = 0;
    while position < tokens.len() {
        let command = tokens[position];
        position += 1;
        if command == "Z" {
            if ring.first() == ring.last() {
                ring.pop();
            }
            ensure!(ring.len() >= 3, "degenerate contour");
            result.push(std::mem::take(&mut ring));
            continue;
        }
        ensure!(
            (command == "M" && ring.is_empty())
                || ((command == "L" || command == "C") && !ring.is_empty()),
            "unclosed or malformed contour"
        );
        let count = if command == "C" { 6 } else { 2 };
        ensure!(position + count <= tokens.len(), "missing coordinates");
        let values: Vec<f64> = tokens[position..position + count]
            .iter()
            .map(|v| v.parse())
            .collect::<std::result::Result<_, _>>()?;
        ensure!(
            values
                .iter()
                .all(|v| v.is_finite() && (0.0..=100.0).contains(v)),
            "coordinates outside viewBox"
        );
        position += count;
        if command != "C" {
            ring.push((values[0], values[1]));
        } else {
            let (ax, ay) = *ring.last().unwrap();
            for step in 1..=64 {
                let t = f64::from(step) / 64.0;
                let u = 1.0 - t;
                let point = |a, b, c, d| {
                    u.powi(3) * a + 3.0 * u * u * t * b + 3.0 * u * t * t * c + t.powi(3) * d
                };
                ring.push((
                    point(ax, values[0], values[2], values[4]),
                    point(ay, values[1], values[3], values[5]),
                ));
            }
        }
        ensure!(ring.len() <= 16_000, "contour too complex");
    }
    ensure!(
        ring.is_empty() && !result.is_empty(),
        "unclosed or empty contour"
    );
    Ok(result)
}

fn polygon(rings: &[Vec<Point>]) -> Polygon<f64> {
    Polygon::new(
        LineString::from(rings[0].clone()),
        rings[1..]
            .iter()
            .map(|r| LineString::from(r.clone()))
            .collect(),
    )
}

fn canonical_ring(mut ring: Vec<Point>, exterior: bool) -> Vec<Point> {
    if (area(&ring) > 0.0) != exterior {
        ring.reverse();
    }
    // Decimal formatting reproduces Python's round-to-three-decimals, including
    // binary floating-point ties, before choosing the canonical start vertex.
    for (x, y) in &mut ring {
        *x = format!("{x:.3}").parse().unwrap();
        *y = format!("{y:.3}").parse().unwrap();
    }
    let first = (0..ring.len())
        .min_by(|&a, &b| ring[a].partial_cmp(&ring[b]).unwrap())
        .unwrap();
    ring.rotate_left(first);
    ring
}

pub fn compile(index: usize, recipe: &Recipe) -> Result<Glyph> {
    ensure!(index < 256, "glyph index outside catalog");
    let rings = rings(&recipe.outline)?;
    let shapes: Vec<_> = rings
        .iter()
        .map(|r| Polygon::new(LineString::from(r.clone()), vec![]))
        .collect();
    ensure!(
        shapes
            .iter()
            .all(|p| p.is_valid() && p.unsigned_area() >= 0.01),
        "invalid authored ring"
    );
    let mut filled = vec![];
    for (i, shape) in shapes.iter().enumerate() {
        if shapes
            .iter()
            .enumerate()
            .any(|(j, p)| j != i && p.contains(shape))
        {
            continue;
        }
        let mut part = vec![rings[i].clone()];
        part.extend(
            shapes
                .iter()
                .enumerate()
                .filter(|&(j, p)| j != i && shape.contains(p))
                .map(|(j, _)| rings[j].clone()),
        );
        let poly = polygon(&part);
        ensure!(poly.is_valid(), "invalid filled contour");
        filled.push((poly, part));
    }
    for (i, (p, _)) in filled.iter().enumerate() {
        for (q, _) in &filled[..i] {
            ensure!(
                !p.intersects(q),
                "overlapping authored pieces need an explicit union implementation"
            );
        }
    }
    filled.sort_by(|(a, _), (b, _)| {
        b.unsigned_area()
            .total_cmp(&a.unsigned_area())
            .then_with(|| {
                let a = a.bounding_rect().unwrap();
                let b = b.bounding_rect().unwrap();
                [a.min().x, a.min().y, a.max().x, a.max().y]
                    .partial_cmp(&[b.min().x, b.min().y, b.max().x, b.max().y])
                    .unwrap()
            })
    });
    let paths: Vec<Vec<Vec<Point>>> = filled
        .into_iter()
        .map(|(_, part)| {
            part.into_iter()
                .enumerate()
                .map(|(i, r)| canonical_ring(r, i == 0))
                .collect()
        })
        .collect();
    ensure!(
        paths.iter().all(|p| polygon(p).is_valid()),
        "rounded contour is invalid"
    );
    let mut svg = String::from(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 100 100\" width=\"128\" height=\"128\">",
    );
    for path in &paths {
        let data: Vec<_> = path
            .iter()
            .map(|ring| {
                let points: Vec<_> = ring.iter().map(|(x, y)| format!("{x} {y}")).collect();
                format!("M{} Z", points.join(" L"))
            })
            .collect();
        svg.push_str(&format!(
            "<path fill=\"#000000\" fill-rule=\"nonzero\" d=\"{}\"/>",
            data.join(" ")
        ));
    }
    svg.push_str("</svg>\n");
    if svg.len() > 262_144 {
        bail!("SVG exceeds feed size limit");
    }
    Ok(Glyph {
        glyph_id: format!("GLYPH-{index:03}"),
        recipe: recipe.recipe.clone(),
        version: "glyph-svg-v2-1".into(),
        svg_sha256: sha(svg.as_bytes()),
        svg,
        paths,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn catalog_is_deterministic_and_feed_compatible() {
        let all = recipes().unwrap();
        let reference: Vec<String> =
            serde_json::from_str(include_str!("../tests/fixtures/python_svg_sha256.json")).unwrap();
        assert_eq!(reference.len(), all.len());
        let mut hashes = std::collections::HashSet::new();
        for (i, r) in all.iter().enumerate() {
            let g = compile(i, r).unwrap();
            assert_eq!(g.svg_sha256, reference[i], "Python SVG parity at {i}");
            assert!(hashes.insert(g.svg_sha256.clone()));
            assert_eq!(g.svg_sha256, compile(i, r).unwrap().svg_sha256);
            assert_eq!(g.glyph_id, format!("GLYPH-{i:03}"));
            assert!(
                g.paths
                    .iter()
                    .all(|p| area(&p[0]) > 0.0 && p[1..].iter().all(|r| area(r) < 0.0))
            );
        }
    }
    #[test]
    fn malformed_or_overlapping_inputs_fail_closed() {
        for outline in [
            "M0 0 L1 1",
            "M0 0 Q5 5 10 10 Z",
            "M0 0 L101 0 L0 10 Z",
            "M0 0 L10 10 L0 10 L10 0 Z",
            "M0 0 L10 0 L10 10 L0 10 Z M5 0 L15 0 L15 10 L5 10 Z",
        ] {
            assert!(
                compile(
                    0,
                    &Recipe {
                        recipe: "bad".into(),
                        outline: outline.into()
                    }
                )
                .is_err()
            );
        }
    }
}
