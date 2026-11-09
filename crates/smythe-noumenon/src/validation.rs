//! Native, versioned raster and acceptance gates. Pixel hashes are specific to
//! tiny-skia. Analytic fixtures qualify native measurements independently.
use crate::geometry::{Glyph, sha};
use anyhow::{Result, anyhow, ensure};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use tiny_skia::{Color, FillRule, Paint, PathBuilder, Pixmap, Transform};

pub const RASTER_VERSION: &str = "tiny-skia-0.11-fill-v1";
pub const GATE_VERSION: &str = "noumenon-native-topology-style-v1";

pub fn render(glyph: &Glyph, size: usize) -> Result<Pixmap> {
    ensure!([16, 32, 64, 128].contains(&size), "unsupported raster size");
    let mut pixels =
        Pixmap::new(size as u32, size as u32).ok_or_else(|| anyhow!("raster allocation failed"))?;
    pixels.fill(Color::WHITE);
    let paint = Paint::default();
    for part in &glyph.paths {
        let mut builder = PathBuilder::new();
        for ring in part {
            for (i, &(x, y)) in ring.iter().enumerate() {
                let x = (x * size as f64 / 100.0) as f32;
                let y = (y * size as f64 / 100.0) as f32;
                if i == 0 {
                    builder.move_to(x, y);
                } else {
                    builder.line_to(x, y);
                }
            }
            builder.close();
        }
        let path = builder
            .finish()
            .ok_or_else(|| anyhow!("invalid raster path"))?;
        pixels.fill_path(
            &path,
            &paint,
            FillRule::Winding,
            Transform::identity(),
            None,
        );
    }
    Ok(pixels)
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Measurement {
    pub size: usize,
    pub pixel_sha256: String,
    pub ink_pixels: usize,
    pub components: usize,
    pub holes: usize,
    pub threshold_topology_stable: bool,
    pub style: Option<crate::measurements::Style>,
}

fn topology(mask: &[bool], size: usize, ink: bool) -> usize {
    let mut seen = vec![false; mask.len()];
    let mut count = 0;
    let mut queue = VecDeque::new();
    for start in 0..mask.len() {
        if seen[start] || mask[start] != ink {
            continue;
        }
        seen[start] = true;
        queue.push_back(start);
        let mut edge = false;
        while let Some(i) = queue.pop_front() {
            let (x, y) = (i % size, i / size);
            edge |= x == 0 || y == 0 || x + 1 == size || y + 1 == size;
            for dy in -1isize..=1 {
                for dx in -1isize..=1 {
                    if (dx == 0 && dy == 0) || (!ink && dx != 0 && dy != 0) {
                        continue;
                    }
                    let (nx, ny) = (x as isize + dx, y as isize + dy);
                    if nx < 0 || ny < 0 || nx >= size as isize || ny >= size as isize {
                        continue;
                    }
                    let j = ny as usize * size + nx as usize;
                    if !seen[j] && mask[j] == ink {
                        seen[j] = true;
                        queue.push_back(j);
                    }
                }
            }
        }
        if ink || !edge {
            count += 1;
        }
    }
    count
}

pub fn measure(pixels: &Pixmap) -> Result<Measurement> {
    ensure!(
        pixels.width() == pixels.height(),
        "measurement raster must be square"
    );
    let size = pixels.width() as usize;
    let mask: Vec<_> = pixels.data().chunks_exact(4).map(|p| p[0] < 128).collect();
    let components = topology(&mask, size, true);
    let holes = topology(&mask, size, false);
    let stable = [96, 160].iter().all(|&threshold| {
        let mask: Vec<_> = pixels
            .data()
            .chunks_exact(4)
            .map(|p| p[0] < threshold)
            .collect();
        topology(&mask, size, true) == components && topology(&mask, size, false) == holes
    });
    Ok(Measurement {
        style: if mask.iter().any(|&p| p) {
            Some(crate::measurements::measure(&mask, size)?)
        } else {
            None
        },
        size,
        pixel_sha256: sha(pixels.data()),
        ink_pixels: mask.iter().filter(|&&p| p).count(),
        components,
        holes,
        threshold_topology_stable: stable,
    })
}

pub struct Verifier;
impl Verifier {
    pub fn validate(&self, glyph: &Glyph) -> Result<(Vec<Measurement>, Pixmap)> {
        let mut measurements = vec![];
        for size in [16, 32, 64] {
            measurements.push(measure(&render(glyph, size)?)?);
        }
        let pixels = render(glyph, 128)?;
        let baseline = measure(&pixels)?;
        ensure!(
            baseline.threshold_topology_stable,
            "128px threshold topology changed"
        );
        ensure!(
            measurements.iter().all(|m| m.ink_pixels > 0
                && m.components == baseline.components
                && m.holes == baseline.holes),
            "small-size topology changed"
        );
        ensure!(baseline.ink_pixels > 0, "blank glyph");
        measurements.push(baseline);
        Ok((measurements, pixels))
    }
}

#[derive(Clone)]
struct Mask {
    rows: [u128; 72],
    area: u32,
}

fn canonical(pixels: &Pixmap) -> Result<Mask> {
    let size = pixels.width() as usize;
    let mask: Vec<_> = pixels.data().chunks_exact(4).map(|p| p[0] < 128).collect();
    let ink: Vec<_> = mask
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| v.then_some((i % size, i / size)))
        .collect();
    ensure!(!ink.is_empty(), "blank distinctness input");
    let x0 = ink.iter().map(|p| p.0).min().unwrap();
    let x1 = ink.iter().map(|p| p.0).max().unwrap() + 1;
    let y0 = ink.iter().map(|p| p.1).min().unwrap();
    let y1 = ink.iter().map(|p| p.1).max().unwrap() + 1;
    let factor = 64.0 / (x1 - x0).max(y1 - y0) as f64;
    let w = (((x1 - x0) as f64 * factor).round_ties_even() as usize).max(1);
    let h = (((y1 - y0) as f64 * factor).round_ties_even() as usize).max(1);
    let mut rows = [0; 72];
    for y in 0..h {
        for x in 0..w {
            let sx = (((x as f64 + 0.5) * (x1 - x0) as f64 / w as f64).floor() as usize)
                .min(x1 - x0 - 1)
                + x0;
            let sy = (((y as f64 + 0.5) * (y1 - y0) as f64 / h as f64).floor() as usize)
                .min(y1 - y0 - 1)
                + y0;
            if mask[sy * size + sx] {
                rows[y + (72 - h) / 2] |= 1u128 << (x + (72 - w) / 2);
            }
        }
    }
    let area = rows.iter().map(|r| r.count_ones()).sum();
    Ok(Mask { rows, area })
}

fn variants(mask: &Mask) -> Vec<Mask> {
    (0..4)
        .map(|flip| Mask {
            rows: std::array::from_fn(|y| {
                let row = mask.rows[if flip & 2 != 0 { 71 - y } else { y }];
                if flip & 1 != 0 {
                    row.reverse_bits() >> 56
                } else {
                    row
                }
            }),
            area: mask.area,
        })
        .collect()
}

fn overlap(left: &Mask, right: &Mask) -> f64 {
    if f64::from(left.area.min(right.area)) / f64::from(left.area.max(right.area)) < 0.85 {
        return 0.0;
    }
    let mut maximum = 0.0f64;
    for dy in -2isize..=2 {
        for dx in -2isize..=2 {
            let mut intersection = 0;
            for y in 2..70usize {
                let row = right.rows[(y as isize - dy) as usize];
                let shifted = if dx >= 0 { row << dx } else { row >> -dx };
                intersection += (left.rows[y] & shifted).count_ones();
            }
            maximum = maximum
                .max(f64::from(intersection) / f64::from(left.area + right.area - intersection));
        }
    }
    maximum
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NearMatch {
    pub left: usize,
    pub right: usize,
    pub iou: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Distinctness {
    pub compared_pairs: usize,
    pub threshold: f64,
    pub near_matches: Vec<NearMatch>,
}

pub fn distinctness(pixels: &[Pixmap], workers: &rayon::ThreadPool) -> Result<Distinctness> {
    let masks = pixels.iter().map(canonical).collect::<Result<Vec<_>>>()?;
    let reflected: Vec<_> = masks.iter().map(variants).collect();
    let matches: Vec<Vec<NearMatch>> = workers.install(|| {
        (0..masks.len())
            .into_par_iter()
            .map(|right| {
                (0..right)
                    .filter_map(|left| {
                        let iou = reflected[right]
                            .iter()
                            .map(|r| overlap(&masks[left], r))
                            .fold(0.0f64, f64::max);
                        (iou >= 0.85).then_some(NearMatch { left, right, iou })
                    })
                    .collect()
            })
            .collect()
    });
    Ok(Distinctness {
        compared_pairs: masks.len() * masks.len().saturating_sub(1) / 2,
        threshold: 0.85,
        near_matches: matches.into_iter().flatten().collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn topology_distinguishes_counters_from_background() {
        let mask = vec![
            false, false, false, false, false, false, true, true, true, false, false, true, false,
            true, false, false, true, true, true, false, false, false, false, false, false,
        ];
        assert_eq!(topology(&mask, 5, true), 1);
        assert_eq!(topology(&mask, 5, false), 1);
    }
    #[test]
    fn reflected_and_translated_duplicates_are_rejected() {
        let g = crate::geometry::compile(0, &crate::geometry::recipes().unwrap()[0]).unwrap();
        let p = render(&g, 128).unwrap();
        let mut reflected = p.clone();
        for y in 0..128usize {
            for x in 0..128usize {
                reflected.data_mut()[(y * 128 + x) * 4..(y * 128 + x + 1) * 4]
                    .copy_from_slice(&p.data()[(y * 128 + 127 - x) * 4..(y * 128 + 128 - x) * 4]);
            }
        }
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        let result = distinctness(&[p, reflected], &pool).unwrap();
        assert_eq!(result.compared_pairs, 1);
        assert_eq!(result.near_matches.len(), 1);
        assert_eq!(result.near_matches[0].iou, 1.0);
    }
}
