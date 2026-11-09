//! Native raster measurements. Distances are between pixel centres; outside is background.
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

pub const METHOD: &str = "noumenon-native-style-v1";
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Summary {
    pub min: f64,
    pub p10: f64,
    pub median: f64,
    pub p90: f64,
    pub max: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Style {
    pub method: String,
    pub bbox: [usize; 4],
    pub ink_fraction: f64,
    pub bbox_ink_fraction: f64,
    pub component_areas: Vec<usize>,
    pub counter_areas: Vec<usize>,
    pub stroke_width: Summary,
    pub skeleton_pixels: usize,
    pub orientation_histogram_15deg: [usize; 12],
    pub component_gap_min: Option<f64>,
    pub counter_width_median: Option<f64>,
    pub horizontal_mirror_iou: f64,
    pub vertical_mirror_iou: f64,
}
fn rounded(n: f64) -> f64 {
    (n * 1e6).round() / 1e6
}
fn percentile(values: &[f64], p: f64) -> f64 {
    let mut ordered = values.to_vec();
    ordered.sort_by(f64::total_cmp);
    let rank = (ordered.len() - 1) as f64 * p;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    rounded(ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo as f64))
}
fn summary(values: &[f64]) -> Summary {
    Summary {
        min: percentile(values, 0.0),
        p10: percentile(values, 0.1),
        median: percentile(values, 0.5),
        p90: percentile(values, 0.9),
        max: percentile(values, 1.0),
    }
}
fn regions(mask: &[bool], n: usize, ink: bool, eight: bool) -> Vec<Vec<usize>> {
    let mut seen = vec![false; mask.len()];
    let mut result = vec![];
    for start in 0..mask.len() {
        if seen[start] || mask[start] != ink {
            continue;
        }
        let mut queue = VecDeque::from([start]);
        let mut region = vec![];
        let mut edge = false;
        seen[start] = true;
        while let Some(i) = queue.pop_front() {
            region.push(i);
            let (x, y) = (i % n, i / n);
            edge |= x == 0 || y == 0 || x + 1 == n || y + 1 == n;
            for dy in -1isize..=1 {
                for dx in -1isize..=1 {
                    if (dx == 0 && dy == 0) || (!eight && dx != 0 && dy != 0) {
                        continue;
                    }
                    let (a, b) = (x as isize + dx, y as isize + dy);
                    if a < 0 || b < 0 || a >= n as isize || b >= n as isize {
                        continue;
                    }
                    let j = b as usize * n + a as usize;
                    if !seen[j] && mask[j] == ink {
                        seen[j] = true;
                        queue.push_back(j);
                    }
                }
            }
        }
        if ink || !edge {
            result.push(region);
        }
    }
    result
}

// Lower envelope of parabolas: exact squared Euclidean transform in linear time.
fn transform(f: &[f64]) -> Vec<f64> {
    let n = f.len();
    let mut vertices = vec![0usize; n];
    let mut boundaries = vec![0.0; n + 1];
    boundaries[0] = f64::NEG_INFINITY;
    boundaries[1] = f64::INFINITY;
    let mut k = 0;
    for q in 1..n {
        let mut s;
        loop {
            let v = vertices[k];
            s = (f[q] + (q * q) as f64 - f[v] - (v * v) as f64) / (2.0 * (q - v) as f64);
            if s > boundaries[k] {
                break;
            }
            k -= 1;
        }
        k += 1;
        vertices[k] = q;
        boundaries[k] = s;
        boundaries[k + 1] = f64::INFINITY;
    }
    k = 0;
    (0..n)
        .map(|q| {
            while boundaries[k + 1] < (q as f64) {
                k += 1;
            }
            (q as f64 - vertices[k] as f64).powi(2) + f[vertices[k]]
        })
        .collect()
}
fn distance(mask: &[bool], n: usize) -> Vec<f64> {
    distance_with_boundary(mask, n, true)
}
fn distance_with_boundary(mask: &[bool], n: usize, outside_background: bool) -> Vec<f64> {
    let width = n + 2;
    let mut grid = vec![if outside_background { 0.0 } else { 1e9 }; width * width];
    for y in 0..n {
        for x in 0..n {
            grid[(y + 1) * width + x + 1] = if mask[y * n + x] { 1e9 } else { 0.0 };
        }
    }
    for y in 0..width {
        let row = transform(&grid[y * width..(y + 1) * width]);
        grid[y * width..(y + 1) * width].copy_from_slice(&row);
    }
    for x in 0..width {
        let col = transform(&(0..width).map(|y| grid[y * width + x]).collect::<Vec<_>>());
        for y in 0..width {
            grid[y * width + x] = col[y];
        }
    }
    (0..n * n)
        .map(|i| grid[(i / n + 1) * width + i % n + 1].sqrt())
        .collect()
}
fn thin(mask: &[bool], n: usize) -> Vec<bool> {
    let mut result = mask.to_vec();
    loop {
        let mut changed = false;
        for step in 0..2 {
            let mut remove = vec![];
            for i in 0..n * n {
                if !result[i] {
                    continue;
                }
                let x = (i % n) as isize;
                let y = (i / n) as isize;
                let p: [bool; 8] = std::array::from_fn(|j| {
                    let (dx, dy) = [
                        (0, -1),
                        (1, -1),
                        (1, 0),
                        (1, 1),
                        (0, 1),
                        (-1, 1),
                        (-1, 0),
                        (-1, -1),
                    ][j];
                    let (a, b) = (x + dx, y + dy);
                    a >= 0
                        && b >= 0
                        && a < n as isize
                        && b < n as isize
                        && result[b as usize * n + a as usize]
                });
                let count = p.iter().filter(|&&v| v).count();
                let transitions = (0..8).filter(|&j| !p[j] && p[(j + 1) % 8]).count();
                let guard = if step == 0 {
                    !p[2] || !p[4] || (!p[0] && !p[6])
                } else {
                    !p[0] || !p[6] || (!p[2] && !p[4])
                };
                if (2..=6).contains(&count) && transitions == 1 && guard {
                    remove.push(i);
                }
            }
            changed |= !remove.is_empty();
            for i in remove {
                result[i] = false;
            }
        }
        if !changed {
            break;
        }
    }
    result
}
fn medial(mask: &[bool], n: usize, eight: bool) -> (Vec<bool>, Vec<f64>) {
    let mut skeleton = thin(mask, n);
    let distances = distance(mask, n);
    for region in regions(mask, n, true, eight) {
        if !region.iter().any(|&i| skeleton[i]) {
            let i = *region
                .iter()
                .max_by(|&&a, &&b| distances[a].total_cmp(&distances[b]))
                .unwrap();
            skeleton[i] = true;
        }
    }
    (skeleton, distances)
}

pub fn measure(mask: &[bool], n: usize) -> Result<Style> {
    ensure!(
        (1..=512).contains(&n) && mask.len() == n * n,
        "invalid measurement mask"
    );
    let ink: Vec<_> = (0..n * n).filter(|&i| mask[i]).collect();
    ensure!(!ink.is_empty(), "blank measurement mask");
    let x0 = ink.iter().map(|i| i % n).min().unwrap();
    let x1 = ink.iter().map(|i| i % n + 1).max().unwrap();
    let y0 = ink.iter().map(|i| i / n).min().unwrap();
    let y1 = ink.iter().map(|i| i / n + 1).max().unwrap();
    let components = regions(mask, n, true, true);
    let counters = regions(mask, n, false, false);
    let (skeleton, distances) = medial(mask, n, true);
    let points: Vec<_> = (0..n * n).filter(|&i| skeleton[i]).collect();
    let widths: Vec<_> = points.iter().map(|&i| 2.0 * distances[i]).collect();
    let mut histogram = [0usize; 12];
    for &i in &points {
        let (x, y) = ((i % n) as isize, (i / n) as isize);
        let mut local = vec![];
        for dy in -5isize..=5 {
            for dx in -5isize..=5 {
                let (a, b) = (x + dx, y + dy);
                if dx * dx + dy * dy <= 25
                    && a >= 0
                    && b >= 0
                    && a < n as isize
                    && b < n as isize
                    && skeleton[b as usize * n + a as usize]
                {
                    local.push((a as f64, b as f64));
                }
            }
        }
        if local.len() < 4 {
            continue;
        }
        let count = local.len() as f64;
        let mean = (
            local.iter().map(|p| p.0).sum::<f64>() / count,
            local.iter().map(|p| p.1).sum::<f64>() / count,
        );
        let xx = local.iter().map(|p| (p.0 - mean.0).powi(2)).sum::<f64>() / (count - 1.0);
        let yy = local.iter().map(|p| (p.1 - mean.1).powi(2)).sum::<f64>() / (count - 1.0);
        let xy = local
            .iter()
            .map(|p| (p.0 - mean.0) * (p.1 - mean.1))
            .sum::<f64>()
            / (count - 1.0);
        let delta = ((xx - yy).powi(2) + 4.0 * xy * xy).sqrt();
        let major = (xx + yy + delta) / 2.0;
        let minor = (xx + yy - delta) / 2.0;
        if major < 1.0 || major < 4.0 * minor.max(1e-8) {
            continue;
        }
        let angle = (0.5 * (2.0 * xy).atan2(xx - yy))
            .to_degrees()
            .rem_euclid(180.0);
        histogram[((angle / 15.0).floor() as usize).min(11)] += 1;
    }
    let mut gap: f64 = f64::INFINITY;
    ensure!(
        components.len() <= 256,
        "measurement component limit exceeded"
    );
    for a in 0..components.len().saturating_sub(1) {
        let mut other = vec![true; n * n];
        for &i in &components[a] {
            other[i] = false;
        }
        let d = distance_with_boundary(&other, n, false);
        for region in &components[a + 1..] {
            for &i in region {
                gap = gap.min(d[i]);
            }
        }
    }
    let counter_width_median = if counters.is_empty() {
        None
    } else {
        let mut holes = vec![false; n * n];
        for region in &counters {
            for &i in region {
                holes[i] = true;
            }
        }
        let (s, d) = medial(&holes, n, false);
        Some(percentile(
            &(0..n * n)
                .filter(|&i| s[i])
                .map(|i| 2.0 * d[i])
                .collect::<Vec<_>>(),
            0.5,
        ))
    };
    let mirror = |horizontal: bool| {
        let mut intersection = 0;
        let mut union = 0;
        for y in y0..y1 {
            for x in x0..x1 {
                let a = mask[y * n + x];
                let b = if horizontal {
                    mask[y * n + x0 + x1 - 1 - x]
                } else {
                    mask[(y0 + y1 - 1 - y) * n + x]
                };
                intersection += usize::from(a && b);
                union += usize::from(a || b);
            }
        }
        rounded(intersection as f64 / union as f64)
    };
    let areas = |regions: Vec<Vec<usize>>| {
        let mut sizes: Vec<_> = regions.into_iter().map(|r| r.len()).collect();
        sizes.sort_unstable_by(|a, b| b.cmp(a));
        sizes
    };
    Ok(Style {
        method: METHOD.into(),
        bbox: [x0, y0, x1, y1],
        ink_fraction: rounded(ink.len() as f64 / (n * n) as f64),
        bbox_ink_fraction: rounded(ink.len() as f64 / ((x1 - x0) * (y1 - y0)) as f64),
        component_areas: areas(components),
        counter_areas: areas(counters),
        stroke_width: summary(&widths),
        skeleton_pixels: points.len(),
        orientation_histogram_15deg: histogram,
        component_gap_min: gap.is_finite().then(|| rounded(gap)),
        counter_width_median,
        horizontal_mirror_iou: mirror(true),
        vertical_mirror_iou: mirror(false),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn distance_transform_matches_exhaustive_euclidean_distances() {
        let n = 9;
        let mask: Vec<_> = (0..n * n).map(|i| i % 7 != 0).collect();
        let d = distance(&mask, n);
        for (i, &actual) in d.iter().enumerate() {
            let (x, y) = ((i % n) as isize, (i / n) as isize);
            let mut expected = f64::INFINITY;
            for by in -1..=n as isize {
                for bx in -1..=n as isize {
                    if bx == -1
                        || by == -1
                        || bx == n as isize
                        || by == n as isize
                        || !mask[by as usize * n + bx as usize]
                    {
                        expected = expected.min(((x - bx) as f64).hypot((y - by) as f64));
                    }
                }
            }
            assert!((actual - expected).abs() < 1e-12);
        }
    }
    #[test]
    fn lines_rings_and_separated_points_have_known_measurements() {
        let n = 15;
        let mut line = vec![false; n * n];
        for x in 3..12 {
            line[7 * n + x] = true;
        }
        let m = measure(&line, n).unwrap();
        assert_eq!(m.component_areas, vec![9]);
        assert_eq!(m.stroke_width.median, 2.0);
        assert_eq!(m.horizontal_mirror_iou, 1.0);
        let mut ring = vec![false; n * n];
        for y in 4..11 {
            for x in 4..11 {
                ring[y * n + x] = x == 4 || x == 10 || y == 4 || y == 10;
            }
        }
        assert_eq!(measure(&ring, n).unwrap().counter_areas, vec![25]);
        let mut points = vec![false; n * n];
        points[2 * n + 2] = true;
        points[2 * n + 6] = true;
        assert_eq!(measure(&points, n).unwrap().component_gap_min, Some(4.0));
        assert!(measure(&vec![false; n * n], n).is_err());
    }
    #[test]
    fn distance_field_and_ink_are_reflection_invariant() {
        let n = 11;
        let mask: Vec<_> = (0..n * n)
            .map(|i| i / n > 2 && i / n < 8 && i % n > 3 && i % n < 7)
            .collect();
        let flipped: Vec<_> = (0..n * n)
            .map(|i| mask[(i / n) * n + n - 1 - i % n])
            .collect();
        let left = distance(&mask, n);
        let right = distance(&flipped, n);
        for i in 0..n * n {
            assert_eq!(left[i], right[(i / n) * n + n - 1 - i % n]);
        }
        let a = measure(&mask, n).unwrap();
        let b = measure(&flipped, n).unwrap();
        assert_eq!(a.ink_fraction, b.ink_fraction);
        assert_eq!(a.component_areas, b.component_areas);
    }
}
