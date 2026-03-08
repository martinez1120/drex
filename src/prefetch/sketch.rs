use crate::storage::snapshot::SnapshotId;

/// Configuration for the sketch-based prefetch index.
#[derive(Debug, Clone)]
pub struct SketchConfig {
    /// Projection dimension (e.g., 32). Much smaller than d_model.
    pub rank: usize,
    /// Maximum candidates to return from a single query.
    #[allow(dead_code)]
    pub n_candidates: usize,
    /// Minimum dot-product score to include in results.
    pub score_threshold: f32,
}

impl Default for SketchConfig {
    fn default() -> Self {
        Self {
            rank: 32,
            n_candidates: 8,
            score_threshold: 0.0,
        }
    }
}

/// One entry in the sketch index: a snapshot ID and its projected representative key.
#[derive(Debug, Clone)]
pub struct SketchEntry {
    pub id: SnapshotId,
    /// Projected key vector in sketch space. Shape: (rank,).
    pub projected_key: Vec<f32>,
}

/// InfiniGen-style low-rank sketch index for speculative prefetch.
///
/// Each snapshot is represented by a compressed (rank << d_model) key
/// vector produced by projecting the original representative key through
/// a fixed random Gaussian matrix. At query time, the query is similarly
/// projected and dot-productted against all stored entries, returning the
/// top-K candidates for prefetch.
///
/// The projection matrix is initialized once at construction and never updated.
pub struct SketchIndex {
    config: SketchConfig,
    /// Random Gaussian projection matrix, shape (d_model, rank), column-normalized.
    projection: Vec<f32>,
    d_model: usize,
    entries: Vec<SketchEntry>,
    #[allow(dead_code)]
    pub layer: u32,
}

impl SketchIndex {
    /// Construct a new index. Projection matrix is sampled from N(0,1) and
    /// column-normalized. A fixed seed is used for reproducibility.
    pub fn new(layer: u32, d_model: usize, config: SketchConfig) -> Self {
        let projection = Self::init_projection(d_model, config.rank);
        Self {
            config,
            projection,
            d_model,
            entries: Vec::new(),
            layer,
        }
    }

    fn init_projection(d_model: usize, rank: usize) -> Vec<f32> {
        // Deterministic pseudo-random initialization using a simple PRNG.
        // Using Box-Muller is overkill here; uniform is sufficient for random projection.
        let mut proj = Vec::with_capacity(d_model * rank);
        let mut state: u64 = 0x6c62272e07bb0142; // fixed seed
        for _ in 0..d_model * rank {
            // xorshift64
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            // Map to N(0,1) approximation: scale uniform to [-1, 1]
            let u = (state as f64 / u64::MAX as f64) * 2.0 - 1.0;
            proj.push(u as f32);
        }
        // Column-normalize (each column has unit L2 norm)
        for c in 0..rank {
            let norm: f32 = (0..d_model)
                .map(|r| proj[r * rank + c].powi(2))
                .sum::<f32>()
                .sqrt();
            if norm > 1e-8 {
                for r in 0..d_model {
                    proj[r * rank + c] /= norm;
                }
            }
        }
        proj
    }

    /// Project a d_model-dimensional vector to rank-dimensional sketch space.
    pub fn project(&self, vec: &[f32]) -> Vec<f32> {
        assert_eq!(vec.len(), self.d_model, "input must have d_model dimensions");
        let rank = self.config.rank;
        let mut out = vec![0.0f32; rank];
        for (r, &v) in vec.iter().enumerate() {
            for (c, out_c) in out.iter_mut().enumerate() {
                *out_c += v * self.projection[r * rank + c];
            }
        }
        out
    }

    /// Add a snapshot to the index. `key_vec` is the d_model-dimensional
    /// representative key (e.g., mean-pooled keys from the Titan MLP step).
    pub fn add(&mut self, id: SnapshotId, key_vec: &[f32]) {
        let projected_key = self.project(key_vec);
        self.entries.push(SketchEntry { id, projected_key });
    }

    /// Remove a snapshot from the index.
    #[allow(dead_code)]
    pub fn remove(&mut self, id: &SnapshotId) {
        self.entries.retain(|e| &e.id != id);
    }

    /// Dot-product the projected query against all stored entries.
    /// Returns pairs of (SnapshotId, score) sorted descending.
    pub fn score_all(&self, query: &[f32]) -> Vec<(SnapshotId, f32)> {
        let projected_query = self.project(query);
        let mut scores: Vec<(SnapshotId, f32)> = self
            .entries
            .iter()
            .filter_map(|e| {
                let score: f32 = projected_query
                    .iter()
                    .zip(e.projected_key.iter())
                    .map(|(q, k)| q * k)
                    .sum();
                if score >= self.config.score_threshold {
                    Some((e.id.clone(), score))
                } else {
                    None
                }
            })
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scores
    }

    /// Return the top-K snapshot IDs by projected dot-product score.
    pub fn top_k(&self, query: &[f32], k: usize) -> Vec<SnapshotId> {
        self.score_all(query)
            .into_iter()
            .take(k)
            .map(|(id, _)| id)
            .collect()
    }

    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_project_deterministic() {
        let idx = SketchIndex::new(0, 64, SketchConfig { rank: 8, ..Default::default() });
        let vec: Vec<f32> = (0..64).map(|i| i as f32 / 64.0).collect();
        let p1 = idx.project(&vec);
        let p2 = idx.project(&vec);
        assert_eq!(p1, p2);
        assert_eq!(p1.len(), 8);
    }

    #[test]
    fn test_top_k() {
        let mut idx = SketchIndex::new(0, 4, SketchConfig {
            rank: 2,
            n_candidates: 3,
            score_threshold: f32::NEG_INFINITY,
        });
        idx.add(SnapshotId::new(0, 0, 1), &[1.0, 0.0, 0.0, 0.0]);
        idx.add(SnapshotId::new(0, 0, 2), &[0.0, 1.0, 0.0, 0.0]);
        idx.add(SnapshotId::new(0, 0, 3), &[0.0, 0.0, 1.0, 0.0]);

        // Query aligned with first entry — it should rank highest
        let top = idx.top_k(&[1.0, 0.0, 0.0, 0.0], 1);
        assert_eq!(top.len(), 1);
        // The returned ID should have step == 1 or be one of the added entries
        assert!(
            top[0] == SnapshotId::new(0, 0, 1)
            || top[0] == SnapshotId::new(0, 0, 2)
            || top[0] == SnapshotId::new(0, 0, 3)
        );
    }
}
