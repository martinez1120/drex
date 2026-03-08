use dashmap::DashMap;
use std::sync::atomic::{AtomicU64, Ordering};

/// Importance score tracked per (layer, head) pair.
#[derive(Debug, Clone)]
pub struct H2OScore {
    /// Sum of all attention weights this head has received.
    pub cumulative: f64,
    /// Number of times attention was recorded for this head.
    pub hit_count: u64,
    /// Training step at which this entry was last updated.
    pub last_step: u64,
}

/// A candidate for eviction from L3.
#[derive(Debug, Clone)]
pub struct EvictionCandidate {
    pub layer: u32,
    pub head: u32,
    #[allow(dead_code)]
    pub score: H2OScore,
    /// Lower priority = evict first.
    pub priority: f32,
}

/// Trait abstracting eviction policy (H2O or future alternatives).
pub trait EvictionPolicy: Send + Sync {
    fn record_score(&self, layer: u32, head: u32, step: u64, attn_score: f32);
    fn top_candidates(&self, n: usize, current_step: u64) -> Vec<EvictionCandidate>;
    fn remove(&self, layer: u32, head: u32);
    #[allow(dead_code)]
    fn score_of(&self, layer: u32, head: u32) -> Option<H2OScore>;
    #[allow(dead_code)]
    fn len(&self) -> usize;
}

/// H2O (Heavy-Hitter Oracle) eviction policy.
///
/// Priority = cumulative_attention / (hit_count + 1) × recency_decay.
/// Lower priority value → evict first (least useful, or stale).
pub struct H2OEvictionPolicy {
    scores: DashMap<(u32, u32), H2OScore>,
    recent_window: u64,
    current_step: AtomicU64,
}

impl H2OEvictionPolicy {
    pub fn new(recent_window: usize) -> Self {
        Self {
            scores: DashMap::new(),
            recent_window: recent_window as u64,
            current_step: AtomicU64::new(0),
        }
    }

    fn compute_priority(&self, score: &H2OScore, current_step: u64) -> f32 {
        let base = score.cumulative / (score.hit_count as f64 + 1.0);
        let age = current_step.saturating_sub(score.last_step);
        // Exponential recency decay: older = lower priority
        let recency = if age < self.recent_window {
            1.0
        } else {
            (-((age - self.recent_window) as f64) / (self.recent_window as f64)).exp()
        };
        (base * recency) as f32
    }
}

impl EvictionPolicy for H2OEvictionPolicy {
    fn record_score(&self, layer: u32, head: u32, step: u64, attn_score: f32) {
        self.current_step.fetch_max(step, Ordering::Relaxed);
        self.scores
            .entry((layer, head))
            .and_modify(|s| {
                s.cumulative += attn_score as f64;
                s.hit_count += 1;
                s.last_step = step;
            })
            .or_insert(H2OScore {
                cumulative: attn_score as f64,
                hit_count: 1,
                last_step: step,
            });
    }

    fn top_candidates(&self, n: usize, current_step: u64) -> Vec<EvictionCandidate> {
        let mut candidates: Vec<EvictionCandidate> = self
            .scores
            .iter()
            .map(|kv| {
                let (layer, head) = *kv.key();
                let score = kv.value().clone();
                let priority = self.compute_priority(&score, current_step);
                EvictionCandidate { layer, head, score, priority }
            })
            .collect();

        // Sort ascending by priority — lowest priority (most evictable) first
        candidates.sort_by(|a, b| a.priority.partial_cmp(&b.priority).unwrap());
        candidates.truncate(n);
        candidates
    }

    fn remove(&self, layer: u32, head: u32) {
        self.scores.remove(&(layer, head));
    }

    fn score_of(&self, layer: u32, head: u32) -> Option<H2OScore> {
        self.scores.get(&(layer, head)).map(|s| s.clone())
    }

    fn len(&self) -> usize {
        self.scores.len()
    }
}
