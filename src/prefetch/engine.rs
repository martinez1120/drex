use std::collections::HashMap;
use std::sync::Arc;
use dashmap::DashMap;
use crate::storage::snapshot::{SnapshotId, SnapshotStore};
use crate::prefetch::sketch::{SketchIndex, SketchConfig};

/// Async speculative prefetch engine.
///
/// After each decode step, call `prefetch(layer, query_vec, k)` to fire
/// background tokio tasks that load likely-needed snapshots into an in-memory
/// cache. On the actual retrieve, `consume_prefetched` returns the weights
/// instantly if the prefetch landed.
pub struct PrefetchEngine {
    /// Per-layer sketch indices.
    indices: HashMap<u32, SketchIndex>,
    store: Arc<SnapshotStore>,
    /// Hot prefetch cache: snapshot ID -> loaded weights.
    prefetch_cache: Arc<DashMap<SnapshotId, Vec<f32>>>,
    max_cache_entries: usize,
    d_model: usize,
    sketch_config: SketchConfig,
    rt: tokio::runtime::Runtime,
}

impl PrefetchEngine {
    pub fn new(
        store: Arc<SnapshotStore>,
        d_model: usize,
        max_cache_entries: usize,
        sketch_config: SketchConfig,
    ) -> Self {
        Self {
            indices: HashMap::new(),
            store,
            prefetch_cache: Arc::new(DashMap::new()),
            max_cache_entries,
            d_model,
            sketch_config,
            rt: tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .thread_name("drex-prefetch")
                .enable_all()
                .build()
                .expect("failed to build prefetch tokio runtime"),
        }
    }

    /// Register a snapshot in this layer's sketch index.
    /// Called after a successful `demote` from the manager.
    pub fn register_snapshot(
        &mut self,
        layer: u32,
        id: SnapshotId,
        key_vec: Vec<f32>,
    ) {
        let d_model = self.d_model;
        let config = self.sketch_config.clone();
        let idx = self
            .indices
            .entry(layer)
            .or_insert_with(|| SketchIndex::new(layer, d_model, config));
        idx.add(id, &key_vec);
    }

    #[allow(dead_code)]
    /// Remove a snapshot from its layer's sketch index.
    pub fn deregister_snapshot(&mut self, id: &SnapshotId) {
        if let Some(idx) = self.indices.get_mut(&id.layer) {
            idx.remove(id);
        }
        self.prefetch_cache.remove(id);
    }

    /// Fire-and-forget: score candidates and spawn async tasks to load top-k.
    pub fn prefetch(&self, layer: u32, query_vec: Vec<f32>, k: usize) {
        let candidates = match self.indices.get(&layer) {
            Some(idx) => idx.top_k(&query_vec, k),
            None => return,
        };

        for id in candidates {
            if self.prefetch_cache.contains_key(&id) {
                continue; // already cached
            }
            let store = Arc::clone(&self.store);
            let cache = Arc::clone(&self.prefetch_cache);
            let id_clone = id.clone();
            self.rt.spawn(async move {
                if let Ok(weights) = store.read(&id_clone).await {
                    cache.insert(id_clone, weights);
                }
            });
        }

        // Trim cache if over budget
        self.trim_cache();
    }

    /// Check if a snapshot was prefetched. Removes it from cache (consume semantics).
    pub fn consume_prefetched(&self, id: &SnapshotId) -> Option<Vec<f32>> {
        self.prefetch_cache.remove(id).map(|(_, v)| v)
    }

    /// Evict oldest entries from the prefetch cache when over budget.
    /// (Simple FIFO approximation — DashMap iteration order is not guaranteed,
    /// but this is best-effort for a cache.)
    pub fn trim_cache(&self) {
        while self.prefetch_cache.len() > self.max_cache_entries {
            if let Some(entry) = self.prefetch_cache.iter().next() {
                let key = entry.key().clone();
                drop(entry);
                self.prefetch_cache.remove(&key);
            } else {
                break;
            }
        }
    }

    /// Clear all prefetch state.
    pub fn clear(&mut self) {
        self.prefetch_cache.clear();
        self.indices.clear();
    }
}
