use pyo3::prelude::*;
use crate::prefetch::engine::PrefetchEngine;
use crate::prefetch::sketch::SketchConfig;
use crate::storage::snapshot::SnapshotId;
use crate::bindings::storage_py::PySnapshotStore;
use std::sync::Arc;

/// Python wrapper for PrefetchEngine.
///
/// Registers snapshot key vectors in a low-rank sketch index. On each decode
/// step, call `prefetch(layer, query_vec, k)` to fire background async tasks
/// that pre-load likely-needed snapshots into an in-memory cache. Retrieve
/// with `consume_prefetched` which returns None on a cache miss.
#[pyclass(name = "PrefetchEngine")]
pub struct PyPrefetchEngine {
    inner: PrefetchEngine,
}

#[pymethods]
impl PyPrefetchEngine {
    /// Create a new PrefetchEngine.
    ///
    /// Args:
    ///     store:              A SnapshotStore instance (shares the same base_path).
    ///     d_model:            Model hidden dimension (for projection matrix sizing).
    ///     max_cache_entries:  Maximum number of prefetched snapshots kept in memory.
    ///     sketch_rank:        Projection rank for the sketch index (default 32).
    #[new]
    #[pyo3(signature = (store, d_model, max_cache_entries = 128, sketch_rank = 32))]
    pub fn new(
        store: &PySnapshotStore,
        d_model: usize,
        max_cache_entries: usize,
        sketch_rank: usize,
    ) -> PyResult<Self> {
        let config = SketchConfig {
            rank: sketch_rank,
            n_candidates: max_cache_entries,
            score_threshold: 0.0,
        };
        let engine = PrefetchEngine::new(
            Arc::clone(&store.inner),
            d_model,
            max_cache_entries,
            config,
        );
        Ok(Self { inner: engine })
    }

    /// Register a snapshot in the sketch index after it has been demoted to disk.
    ///
    /// Args:
    ///     layer:    Transformer layer index.
    ///     head:     Head index.
    ///     step:     Training/inference step.
    ///     key_vec:  Representative key vector (length d_model), e.g. mean-pooled
    ///               keys from the Titan write step.
    pub fn register_snapshot(
        &mut self,
        layer: u32,
        head: u32,
        step: u64,
        key_vec: Vec<f32>,
    ) {
        let id = SnapshotId::new(layer, head, step);
        self.inner.register_snapshot(layer, id, key_vec);
    }

    /// Speculatively prefetch the top-k snapshots most likely needed next.
    ///
    /// This call returns immediately; loading happens on background threads.
    /// Check results with `consume_prefetched`.
    ///
    /// Args:
    ///     layer:      Transformer layer index.
    ///     query_vec:  Current query vector (length d_model).
    ///     k:          Number of candidates to prefetch.
    pub fn prefetch(&self, layer: u32, query_vec: Vec<f32>, k: usize) {
        self.inner.prefetch(layer, query_vec, k);
    }

    /// Consume a prefetched snapshot from the in-memory cache.
    ///
    /// Returns:
    ///     list[float] if the snapshot was prefetched and is ready, else None.
    ///     The snapshot is removed from the cache on retrieval.
    pub fn consume_prefetched(
        &self,
        layer: u32,
        head: u32,
        step: u64,
    ) -> Option<Vec<f32>> {
        let id = SnapshotId::new(layer, head, step);
        self.inner.consume_prefetched(&id)
    }

    /// Clear all prefetch state (sketch indices and in-memory cache).
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    fn __repr__(&self) -> String {
        "PrefetchEngine()".to_string()
    }
}
