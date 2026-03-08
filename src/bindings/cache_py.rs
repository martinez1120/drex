use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use tokio::runtime::Runtime;
use crate::error::DrexError;
use crate::cache::manager::{MemoryTierManager, TierConfig};

fn to_py_err<E: std::fmt::Display>(e: E) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

impl From<DrexError> for PyErr {
    fn from(e: DrexError) -> PyErr {
        PyRuntimeError::new_err(e.to_string())
    }
}

/// Python wrapper for MemoryTierManager.
///
/// Coordinates L3 disk tier operations: demote weights from Python to disk,
/// promote them back, track H2O eviction scores, and run eviction passes.
#[pyclass(name = "MemoryTierManager")]
pub struct PyMemoryTierManager {
    inner: MemoryTierManager,
    rt: Runtime,
}

#[pymethods]
impl PyMemoryTierManager {
    /// Create a new MemoryTierManager.
    ///
    /// Args:
    ///     base_path:            Directory for L3 snapshot storage.
    ///     l3_max_snapshots:     Maximum snapshots before eviction triggers.
    ///     eviction_batch_size:  How many snapshots to evict per pass.
    ///     recent_window:        Training steps before recency decay kicks in.
    ///     compress:             Enable zstd compression for stored snapshots.
    #[new]
    #[pyo3(signature = (
        base_path,
        l3_max_snapshots = 512,
        eviction_batch_size = 32,
        recent_window = 1000,
        compress = false
    ))]
    pub fn new(
        base_path: &str,
        l3_max_snapshots: usize,
        eviction_batch_size: usize,
        recent_window: usize,
        compress: bool,
    ) -> PyResult<Self> {
        let config = TierConfig {
            l3_max_snapshots,
            eviction_batch_size,
            recent_window,
            base_path: base_path.into(),
            compress,
        };
        let inner = MemoryTierManager::new(config).map_err(to_py_err)?;
        let rt = Runtime::new().map_err(to_py_err)?;
        Ok(Self { inner, rt })
    }

    /// Record an attention score for H2O eviction tracking.
    /// Call this once per decode step with the attention weight sum for each head.
    pub fn record_attention_score(
        &self,
        layer: u32,
        head: u32,
        step: u64,
        score: f32,
    ) {
        self.inner.record_attention_score(layer, head, step, score);
    }

    /// Run an eviction pass. Evicts lowest-priority snapshots if L3 is over budget.
    ///
    /// Returns:
    ///     list[tuple[int, int]] — list of (layer, head) pairs that were evicted.
    pub fn run_eviction(&self) -> Result<Vec<(u32, u32)>, DrexError> {
        self.rt.block_on(self.inner.maybe_evict())
    }

    /// Demote weights from Python/L2 to disk/L3.
    ///
    /// Args:
    ///     layer:   Transformer layer index.
    ///     head:    Head index.
    ///     step:    Current training/inference step.
    ///     weights: Flat list of f32 values (TitanMemory.snapshot_weights() output).
    pub fn demote(
        &self,
        layer: u32,
        head: u32,
        step: u64,
        weights: Vec<f32>,
    ) -> Result<(), DrexError> {
        self.rt.block_on(self.inner.demote(layer, head, step, &weights))
    }

    /// Promote weights from disk/L3 back to Python/L2.
    ///
    /// Returns:
    ///     list[float] — the stored weight values.
    pub fn promote(&self, layer: u32, head: u32, step: u64) -> Result<Vec<f32>, DrexError> {
        self.rt.block_on(self.inner.promote(layer, head, step))
    }

    /// Return the number of snapshots currently in L3.
    pub fn snapshot_count(&self) -> usize {
        self.inner.snapshot_count()
    }

    fn __repr__(&self) -> String {
        format!(
            "MemoryTierManager(snapshot_count={})",
            self.inner.snapshot_count()
        )
    }
}
