use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use crate::error::StorageError;
use crate::storage::snapshot::{SnapshotStore, SnapshotId};
use tokio::runtime::Runtime;
use std::sync::Arc;

fn to_py_err<E: std::fmt::Display>(e: E) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

impl From<StorageError> for PyErr {
    fn from(e: StorageError) -> PyErr {
        PyRuntimeError::new_err(e.to_string())
    }
}

/// Metadata about a stored snapshot, accessible from Python.
#[pyclass(name = "SnapshotMeta")]
#[derive(Clone)]
pub struct PySnapshotMeta {
    #[pyo3(get)]
    pub layer: u32,
    #[pyo3(get)]
    pub head: u32,
    #[pyo3(get)]
    pub step: u64,
    #[pyo3(get)]
    pub n_weights: usize,
    #[pyo3(get)]
    pub size_bytes: u64,
    #[pyo3(get)]
    pub compressed: bool,
}

#[pymethods]
impl PySnapshotMeta {
    fn __repr__(&self) -> String {
        format!(
            "SnapshotMeta(layer={}, head={}, step={}, n_weights={}, size_bytes={}, compressed={})",
            self.layer, self.head, self.step, self.n_weights, self.size_bytes, self.compressed
        )
    }
}

/// Python wrapper for SnapshotStore.
///
/// Provides synchronous (blocking) methods that call into async Rust via
/// a dedicated tokio runtime. All methods release the GIL during I/O.
#[pyclass(name = "SnapshotStore")]
pub struct PySnapshotStore {
    pub(crate) inner: Arc<SnapshotStore>,
    pub(crate) rt: Runtime,
}

#[pymethods]
impl PySnapshotStore {
    /// Create a new SnapshotStore at `base_path`.
    ///
    /// Args:
    ///     base_path: Directory where snapshots are stored (created if absent).
    ///     compress:  If True, use zstd compression (slower writes, smaller files).
    #[new]
    #[pyo3(signature = (base_path, compress = false))]
    pub fn new(base_path: &str, compress: bool) -> PyResult<Self> {
        let store = SnapshotStore::new(base_path, compress).map_err(to_py_err)?;
        let rt = Runtime::new().map_err(to_py_err)?;
        Ok(Self {
            inner: Arc::new(store),
            rt,
        })
    }

    /// Write weights to disk.
    ///
    /// Args:
    ///     layer:   Transformer layer index.
    ///     head:    Head index (use 0 for whole-layer Titan snapshots).
    ///     step:    Training/inference step (used to version snapshots).
    ///     weights: Flat list of f32 values.
    pub fn write(
        &self,
        layer: u32,
        head: u32,
        step: u64,
        weights: Vec<f32>,
    ) -> Result<(), StorageError> {
        let id = SnapshotId::new(layer, head, step);
        self.rt
            .block_on(self.inner.write(id, &weights))
            .map(|_| ())
    }

    /// Read weights from disk.
    ///
    /// Returns:
    ///     list[float] — the stored weight values.
    ///
    /// Raises:
    ///     RuntimeError: If the snapshot does not exist or checksum fails.
    pub fn read(&self, layer: u32, head: u32, step: u64) -> Result<Vec<f32>, StorageError> {
        let id = SnapshotId::new(layer, head, step);
        self.rt.block_on(self.inner.read(&id))
    }

    /// Delete a snapshot.
    pub fn delete(&self, layer: u32, head: u32, step: u64) -> Result<(), StorageError> {
        let id = SnapshotId::new(layer, head, step);
        self.rt.block_on(self.inner.delete(&id))
    }

    /// Return True if the snapshot exists on disk.
    pub fn exists(&self, layer: u32, head: u32, step: u64) -> bool {
        let id = SnapshotId::new(layer, head, step);
        self.rt.block_on(self.inner.exists(&id))
    }

    /// List all snapshots in the store.
    pub fn list_snapshots(&self) -> Result<Vec<PySnapshotMeta>, StorageError> {
        let metas = self.rt.block_on(self.inner.list_all())?;
        Ok(metas
            .into_iter()
            .map(|m| PySnapshotMeta {
                layer: m.id.layer,
                head: m.id.head,
                step: m.id.step,
                n_weights: m.n_weights,
                size_bytes: m.size_bytes,
                compressed: m.compressed,
            })
            .collect())
    }

    /// List all snapshots for a given layer.
    pub fn list_by_layer(&self, layer: u32) -> Result<Vec<PySnapshotMeta>, StorageError> {
        let metas = self.rt.block_on(self.inner.list_by_layer(layer))?;
        Ok(metas
            .into_iter()
            .map(|m| PySnapshotMeta {
                layer: m.id.layer,
                head: m.id.head,
                step: m.id.step,
                n_weights: m.n_weights,
                size_bytes: m.size_bytes,
                compressed: m.compressed,
            })
            .collect())
    }

    /// Read weights from disk using memory-mapped I/O.
    ///
    /// For uncompressed stores this avoids the intermediate ``Vec<u8>``
    /// allocation that :py:meth:`read` incurs, mapping the file pages
    /// directly into the process address space before parsing the f32 values.
    /// For compressed stores it transparently falls back to :py:meth:`read`.
    ///
    /// Returns:
    ///     list[float] — the stored weight values.
    ///
    /// Raises:
    ///     RuntimeError: If the snapshot does not exist or checksum fails.
    pub fn read_mmap(&self, layer: u32, head: u32, step: u64) -> Result<Vec<f32>, StorageError> {
        let id = SnapshotId::new(layer, head, step);
        self.rt.block_on(self.inner.read_mmap(&id))
    }

    fn __repr__(&self) -> String {
        format!("SnapshotStore(base_path='{}')", self.inner.base_path().display())
    }
}
