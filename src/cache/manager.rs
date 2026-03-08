use std::sync::Arc;
use std::path::PathBuf;
use crate::error::DrexError;
use crate::storage::snapshot::{SnapshotId, SnapshotStore};
use crate::storage::journal::{WriteAheadJournal, JournalEntry, JournalOp};
use crate::cache::eviction::{H2OEvictionPolicy, EvictionPolicy};
use crate::cache::registry::SnapshotRegistry;

/// Configuration for the memory tier manager.
#[derive(Debug, Clone)]
pub struct TierConfig {
    pub l3_max_snapshots: usize,
    pub eviction_batch_size: usize,
    pub recent_window: usize,
    pub base_path: PathBuf,
    pub compress: bool,
}

impl Default for TierConfig {
    fn default() -> Self {
        Self {
            l3_max_snapshots: 512,
            eviction_batch_size: 32,
            recent_window: 1000,
            base_path: PathBuf::from("/tmp/drex_l3"),
            compress: false,
        }
    }
}

/// Coordinates all L3 disk-tier operations: storage, registry, eviction, journal.
///
/// This is the primary entry point called from Python via the PyO3 binding.
pub struct MemoryTierManager {
    config: TierConfig,
    store: Arc<SnapshotStore>,
    registry: Arc<SnapshotRegistry>,
    eviction: Arc<H2OEvictionPolicy>,
    journal: Arc<WriteAheadJournal>,
    current_step: std::sync::atomic::AtomicU64,
}

impl MemoryTierManager {
    pub fn new(config: TierConfig) -> Result<Self, DrexError> {
        let store = Arc::new(
            SnapshotStore::new(&config.base_path, config.compress)
                .map_err(DrexError::Storage)?,
        );
        let registry = Arc::new(SnapshotRegistry::new());
        let eviction = Arc::new(H2OEvictionPolicy::new(config.recent_window));
        let journal = Arc::new(
            WriteAheadJournal::new(&config.base_path).map_err(DrexError::Storage)?,
        );
        Ok(Self {
            config,
            store,
            registry,
            eviction,
            journal,
            current_step: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// Record an attention score for H2O eviction tracking.
    pub fn record_attention_score(&self, layer: u32, head: u32, step: u64, score: f32) {
        self.current_step
            .fetch_max(step, std::sync::atomic::Ordering::Relaxed);
        self.eviction.record_score(layer, head, step, score);
    }

    /// Run an eviction pass if L3 is over budget.
    /// Returns the (layer, head) pairs that were evicted.
    pub async fn maybe_evict(&self) -> Result<Vec<(u32, u32)>, DrexError> {
        let count = self.registry.count();
        if count <= self.config.l3_max_snapshots {
            return Ok(Vec::new());
        }

        let current_step = self.current_step.load(std::sync::atomic::Ordering::Relaxed);
        let to_evict = self.eviction.top_candidates(
            self.config.eviction_batch_size,
            current_step,
        );

        let mut evicted = Vec::new();
        for candidate in to_evict {
            // Find all snapshots for this (layer, head) and delete all
            let metas = self.registry.all_for_layer(candidate.layer);
            for meta in metas {
                if meta.id.head == candidate.head {
                    self.store.delete(&meta.id).await?;
                    self.registry.remove(&meta.id);
                    self.journal
                        .append(&JournalEntry {
                            op: JournalOp::Delete,
                            id: meta.id.clone(),
                            timestamp_secs: std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_secs(),
                        })
                        .await
                        .map_err(DrexError::Storage)?;
                }
            }
            self.eviction.remove(candidate.layer, candidate.head);
            evicted.push((candidate.layer, candidate.head));
        }

        Ok(evicted)
    }

    /// Demote weights from Python (L2) to disk (L3).
    pub async fn demote(
        &self,
        layer: u32,
        head: u32,
        step: u64,
        weights: &[f32],
    ) -> Result<(), DrexError> {
        let id = SnapshotId::new(layer, head, step);
        let meta = self.store.write(id.clone(), weights).await?;
        self.registry.register(meta);
        self.journal
            .append(&JournalEntry {
                op: JournalOp::Write,
                id: id.clone(),
                timestamp_secs: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            })
            .await
            .map_err(DrexError::Storage)?;
        Ok(())
    }

    /// Promote weights from disk (L3) back to Python (L2/active).
    pub async fn promote(
        &self,
        layer: u32,
        head: u32,
        step: u64,
    ) -> Result<Vec<f32>, DrexError> {
        let id = SnapshotId::new(layer, head, step);
        let weights = self.store.read(&id).await?;
        Ok(weights)
    }

    pub fn snapshot_count(&self) -> usize {
        self.registry.count()
    }
    #[allow(dead_code)]
    pub fn store(&self) -> Arc<SnapshotStore> {
        Arc::clone(&self.store)
    }
}
