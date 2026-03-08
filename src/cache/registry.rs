use dashmap::DashMap;
use crate::storage::snapshot::{SnapshotId, SnapshotMetadata};

/// In-memory index of all snapshots currently on disk.
///
/// Avoids filesystem scans on every lookup during the hot inference path.
/// Must be rebuilt from the SnapshotStore at startup.
pub struct SnapshotRegistry {
    entries: DashMap<SnapshotId, SnapshotMetadata>,
    /// Secondary index: layer -> list of snapshot IDs
    by_layer: DashMap<u32, Vec<SnapshotId>>,
}

impl SnapshotRegistry {
    pub fn new() -> Self {
        Self {
            entries: DashMap::new(),
            by_layer: DashMap::new(),
        }
    }

    /// Register a new snapshot.
    pub fn register(&self, meta: SnapshotMetadata) {
        let id = meta.id.clone();
        let layer = id.layer;
        self.entries.insert(id.clone(), meta);
        self.by_layer
            .entry(layer)
            .and_modify(|v| {
                if !v.contains(&id) {
                    v.push(id.clone());
                }
            })
            .or_insert_with(|| vec![id]);
    }

    #[allow(dead_code)]
    /// Look up metadata for a snapshot by ID.
    pub fn lookup(&self, id: &SnapshotId) -> Option<SnapshotMetadata> {
        self.entries.get(id).map(|m| m.clone())
    }

    /// All snapshots for a given layer, in insertion order.
    pub fn all_for_layer(&self, layer: u32) -> Vec<SnapshotMetadata> {
        self.by_layer
            .get(&layer)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.entries.get(id).map(|m| m.clone()))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Remove a snapshot from the index.
    pub fn remove(&self, id: &SnapshotId) {
        self.entries.remove(id);
        if let Some(mut ids) = self.by_layer.get_mut(&id.layer) {
            ids.retain(|i| i != id);
        }
    }

    pub fn count(&self) -> usize {
        self.entries.len()
    }
    #[allow(dead_code)]
    pub fn contains(&self, id: &SnapshotId) -> bool {
        self.entries.contains_key(id)
    }
}

impl Default for SnapshotRegistry {
    fn default() -> Self {
        Self::new()
    }
}
