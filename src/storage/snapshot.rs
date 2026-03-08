use std::path::{Path, PathBuf};
use crate::error::StorageError;

/// Uniquely identifies one snapshot (Titans MLP weight dump) on disk.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct SnapshotId {
    pub layer: u32,
    pub head: u32,
    pub step: u64,
}

impl SnapshotId {
    pub fn new(layer: u32, head: u32, step: u64) -> Self {
        Self { layer, head, step }
    }

    /// Canonical filename: `l{layer}_h{head}_s{step}.snap`
    pub fn to_filename(&self) -> String {
        format!("l{}_h{}_s{}.snap", self.layer, self.head, self.step)
    }

    #[allow(dead_code)]
    /// Parse from a filename produced by `to_filename()`.
    pub fn from_filename(name: &str) -> Option<Self> {
        let name = name.strip_suffix(".snap")?;
        let parts: Vec<&str> = name.split('_').collect();
        if parts.len() != 3 {
            return None;
        }
        let layer = parts[0].strip_prefix('l')?.parse().ok()?;
        let head = parts[1].strip_prefix('h')?.parse().ok()?;
        let step = parts[2].strip_prefix('s')?.parse().ok()?;
        Some(Self { layer, head, step })
    }

    pub fn meta_filename(&self) -> String {
        format!("l{}_h{}_s{}.meta", self.layer, self.head, self.step)
    }
}

/// Metadata stored in a sidecar `.meta` file alongside each snapshot.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SnapshotMetadata {
    pub id: SnapshotId,
    pub n_weights: usize,
    pub size_bytes: u64,
    pub created_at_secs: u64,
    pub checksum: u32,
    pub compressed: bool,
}

/// Magic header bytes written at the start of every `.snap` file.
/// Allows basic corruption detection.
const MAGIC: &[u8; 8] = b"DREX\x01\x00\x00\x00";

/// Serialize/deserialize a `Vec<f32>` to/from bytes.
pub trait SnapshotSerializer: Send + Sync {
    fn serialize(&self, weights: &[f32]) -> Result<Vec<u8>, StorageError>;
    fn deserialize(&self, data: &[u8]) -> Result<Vec<f32>, StorageError>;
    fn is_compressed(&self) -> bool;
}

/// Raw little-endian f32 bytes preceded by the magic header. Fast.
pub struct FlatfileSerializer;

impl SnapshotSerializer for FlatfileSerializer {
    fn serialize(&self, weights: &[f32]) -> Result<Vec<u8>, StorageError> {
        let mut buf = Vec::with_capacity(8 + weights.len() * 4);
        buf.extend_from_slice(MAGIC);
        for &w in weights {
            buf.extend_from_slice(&w.to_le_bytes());
        }
        Ok(buf)
    }

    fn deserialize(&self, data: &[u8]) -> Result<Vec<f32>, StorageError> {
        if data.len() < 8 || &data[..8] != MAGIC {
            return Err(StorageError::Corrupt("invalid magic header".into()));
        }
        let payload = &data[8..];
        if payload.len() % 4 != 0 {
            return Err(StorageError::Corrupt("payload length not divisible by 4".into()));
        }
        let weights = payload
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        Ok(weights)
    }

    fn is_compressed(&self) -> bool {
        false
    }
}

/// Zstd-compressed snapshot. Smaller disk footprint at the cost of CPU.
pub struct ZstdSerializer {
    level: i32,
}

impl ZstdSerializer {
    pub fn new(level: i32) -> Self {
        Self { level }
    }
}

impl SnapshotSerializer for ZstdSerializer {
    fn serialize(&self, weights: &[f32]) -> Result<Vec<u8>, StorageError> {
        let flat = FlatfileSerializer;
        let raw = flat.serialize(weights)?;
        zstd::encode_all(std::io::Cursor::new(&raw), self.level)
            .map_err(|e| StorageError::Serialization(e.to_string()))
    }

    fn deserialize(&self, data: &[u8]) -> Result<Vec<f32>, StorageError> {
        let raw = zstd::decode_all(std::io::Cursor::new(data))
            .map_err(|e| StorageError::Serialization(e.to_string()))?;
        let flat = FlatfileSerializer;
        flat.deserialize(&raw)
    }

    fn is_compressed(&self) -> bool {
        true
    }
}

/// Persists Titan MLP weight snapshots to disk.
///
/// Layout on disk:
///   `{base_path}/{layer}_{head}_{step}.snap`  — weight bytes
///   `{base_path}/{layer}_{head}_{step}.meta`  — JSON metadata
///
/// Writes are atomic: data goes to a `.tmp` file first, then
/// `rename()` swaps it in. APFS rename is atomic.
pub struct SnapshotStore {
    base_path: PathBuf,
    serializer: Box<dyn SnapshotSerializer>,
}

impl SnapshotStore {
    pub fn new(base_path: impl Into<PathBuf>, compress: bool) -> Result<Self, StorageError> {
        let base_path = base_path.into();
        std::fs::create_dir_all(&base_path)?;
        let serializer: Box<dyn SnapshotSerializer> = if compress {
            Box::new(ZstdSerializer::new(3))
        } else {
            Box::new(FlatfileSerializer)
        };
        Ok(Self { base_path, serializer })
    }

    pub fn base_path(&self) -> &Path {
        &self.base_path
    }

    fn snap_path(&self, id: &SnapshotId) -> PathBuf {
        self.base_path.join(id.to_filename())
    }

    fn meta_path(&self, id: &SnapshotId) -> PathBuf {
        self.base_path.join(id.meta_filename())
    }

    fn tmp_path(&self, id: &SnapshotId) -> PathBuf {
        self.base_path.join(format!("{}.tmp", id.to_filename()))
    }

    /// Write weights to disk atomically. Returns metadata.
    pub async fn write(
        &self,
        id: SnapshotId,
        weights: &[f32],
    ) -> Result<SnapshotMetadata, StorageError> {
        let data = self.serializer.serialize(weights)?;
        let checksum = crc32fast::hash(&data);
        let size_bytes = data.len() as u64;

        let tmp = self.tmp_path(&id);
        let snap = self.snap_path(&id);

        tokio::fs::write(&tmp, &data).await?;
        tokio::fs::rename(&tmp, &snap).await?;

        let meta = SnapshotMetadata {
            id: id.clone(),
            n_weights: weights.len(),
            size_bytes,
            created_at_secs: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            checksum,
            compressed: self.serializer.is_compressed(),
        };

        let meta_json = serde_json::to_string(&meta)
            .map_err(|e| StorageError::Serialization(e.to_string()))?;
        tokio::fs::write(self.meta_path(&id), meta_json).await?;

        Ok(meta)
    }

    /// Read weights from disk. Verifies checksum.
    pub async fn read(&self, id: &SnapshotId) -> Result<Vec<f32>, StorageError> {
        let path = self.snap_path(id);
        if !path.exists() {
            return Err(StorageError::NotFound(id.layer, id.head, id.step));
        }
        let data = tokio::fs::read(&path).await?;
        let checksum = crc32fast::hash(&data);

        // Verify against sidecar metadata if it exists
        let meta_path = self.meta_path(id);
        if meta_path.exists() {
            let meta_bytes = tokio::fs::read(&meta_path).await?;
            let meta: SnapshotMetadata = serde_json::from_slice(&meta_bytes)
                .map_err(|e| StorageError::Serialization(e.to_string()))?;
            if meta.checksum != checksum {
                return Err(StorageError::ChecksumMismatch(id.layer, id.head, id.step));
            }
        }

        self.serializer.deserialize(&data)
    }

    /// Read weights using memory-mapped I/O (uncompressed stores only).
    ///
    /// Avoids the intermediate `Vec<u8>` heap allocation that `tokio::fs::read`
    /// produces. The OS maps the `.snap` file pages directly into the process
    /// address space; the only allocation is the output `Vec<f32>`.
    ///
    /// For compressed stores the call transparently falls back to `read()`.
    pub async fn read_mmap(&self, id: &SnapshotId) -> Result<Vec<f32>, StorageError> {
        if self.serializer.is_compressed() {
            return self.read(id).await;
        }

        let path = self.snap_path(id);
        if !path.exists() {
            return Err(StorageError::NotFound(id.layer, id.head, id.step));
        }

        // Read expected checksum from sidecar (async, before entering the blocking closure).
        let meta_path = self.meta_path(id);
        let expected_checksum: Option<u32> = if meta_path.exists() {
            let meta_bytes = tokio::fs::read(&meta_path).await?;
            let meta: SnapshotMetadata = serde_json::from_slice(&meta_bytes)
                .map_err(|e| StorageError::Serialization(e.to_string()))?;
            Some(meta.checksum)
        } else {
            None
        };

        let (layer, head, step) = (id.layer, id.head, id.step);
        tokio::task::spawn_blocking(move || -> Result<Vec<f32>, StorageError> {
            let file = std::fs::File::open(&path)?;
            // SAFETY: .snap files are write-once (atomic rename in write()).
            // No other thread or process modifies the file after the rename.
            let mmap = unsafe { memmap2::Mmap::map(&file)? };
            let data: &[u8] = &mmap;

            if let Some(expected) = expected_checksum {
                let got = crc32fast::hash(data);
                if got != expected {
                    return Err(StorageError::ChecksumMismatch(layer, head, step));
                }
            }

            if data.len() < 8 || &data[..8] != MAGIC {
                return Err(StorageError::Corrupt("invalid magic header".into()));
            }
            let payload = &data[8..];
            if payload.len() % 4 != 0 {
                return Err(StorageError::Corrupt("payload length not divisible by 4".into()));
            }
            Ok(payload
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect())
        })
        .await
        .map_err(|e| StorageError::Io(std::io::Error::other(e.to_string())))?
    }

    /// Delete a snapshot and its metadata sidecar.
    pub async fn delete(&self, id: &SnapshotId) -> Result<(), StorageError> {
        let snap = self.snap_path(id);
        if snap.exists() {
            tokio::fs::remove_file(&snap).await?;
        }
        let meta = self.meta_path(id);
        if meta.exists() {
            tokio::fs::remove_file(&meta).await?;
        }
        Ok(())
    }

    /// Check if a snapshot exists on disk.
    pub async fn exists(&self, id: &SnapshotId) -> bool {
        self.snap_path(id).exists()
    }

    /// List all snapshots in the store by reading metadata sidecars.
    pub async fn list_all(&self) -> Result<Vec<SnapshotMetadata>, StorageError> {
        let mut entries = Vec::new();
        let mut dir = tokio::fs::read_dir(&self.base_path).await?;
        while let Some(entry) = dir.next_entry().await? {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if name_str.ends_with(".meta") {
                let bytes = tokio::fs::read(entry.path()).await?;
                if let Ok(meta) = serde_json::from_slice::<SnapshotMetadata>(&bytes) {
                    entries.push(meta);
                }
            }
        }
        Ok(entries)
    }

    /// List snapshots for a specific layer.
    pub async fn list_by_layer(&self, layer: u32) -> Result<Vec<SnapshotMetadata>, StorageError> {
        let all = self.list_all().await?;
        Ok(all.into_iter().filter(|m| m.id.layer == layer).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_roundtrip_flatfile() {
        let dir = tempdir().unwrap();
        let store = SnapshotStore::new(dir.path(), false).unwrap();
        let id = SnapshotId::new(0, 0, 1);
        let weights: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();

        store.write(id.clone(), &weights).await.unwrap();
        let recovered = store.read(&id).await.unwrap();

        assert_eq!(weights, recovered);
    }

    #[tokio::test]
    async fn test_roundtrip_zstd() {
        let dir = tempdir().unwrap();
        let store = SnapshotStore::new(dir.path(), true).unwrap();
        let id = SnapshotId::new(1, 2, 100);
        let weights: Vec<f32> = (0..256).map(|i| i as f32).collect();

        store.write(id.clone(), &weights).await.unwrap();
        let recovered = store.read(&id).await.unwrap();

        assert_eq!(weights, recovered);
    }

    #[tokio::test]
    async fn test_exists_and_delete() {
        let dir = tempdir().unwrap();
        let store = SnapshotStore::new(dir.path(), false).unwrap();
        let id = SnapshotId::new(0, 0, 5);

        assert!(!store.exists(&id).await);
        store.write(id.clone(), &[1.0, 2.0, 3.0]).await.unwrap();
        assert!(store.exists(&id).await);
        store.delete(&id).await.unwrap();
        assert!(!store.exists(&id).await);
    }

    #[test]
    fn test_snapshot_id_filename_roundtrip() {
        let id = SnapshotId::new(3, 7, 12345);
        let filename = id.to_filename();
        let recovered = SnapshotId::from_filename(&filename).unwrap();
        assert_eq!(id, recovered);
    }

    #[tokio::test]
    async fn test_read_mmap_roundtrip() {
        let dir = tempdir().unwrap();
        let store = SnapshotStore::new(dir.path(), false).unwrap();
        let id = SnapshotId::new(0, 0, 42);
        let weights: Vec<f32> = (0..128).map(|i| i as f32 * 0.5).collect();

        store.write(id.clone(), &weights).await.unwrap();
        let recovered = store.read_mmap(&id).await.unwrap();

        assert_eq!(weights, recovered);
    }

    #[tokio::test]
    async fn test_read_mmap_falls_through_for_compressed() {
        let dir = tempdir().unwrap();
        let store = SnapshotStore::new(dir.path(), true).unwrap(); // compressed
        let id = SnapshotId::new(0, 0, 1);
        let weights: Vec<f32> = vec![1.0, 2.0, 3.0];

        store.write(id.clone(), &weights).await.unwrap();
        // read_mmap must still work for compressed stores (falls back to read)
        let recovered = store.read_mmap(&id).await.unwrap();
        assert_eq!(weights, recovered);
    }
}
