use crate::error::StorageError;
use crate::storage::snapshot::SnapshotId;
use std::path::{Path, PathBuf};

/// Operation recorded in the write-ahead journal.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum JournalOp {
    Write,
    Delete,
}

/// One entry in the journal.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct JournalEntry {
    pub op: JournalOp,
    pub id: SnapshotId,
    pub timestamp_secs: u64,
}

/// Append-only write-ahead journal stored as newline-delimited JSON.
///
/// Used to reconstruct the in-memory registry after a crash or restart,
/// and to compact stale tombstone entries.
pub struct WriteAheadJournal {
    path: PathBuf,
}

impl WriteAheadJournal {
    pub fn new(base_path: &Path) -> Result<Self, StorageError> {
        std::fs::create_dir_all(base_path)?;
        Ok(Self {
            path: base_path.join("journal.ndjson"),
        })
    }

    /// Append one entry to the journal. Uses `O_APPEND` semantics via tokio.
    pub async fn append(&self, entry: &JournalEntry) -> Result<(), StorageError> {
        use tokio::io::AsyncWriteExt;
        let mut line = serde_json::to_string(entry)
            .map_err(|e| StorageError::Serialization(e.to_string()))?;
        line.push('\n');

        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .await?;
        file.write_all(line.as_bytes()).await?;
        Ok(())
    }

    #[allow(dead_code)]
    /// Read all journal entries in order.
    pub async fn read_all(&self) -> Result<Vec<JournalEntry>, StorageError> {
        if !self.path.exists() {
            return Ok(Vec::new());
        }
        let contents = tokio::fs::read_to_string(&self.path).await?;
        let mut entries = Vec::new();
        for line in contents.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let entry: JournalEntry = serde_json::from_str(line)
                .map_err(|e| StorageError::Serialization(e.to_string()))?;
            entries.push(entry);
        }
        Ok(entries)
    }

    #[allow(dead_code)]
    /// Rewrite the journal keeping only entries for IDs still in `keep_ids`.
    pub async fn compact(&self, keep_ids: &[SnapshotId]) -> Result<(), StorageError> {
        let entries = self.read_all().await?;
        let keep_set: std::collections::HashSet<&SnapshotId> = keep_ids.iter().collect();
        let retained: Vec<_> = entries
            .iter()
            .filter(|e| keep_set.contains(&e.id))
            .collect();

        let mut content = String::new();
        for entry in retained {
            let line = serde_json::to_string(entry)
                .map_err(|e| StorageError::Serialization(e.to_string()))?;
            content.push_str(&line);
            content.push('\n');
        }
        tokio::fs::write(&self.path, content).await?;
        Ok(())
    }
}
