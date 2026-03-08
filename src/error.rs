/// Top-level error type for the Drex systems layer.
#[derive(thiserror::Error, Debug)]
pub enum DrexError {
    #[error("storage error: {0}")]
    Storage(#[from] StorageError),

    #[allow(dead_code)]
    #[error("cache error: {0}")]
    Cache(String),

    #[allow(dead_code)]
    #[error("prefetch error: {0}")]
    Prefetch(String),
}

/// Storage-layer errors.
#[derive(thiserror::Error, Debug)]
pub enum StorageError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("serialization error: {0}")]
    Serialization(String),

    #[error("snapshot not found: layer={0} head={1} step={2}")]
    NotFound(u32, u32, u64),

    #[error("checksum mismatch for snapshot: layer={0} head={1} step={2}")]
    ChecksumMismatch(u32, u32, u64),

    #[error("corrupt data: {0}")]
    Corrupt(String),
}
