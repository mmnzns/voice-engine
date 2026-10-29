//! Ids for GGUF files the person brought themselves (spec 5.6). The file stays where the person put it; the id
//! only has to survive a rename and be the same in every app.

use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;

const MIB: u64 = 1024 * 1024;

/// `import-` plus the first 12 hex digits of sha256 over the file size (8 bytes, little endian), the first MiB and
/// the last MiB. A file under 2 MiB is hashed whole, once. Reads at most 2 MiB, whatever the file size.
pub fn import_id(path: &Path) -> io::Result<String> {
    let mut file = File::open(path)?;
    let size = file.metadata()?.len();
    let mut hasher = Sha256::new();
    hasher.update(size.to_le_bytes());
    if size < 2 * MIB {
        io::copy(&mut file, &mut HashWriter(&mut hasher))?;
    } else {
        let mut chunk = vec![0u8; MIB as usize];
        file.read_exact(&mut chunk)?;
        hasher.update(&chunk);
        file.seek(SeekFrom::Start(size - MIB))?;
        file.read_exact(&mut chunk)?;
        hasher.update(&chunk);
    }
    let digest = hasher.finalize();
    let hex: String = digest.iter().take(6).map(|b| format!("{b:02x}")).collect();
    Ok(format!("import-{hex}"))
}

/// Lets `io::copy` feed the hasher.
struct HashWriter<'a>(&'a mut Sha256);

impl io::Write for HashWriter<'_> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.update(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
