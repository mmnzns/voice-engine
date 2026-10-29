//! `import_id`: a stable id for a GGUF the person brought, the same under any name or path (spec 5.6).

use std::fs;
use std::path::Path;
use voice_engine_models::imports::import_id;

const MIB: usize = 1024 * 1024;

fn write(path: &Path, len: usize) {
    // Not constant, so a change at any place shows.
    let bytes: Vec<u8> = (0..len).map(|i| (i.wrapping_mul(31) % 251) as u8).collect();
    fs::write(path, bytes).unwrap();
}

fn change_byte(path: &Path, at: usize) {
    let mut bytes = fs::read(path).unwrap();
    bytes[at] ^= 0xff;
    fs::write(path, bytes).unwrap();
}

#[test]
fn the_id_is_import_and_twelve_hex_digits() {
    let tmp = tempfile::tempdir().unwrap();
    let f = tmp.path().join("a.gguf");
    write(&f, 3 * MIB);
    let id = import_id(&f).unwrap();
    let hex = id.strip_prefix("import-").expect("starts with import-");
    assert_eq!(hex.len(), 12);
    assert!(
        hex.chars()
            .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c))
    );
}

#[test]
fn the_same_file_under_another_name_or_folder_has_the_same_id() {
    let tmp = tempfile::tempdir().unwrap();
    let a = tmp.path().join("one.gguf");
    write(&a, 3 * MIB);
    fs::create_dir(tmp.path().join("elsewhere")).unwrap();
    let b = tmp.path().join("elsewhere").join("renamed.gguf");
    fs::copy(&a, &b).unwrap();
    assert_eq!(import_id(&a).unwrap(), import_id(&b).unwrap());
    // Renaming in place changes nothing either.
    let c = tmp.path().join("third-name.gguf");
    fs::rename(&b, &c).unwrap();
    assert_eq!(import_id(&a).unwrap(), import_id(&c).unwrap());
}

#[test]
fn a_changed_byte_in_the_first_or_last_mib_or_a_changed_size_changes_the_id() {
    let tmp = tempfile::tempdir().unwrap();
    let original = tmp.path().join("o.gguf");
    write(&original, 3 * MIB);
    let id = import_id(&original).unwrap();

    let first = tmp.path().join("first.gguf");
    fs::copy(&original, &first).unwrap();
    change_byte(&first, 1000);
    assert_ne!(import_id(&first).unwrap(), id);

    let last = tmp.path().join("last.gguf");
    fs::copy(&original, &last).unwrap();
    change_byte(&last, 3 * MIB - 5);
    assert_ne!(import_id(&last).unwrap(), id);

    let longer = tmp.path().join("longer.gguf");
    write(&longer, 3 * MIB + 1);
    assert_ne!(import_id(&longer).unwrap(), id);
}

#[test]
fn a_byte_between_the_first_and_last_mib_is_not_read() {
    // The id is a fingerprint of the size and both ends (a multi-gigabyte file is not read whole).
    let tmp = tempfile::tempdir().unwrap();
    let a = tmp.path().join("a.gguf");
    write(&a, 4 * MIB);
    let b = tmp.path().join("b.gguf");
    fs::copy(&a, &b).unwrap();
    change_byte(&b, 2 * MIB);
    assert_eq!(import_id(&a).unwrap(), import_id(&b).unwrap());
}

#[test]
fn a_file_under_two_mib_is_hashed_whole() {
    let tmp = tempfile::tempdir().unwrap();
    let a = tmp.path().join("a.gguf");
    write(&a, 2 * MIB - 1);
    let id = import_id(&a).unwrap();
    // The middle of a file this small is read: the first and last MiB overlap and cover all of it.
    let b = tmp.path().join("b.gguf");
    fs::copy(&a, &b).unwrap();
    change_byte(&b, MIB - 100);
    assert_ne!(import_id(&b).unwrap(), id);
    let c = tmp.path().join("c.gguf");
    fs::copy(&a, &c).unwrap();
    change_byte(&c, MIB + 100);
    assert_ne!(import_id(&c).unwrap(), id);
}

#[test]
fn a_tiny_or_empty_file_works_and_a_missing_one_is_an_error() {
    let tmp = tempfile::tempdir().unwrap();
    let five = tmp.path().join("five.gguf");
    fs::write(&five, b"GGUF!").unwrap();
    let id = import_id(&five).unwrap();
    assert!(id.starts_with("import-") && id.len() == "import-".len() + 12);
    assert_eq!(id, import_id(&five).unwrap());
    let other = tmp.path().join("other.gguf");
    fs::write(&other, b"GGUF?").unwrap();
    assert_ne!(import_id(&other).unwrap(), id);
    let empty = tmp.path().join("empty.gguf");
    fs::write(&empty, b"").unwrap();
    assert_eq!(import_id(&empty).unwrap().len(), "import-".len() + 12);
    assert!(import_id(&tmp.path().join("missing.gguf")).is_err());
}
