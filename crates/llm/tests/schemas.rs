//! Every reply schema the apps send converts to a llama.cpp grammar. Ensayo's 11 and Talaan's summary schema are
//! copied in tests/schemas (see SOURCES.md). Keywords llama.cpp can't enforce (a `pattern` with `\d`) make it print
//! a warning and accept any string there; the apps keep checking replies themselves.

#[test]
fn every_app_schema_converts_to_a_grammar() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/schemas");
    let mut n = 0;
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|e| e != "json") {
            continue;
        }
        let schema = std::fs::read_to_string(&path).unwrap();
        serde_json::from_str::<serde_json::Value>(&schema)
            .unwrap_or_else(|e| panic!("{}: not JSON: {e}", path.display()));
        let grammar = llama_cpp_2::json_schema_to_grammar(&schema)
            .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        assert!(grammar.contains("root"), "{}", path.display());
        n += 1;
    }
    assert_eq!(n, 12, "11 Ensayo schemas and Talaan's summary");
}
