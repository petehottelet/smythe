use smythe_noumenon::geometry::{implementation_sources, sha};
use std::{collections::BTreeSet, fs, path::Path};

fn sources(root: &Path, dir: &Path, result: &mut BTreeSet<String>) {
    for entry in fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            sources(root, &path, result);
        } else if path.extension().is_some_and(|e| e == "rs") {
            result.insert(
                path.strip_prefix(root)
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .replace('\\', "/"),
            );
        }
    }
}
#[test]
fn report_provenance_covers_every_runtime_source_and_manifest() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let mut expected: BTreeSet<String> = ["Cargo.toml", "Cargo.lock", "clippy.toml"]
        .into_iter()
        .map(str::to_owned)
        .collect();
    for name in ["smythe-core", "smythe", "smythe-cli", "smythe-noumenon"] {
        expected.insert(format!("crates/{name}/Cargo.toml"));
        sources(
            root,
            &root.join(format!("crates/{name}/src")),
            &mut expected,
        );
    }
    let actual = implementation_sources();
    assert_eq!(
        actual
            .keys()
            .map(|s| s.to_string())
            .collect::<BTreeSet<_>>(),
        expected
    );
    for (name, digest) in actual {
        assert_eq!(digest, sha(&fs::read(root.join(name)).unwrap()), "{name}");
    }
}
