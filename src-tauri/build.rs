use std::process::Command;

fn main() {
    // The commit this app is built from. The launcher compares it with the
    // installed GPU backend's manifest (stamped by packaging/package_gpu.py
    // from the same checkout): every X.Y.Z-beta.N shares one version, so the
    // version alone can't tell an older beta's backend apart. Empty outside
    // a git checkout, which skips the comparison.
    let build_id = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    println!("cargo:rustc-env=PODCODEX_BUILD_ID={build_id}");
    println!("cargo:rerun-if-changed=../.git/HEAD");
    println!("cargo:rerun-if-changed=../.git/refs/heads");
    tauri_build::build()
}
