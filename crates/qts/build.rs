use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("manifest dir"));
    let proto_root = manifest_dir.join("proto");
    let proto_file = proto_root.join("voice_clone_prompt.proto");

    println!("cargo:rerun-if-changed={}", proto_file.display());

    let protoc = protoc_bin_vendored::protoc_bin_path().expect("vendored protoc");
    let mut config = prost_build::Config::new();
    config.protoc_executable(protoc);
    config
        .compile_protos(&[proto_file], &[proto_root])
        .expect("compile protos");

    bundle_soxr_if_possible(&manifest_dir);
}

fn bundle_soxr_if_possible(manifest_dir: &Path) {
    println!("cargo:rerun-if-env-changed=QWEN3_TTS_SOXR_SRC");
    println!("cargo:rerun-if-env-changed=QWEN3_TTS_SKIP_BUNDLED_SOXR");
    if std::env::var_os("QWEN3_TTS_SKIP_BUNDLED_SOXR").is_some() {
        return;
    }

    let Ok(out_dir) = std::env::var("OUT_DIR").map(PathBuf::from) else {
        return;
    };
    let Some(profile_dir) = profile_dir_from_out_dir(&out_dir) else {
        return;
    };

    let dll_name = if cfg!(windows) {
        "soxr.dll"
    } else if cfg!(target_os = "macos") {
        "libsoxr.dylib"
    } else {
        "libsoxr.so"
    };
    let profile_dll = profile_dir.join(dll_name);
    if profile_dll.is_file() {
        println!(
            "cargo:rustc-env=QWEN3_TTS_BUNDLED_SOXR_DLL={}",
            profile_dll.display()
        );
        return;
    }

    let Some(source_dir) = resolve_or_fetch_soxr_source(manifest_dir) else {
        println!("cargo:warning=qts: bundled libsoxr skipped; source checkout unavailable");
        return;
    };
    let build_dir = out_dir.join("soxr-build");
    if !configure_and_build_soxr(&source_dir, &build_dir) {
        println!("cargo:warning=qts: bundled libsoxr build failed; WAV ICL clone will require QWEN3_TTS_SOXR_DLL");
        return;
    }
    let Some(built_dll) = find_file(&build_dir, dll_name) else {
        println!(
            "cargo:warning=qts: bundled libsoxr build did not produce {}",
            dll_name
        );
        return;
    };

    let _ = std::fs::create_dir_all(&profile_dir);
    if let Err(err) = std::fs::copy(&built_dll, &profile_dll) {
        println!(
            "cargo:warning=qts: failed to copy bundled libsoxr {} -> {}: {}",
            built_dll.display(),
            profile_dll.display(),
            err
        );
        return;
    }
    println!(
        "cargo:warning=qts: bundled libsoxr copied to {}",
        profile_dll.display()
    );
    println!(
        "cargo:rustc-env=QWEN3_TTS_BUNDLED_SOXR_DLL={}",
        profile_dll.display()
    );
}

fn profile_dir_from_out_dir(out_dir: &Path) -> Option<PathBuf> {
    // OUT_DIR is target/{profile}/build/{pkg-hash}/out.
    out_dir.ancestors().nth(3).map(Path::to_path_buf)
}

fn resolve_or_fetch_soxr_source(manifest_dir: &Path) -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("QWEN3_TTS_SOXR_SRC").map(PathBuf::from) {
        if path.join("CMakeLists.txt").is_file() {
            return Some(path);
        }
    }

    let workspace_root = manifest_dir.join("../..");
    let existing = workspace_root.join("target/soxr-build/soxr-src");
    if existing.join("CMakeLists.txt").is_file() {
        return Some(existing);
    }

    let fetched = workspace_root.join("target/qts-bundled/soxr-src");
    if fetched.join("CMakeLists.txt").is_file() {
        return Some(fetched);
    }
    let parent = fetched.parent()?;
    let _ = std::fs::create_dir_all(parent);
    let status = Command::new("git")
        .args([
            "clone",
            "--depth",
            "1",
            "https://github.com/chirlu/soxr.git",
        ])
        .arg(&fetched)
        .status()
        .ok()?;
    status.success().then_some(fetched)
}

fn configure_and_build_soxr(source_dir: &Path, build_dir: &Path) -> bool {
    let expected_generator = expected_cmake_generator();
    if let Some(generator) = &expected_generator {
        if cmake_cache_uses_different_generator(build_dir, generator) {
            let _ = std::fs::remove_dir_all(build_dir);
        }
    }
    let _ = std::fs::create_dir_all(build_dir);
    let mut configure = Command::new("cmake");
    configure
        .arg("-S")
        .arg(source_dir)
        .arg("-B")
        .arg(build_dir)
        .args([
            "-DBUILD_SHARED_LIBS=ON",
            "-DBUILD_TESTS=OFF",
            "-DBUILD_EXAMPLES=OFF",
            "-DWITH_OPENMP=OFF",
            "-DCMAKE_BUILD_TYPE=Release",
        ]);
    if let Some(generator) = expected_generator {
        configure.arg("-G").arg(generator);
        if let Some(arch) = visual_studio_arch() {
            configure.arg("-A").arg(arch);
        }
    } else if command_exists("ninja") {
        configure.args(["-G", "Ninja"]);
    }
    if !configure.status().is_ok_and(|status| status.success()) {
        return false;
    }

    Command::new("cmake")
        .arg("--build")
        .arg(build_dir)
        .args(["--config", "Release"])
        .status()
        .is_ok_and(|status| status.success())
}

fn expected_cmake_generator() -> Option<String> {
    if let Ok(generator) = std::env::var("QWEN3_TTS_SOXR_CMAKE_GENERATOR") {
        if !generator.trim().is_empty() {
            return Some(generator);
        }
    }
    cfg!(windows).then(|| "Visual Studio 17 2022".to_string())
}

fn visual_studio_arch() -> Option<&'static str> {
    let target = std::env::var("TARGET").ok()?;
    if target.contains("x86_64") {
        Some("x64")
    } else if target.contains("aarch64") {
        Some("ARM64")
    } else if target.contains("i686") {
        Some("Win32")
    } else {
        None
    }
}

fn cmake_cache_uses_different_generator(build_dir: &Path, expected: &str) -> bool {
    let cache = build_dir.join("CMakeCache.txt");
    let Ok(cache) = std::fs::read_to_string(cache) else {
        return false;
    };
    cache
        .lines()
        .find_map(|line| line.strip_prefix("CMAKE_GENERATOR:INTERNAL="))
        .is_some_and(|actual| actual != expected)
}

fn command_exists(name: &str) -> bool {
    Command::new(name)
        .arg("--version")
        .status()
        .is_ok_and(|status| status.success())
}

fn find_file(root: &Path, file_name: &str) -> Option<PathBuf> {
    let entries = std::fs::read_dir(root).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_file() && path.file_name().is_some_and(|name| name == file_name) {
            return Some(path);
        }
        if path.is_dir() {
            if let Some(found) = find_file(&path, file_name) {
                return Some(found);
            }
        }
    }
    None
}
