use std::env;
use std::path::{Path, PathBuf};

fn main() {
    let target = env::var("TARGET").expect("TARGET");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    let ggml_root = env::var("GGML_SRC")
        .map(PathBuf::from)
        .unwrap_or_else(|_| manifest_dir.join("vendor/ggml"));
    let ggml_root = normalize_source_path(ggml_root);
    let include = ggml_root.join("include");

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=wrapper.h");
    println!(
        "cargo:rerun-if-changed={}",
        ggml_root.join("src/ggml-vulkan/ggml-vulkan.cpp").display()
    );
    println!("cargo:rerun-if-env-changed=GGML_SRC");
    println!("cargo:rerun-if-env-changed=VULKAN_SDK");
    println!("cargo:rerun-if-env-changed=BLA_VENDOR");
    println!("cargo:rerun-if-env-changed=GGML_BLAS_VENDOR");
    validate_features(&target);

    if feature_enabled("vulkan") {
        println!(
            "cargo:warning=ggml-sys: Vulkan builds require a Vulkan SDK/loader and `glslc` (for example `libvulkan-dev` and `glslc` on Linux)"
        );
    }

    let mut cfg = cmake::Config::new(&ggml_root);
    cfg.profile("Release");
    cfg.define("BUILD_SHARED_LIBS", "OFF");
    cfg.define("GGML_STATIC", "ON");
    cfg.define("GGML_BUILD_EXAMPLES", "OFF");
    cfg.define("GGML_BUILD_TESTS", "OFF");

    if feature_enabled("native") {
        cfg.define("GGML_NATIVE", "ON");
    } else {
        cfg.define("GGML_NATIVE", "OFF");
    }

    // Metal is Apple-only; silently skip on other targets even if the feature is enabled.
    if feature_enabled("metal") && target.contains("apple") {
        cfg.define("GGML_METAL", "ON");
        cfg.define("GGML_METAL_EMBED_LIBRARY", "ON");
    } else {
        cfg.define("GGML_METAL", "OFF");
    }

    if feature_enabled("blas") {
        cfg.define("GGML_BLAS", "ON");
        if let Some(blas_vendor) = env::var("GGML_BLAS_VENDOR")
            .ok()
            .or_else(|| env::var("BLA_VENDOR").ok())
            .filter(|value| !value.trim().is_empty())
        {
            cfg.define("BLA_VENDOR", &blas_vendor);
        }
        if target.contains("apple") {
            cfg.define("GGML_ACCELERATE", "ON");
        }
    } else {
        cfg.define("GGML_BLAS", "OFF");
    }

    map_feature_cmake(&mut cfg, "cuda", "GGML_CUDA");
    map_feature_cmake(&mut cfg, "vulkan", "GGML_VULKAN");
    map_feature_cmake(&mut cfg, "hip", "GGML_HIP");
    map_feature_cmake(&mut cfg, "musa", "GGML_MUSA");
    map_feature_cmake(&mut cfg, "opencl", "GGML_OPENCL");
    map_feature_cmake(&mut cfg, "rpc", "GGML_RPC");
    map_feature_cmake(&mut cfg, "sycl", "GGML_SYCL");
    map_feature_cmake(&mut cfg, "webgpu", "GGML_WEBGPU");
    map_feature_cmake(&mut cfg, "openvino", "GGML_OPENVINO");
    map_feature_cmake(&mut cfg, "hexagon", "GGML_HEXAGON");
    map_feature_cmake(&mut cfg, "cann", "GGML_CANN");
    map_feature_cmake(&mut cfg, "zendnn", "GGML_ZENDNN");
    map_feature_cmake(&mut cfg, "zdnn", "GGML_ZDNN");
    map_feature_cmake(&mut cfg, "virtgpu", "GGML_VIRTGPU");

    let dst = cfg.build();
    let lib_dir = find_lib_dir(&dst, &out_dir).unwrap_or_else(|| {
        panic!(
            "ggml-sys: could not locate static libs under cmake output {:?} or {:?}/build",
            dst, out_dir
        )
    });
    println!("cargo:rustc-link-search=native={}", lib_dir.display());

    // Link order: dependents before their dependencies (GNU ld).
    println!("cargo:rustc-link-lib=static=ggml");
    if feature_enabled("metal") && target.contains("apple") {
        println!("cargo:rustc-link-lib=static=ggml-metal");
    }
    if feature_enabled("cuda") {
        println!("cargo:rustc-link-lib=static=ggml-cuda");
    }
    if feature_enabled("vulkan") {
        println!("cargo:rustc-link-lib=static=ggml-vulkan");
        emit_vulkan_loader_links(&target);
    }
    if feature_enabled("hip") {
        println!("cargo:rustc-link-lib=static=ggml-hip");
    }
    if feature_enabled("musa") {
        println!("cargo:rustc-link-lib=static=ggml-musa");
    }
    if feature_enabled("opencl") {
        println!("cargo:rustc-link-lib=static=ggml-opencl");
    }
    if feature_enabled("blas") {
        println!("cargo:rustc-link-lib=static=ggml-blas");
    }
    if feature_enabled("rpc") {
        println!("cargo:rustc-link-lib=static=ggml-rpc");
    }
    if feature_enabled("sycl") {
        println!("cargo:rustc-link-lib=static=ggml-sycl");
    }
    if feature_enabled("webgpu") {
        println!("cargo:rustc-link-lib=static=ggml-webgpu");
    }
    if feature_enabled("openvino") {
        println!("cargo:rustc-link-lib=static=ggml-openvino");
    }
    if feature_enabled("hexagon") {
        println!("cargo:rustc-link-lib=static=ggml-hexagon");
    }
    if feature_enabled("cann") {
        println!("cargo:rustc-link-lib=static=ggml-cann");
    }
    if feature_enabled("zendnn") {
        println!("cargo:rustc-link-lib=static=ggml-zendnn");
    }
    if feature_enabled("zdnn") {
        println!("cargo:rustc-link-lib=static=ggml-zdnn");
    }
    if feature_enabled("virtgpu") {
        println!("cargo:rustc-link-lib=static=ggml-virtgpu");
    }

    println!("cargo:rustc-link-lib=static=ggml-cpu");
    println!("cargo:rustc-link-lib=static=ggml-base");

    if feature_enabled("metal") && target.contains("apple") {
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=MetalKit");
        println!("cargo:rustc-link-lib=framework=Foundation");
    }
    if target.contains("apple") {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }

    if target.contains("apple") {
        println!("cargo:rustc-link-lib=c++");
    } else if target.contains("windows") && target.contains("msvc") {
        // C++ runtime linked via ggml's MSVC build flags.
        println!("cargo:rustc-link-lib=advapi32");
    } else {
        println!("cargo:rustc-link-lib=stdc++");
    }

    if target.contains("linux") {
        println!("cargo:rustc-link-lib=gomp");
        println!("cargo:rustc-link-lib=pthread");
        println!("cargo:rustc-link-lib=m");
        println!("cargo:rustc-link-lib=dl");
    }

    generate_bindings(&include, &out_dir);
}

fn feature_enabled(name: &str) -> bool {
    env::var(format!(
        "CARGO_FEATURE_{}",
        name.to_ascii_uppercase().replace('-', "_")
    ))
    .is_ok()
}

fn normalize_source_path(path: PathBuf) -> PathBuf {
    let path = path
        .canonicalize()
        .unwrap_or_else(|e| panic!("ggml source path missing or invalid ({path:?}): {e}"));

    #[cfg(windows)]
    {
        if let Some(stripped) = path
            .to_str()
            .and_then(|raw| raw.strip_prefix(r"\\?\"))
            .map(PathBuf::from)
        {
            return stripped;
        }
    }

    path
}

fn map_feature_cmake(cfg: &mut cmake::Config, feature: &str, cmake_opt: &str) {
    if feature_enabled(feature) {
        cfg.define(cmake_opt, "ON");
    }
}

fn validate_features(target: &str) {
    if feature_enabled("metal") && !target.contains("apple") {
        println!("cargo:warning=ggml-sys: `metal` feature ignored on non-Apple target ({target})");
    }
    if feature_enabled("cuda") && target.contains("apple") {
        panic!("ggml-sys: `cuda` feature is not supported on Apple targets");
    }
}

fn emit_vulkan_loader_links(target: &str) {
    for dir in vulkan_search_dirs(target) {
        if dir.exists() {
            println!("cargo:rustc-link-search=native={}", dir.display());
        }
    }

    let lib = if target.contains("apple") {
        "dylib=vulkan"
    } else if target.contains("windows") {
        "vulkan-1"
    } else {
        "vulkan"
    };

    println!("cargo:rustc-link-lib={lib}");
}

fn vulkan_search_dirs(target: &str) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Ok(sdk) = env::var("VULKAN_SDK") {
        let sdk = PathBuf::from(sdk);
        if target.contains("windows") {
            dirs.push(sdk.join("Lib"));
        } else {
            dirs.push(sdk.join("lib"));
            if target.contains("apple") {
                dirs.push(sdk.join("macOS").join("lib"));
            }
        }
    }
    if target.contains("apple") {
        dirs.push(PathBuf::from("/opt/homebrew/lib"));
        dirs.push(PathBuf::from("/usr/local/lib"));
    }
    dirs
}

fn find_lib_dir(dst: &Path, out_dir: &Path) -> Option<PathBuf> {
    let candidates = [
        dst.join("src"),
        dst.join("lib"),
        dst.to_path_buf(),
        out_dir.join("build").join("src"),
        out_dir.join("build").join("lib"),
        out_dir.join("build").join("Release").join("src"),
        out_dir.join("build").join("Debug").join("src"),
    ];
    candidates.into_iter().find(|p| has_ggml(p))
}

fn has_ggml(dir: &Path) -> bool {
    dir.join("libggml.a").exists()
        || dir.join("libggml.lib").exists()
        || dir.join("ggml.lib").exists()
}

fn generate_bindings(include: &Path, out_dir: &Path) {
    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}", include.display()))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .allowlist_function("ggml_.*")
        .allowlist_function("gguf_.*")
        .allowlist_type("ggml_.*")
        .allowlist_type("gguf_.*")
        .allowlist_var("GGML_.*")
        .allowlist_var("GGUF_.*")
        .size_t_is_usize(true)
        .generate()
        .expect("bindgen failed on ggml headers");

    let path = out_dir.join("bindings.rs");
    bindings.write_to_file(&path).expect("write bindings.rs");
}
