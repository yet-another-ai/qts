use sha2::{Digest, Sha256};
use std::env;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    match args.next().as_deref() {
        Some("bench") => run_bench(args.collect()),
        Some("profile") => run_profile(args.collect()),
        Some("hf-release") => run_hf_release(args.collect()),
        _ => {
            print_usage();
            Ok(())
        }
    }
}

fn run_bench(args: Vec<String>) -> Result<(), Box<dyn std::error::Error>> {
    let workspace_root = workspace_root()?;
    let mut backend = String::from("cpu");
    let mut model_dir = default_model_dir(&workspace_root);
    let mut text = String::from("hello");
    let mut threads = String::from("4");
    let mut frames = String::from("16");
    let mut temperature = String::from("0.0");
    let mut top_k = String::from("0");
    let mut top_p = String::from("1.0");
    let mut criterion_args = Vec::new();

    let mut idx = 0;
    while idx < args.len() {
        match args[idx].as_str() {
            "cpu" | "metal" | "vulkan" | "all" | "both" => {
                backend = args[idx].clone();
                idx += 1;
            }
            "--model-dir" => {
                model_dir = PathBuf::from(value_arg(&args, &mut idx, "--model-dir")?);
            }
            "--text" => {
                text = value_arg(&args, &mut idx, "--text")?;
            }
            "--threads" => {
                threads = value_arg(&args, &mut idx, "--threads")?;
            }
            "--frames" => {
                frames = value_arg(&args, &mut idx, "--frames")?;
            }
            "--temperature" => {
                temperature = value_arg(&args, &mut idx, "--temperature")?;
            }
            "--top-k" => {
                top_k = value_arg(&args, &mut idx, "--top-k")?;
            }
            "--top-p" => {
                top_p = value_arg(&args, &mut idx, "--top-p")?;
            }
            "--" => {
                criterion_args.extend_from_slice(&args[idx + 1..]);
                break;
            }
            other => {
                return Err(format!("unknown xtask bench argument: {other}").into());
            }
        }
    }

    let backends = match backend.as_str() {
        "cpu" => vec!["cpu"],
        "metal" => vec!["metal"],
        "vulkan" => vec!["vulkan"],
        "all" => vec!["cpu", "metal", "vulkan"],
        "both" => vec!["cpu", "metal"],
        _ => return Err(format!("unsupported backend: {backend}").into()),
    };

    for backend in backends {
        run_single_bench(
            &workspace_root,
            backend,
            &model_dir,
            &text,
            &threads,
            &frames,
            &temperature,
            &top_k,
            &top_p,
            &criterion_args,
        )?;
    }

    Ok(())
}

fn run_profile(args: Vec<String>) -> Result<(), Box<dyn std::error::Error>> {
    if args.iter().any(|a| matches!(a.as_str(), "--help" | "-h")) {
        print_profile_help();
        return Ok(());
    }

    let workspace_root = workspace_root()?;
    let mut backend = String::from("cpu");
    let mut forwarded = args;
    if let Some(first) = forwarded.first() {
        if matches!(first.as_str(), "cpu" | "metal" | "vulkan" | "all" | "both") {
            backend = forwarded.remove(0);
        }
    }

    let backends = match backend.as_str() {
        "cpu" => vec!["cpu"],
        "metal" => vec!["metal"],
        "vulkan" => vec!["vulkan"],
        "all" => vec!["cpu", "metal", "vulkan"],
        "both" => vec!["cpu", "metal"],
        _ => return Err(format!("unsupported backend: {backend}").into()),
    };

    for backend in backends {
        let mut command = Command::new("cargo");
        command.current_dir(&workspace_root);
        command.args(["run", "--release", "-p", "qts_cli"]);
        if backend != "cpu" {
            command.args(["--features", backend]);
        }
        // Decouple Cargo features from GGML backend selection: on macOS, Vulkan is only used when
        // explicitly requested (see `QWEN3_TTS_BACKEND` in `qts` `pipeline/backend.rs`).
        command.env("QWEN3_TTS_BACKEND", backend);
        command.arg("--");
        command.arg("profile");
        command.args(&forwarded);

        eprintln!(
            "running synthesis profile: cargo_features={backend} QWEN3_TTS_BACKEND={backend}"
        );

        let status = command.status()?;
        if !status.success() {
            return Err(format!("profile failed for backend={backend}").into());
        }
    }

    Ok(())
}

fn run_hf_release(args: Vec<String>) -> Result<(), Box<dyn std::error::Error>> {
    if args.iter().any(|a| matches!(a.as_str(), "--help" | "-h")) {
        print_hf_release_help();
        return Ok(());
    }

    let workspace_root = workspace_root()?;
    let mut artifacts_dir = default_model_dir(&workspace_root);
    let mut out_dir = workspace_root.join("target/hf-qts-release");
    let mut readme_template = workspace_root.join("docs/huggingface-model-card.md");
    let mut source_commit: Option<String> = None;
    let mut hf_repo_dir: Option<PathBuf> = None;
    let mut model: Option<String> = None;
    let mut raw_main_types = Vec::new();
    let mut local_files_only = false;
    let mut verbose = false;
    let mut skip_export = false;
    let mut artifacts_dir_explicit = false;
    let mut out_dir_explicit = false;

    let mut idx = 0;
    while idx < args.len() {
        match args[idx].as_str() {
            "--model" => {
                model = Some(value_arg(&args, &mut idx, "--model")?);
            }
            "--main-type" => {
                raw_main_types.push(value_arg(&args, &mut idx, "--main-type")?);
            }
            "--artifacts-dir" => {
                artifacts_dir = PathBuf::from(value_arg(&args, &mut idx, "--artifacts-dir")?);
                artifacts_dir_explicit = true;
            }
            "--out-dir" => {
                out_dir = PathBuf::from(value_arg(&args, &mut idx, "--out-dir")?);
                out_dir_explicit = true;
            }
            "--readme-template" => {
                readme_template = PathBuf::from(value_arg(&args, &mut idx, "--readme-template")?);
            }
            "--source-commit" => {
                source_commit = Some(value_arg(&args, &mut idx, "--source-commit")?);
            }
            "--hf-repo-dir" => {
                hf_repo_dir = Some(PathBuf::from(value_arg(&args, &mut idx, "--hf-repo-dir")?));
            }
            "--local-files-only" => {
                local_files_only = true;
                idx += 1;
            }
            "--verbose" => {
                verbose = true;
                idx += 1;
            }
            "--skip-export" => {
                skip_export = true;
                idx += 1;
            }
            other => {
                return Err(format!("unknown xtask hf-release argument: {other}").into());
            }
        }
    }
    let main_types = resolve_release_main_types(&raw_main_types)?;

    if !skip_export && model.is_none() {
        return Err("--model is required unless --skip-export is used".into());
    }

    if let Some(hf_repo_dir) = hf_repo_dir.as_ref() {
        ensure_git_repo_root(hf_repo_dir)?;
        if !artifacts_dir_explicit && !out_dir_explicit {
            artifacts_dir = hf_repo_dir.clone();
            out_dir = hf_repo_dir.clone();
        }
        if !skip_export && same_existing_path(&artifacts_dir, hf_repo_dir)? {
            remove_managed_release_files(hf_repo_dir)?;
        }
    }

    if !skip_export {
        run_export_model_artifacts(
            &workspace_root,
            model.as_deref().expect("validated model is present"),
            &artifacts_dir,
            &main_types,
            local_files_only,
            verbose,
        )?;
    }

    let source_commit = source_commit.unwrap_or(resolve_git_commit(&workspace_root)?);
    let artifacts = collect_release_artifacts(&artifacts_dir)?;
    let gguf_files = artifacts
        .iter()
        .filter(|path| path.extension().is_some_and(|ext| ext == "gguf"))
        .cloned()
        .collect::<Vec<_>>();
    let quantizations = gguf_files
        .iter()
        .filter_map(|path| quantization_name(path))
        .collect::<Vec<_>>();

    let package_in_place = same_existing_path(&artifacts_dir, &out_dir)?;
    if package_in_place {
        fs::create_dir_all(&out_dir)?;
    } else {
        if out_dir.exists() {
            fs::remove_dir_all(&out_dir)?;
        }
        fs::create_dir_all(&out_dir)?;

        for artifact in &artifacts {
            let file_name = artifact
                .file_name()
                .ok_or("artifact path missing file name")?;
            fs::copy(artifact, out_dir.join(file_name))?;
        }
    }

    fs::write(out_dir.join(".gitattributes"), hf_xet_gitattributes())?;

    let copied_files = collect_release_artifacts(&out_dir)?;
    let checksums = copied_files
        .iter()
        .map(|path| {
            Ok::<_, Box<dyn std::error::Error>>((file_name_string(path)?, sha256_hex(path)?))
        })
        .collect::<Result<Vec<_>, _>>()?;

    fs::write(out_dir.join("SHA256SUMS"), render_sha256sums(&checksums))?;

    let template = fs::read_to_string(&readme_template)?;
    let readme = render_hf_model_card(
        &template,
        &source_commit,
        &copied_files,
        &quantizations,
        &checksums,
    )?;
    fs::write(out_dir.join("README.md"), readme)?;

    if let Some(hf_repo_dir) = hf_repo_dir {
        let packaged_in_repo = same_existing_path(&out_dir, &hf_repo_dir)?;
        sync_release_to_hf_repo(&out_dir, &hf_repo_dir)?;
        if packaged_in_repo {
            eprintln!(
                "prepared release files directly in git repo: {}",
                hf_repo_dir.display()
            );
        } else {
            eprintln!(
                "synced prepared release files into git repo: {}",
                hf_repo_dir.display()
            );
        }
    }

    eprintln!(
        "prepared Hugging Face release directory: {}\nartifacts from: {}",
        out_dir.display(),
        artifacts_dir.display()
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_single_bench(
    workspace_root: &Path,
    backend: &str,
    model_dir: &Path,
    text: &str,
    threads: &str,
    frames: &str,
    temperature: &str,
    top_k: &str,
    top_p: &str,
    criterion_args: &[String],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut command = Command::new("cargo");
    command.current_dir(workspace_root);
    command.arg("bench");
    command.arg("-p").arg("qts");
    command.arg("--bench").arg("synthesize");
    match backend {
        "metal" => {
            command.arg("--features").arg("metal");
        }
        "vulkan" => {
            command.arg("--features").arg("vulkan");
        }
        _ => {}
    }
    command.arg("--");
    for arg in criterion_args {
        command.arg(arg);
    }

    command.env(
        "CARGO_TARGET_DIR",
        workspace_root.join(format!("target/bench-{backend}")),
    );
    command.env("QWEN3_TTS_BENCH_BACKEND", backend);
    command.env("QWEN3_TTS_BENCH_MODEL_DIR", model_dir);
    command.env("QWEN3_TTS_BENCH_TEXT", text);
    command.env("QWEN3_TTS_BENCH_THREADS", threads);
    command.env("QWEN3_TTS_BENCH_MAX_AUDIO_FRAMES", frames);
    command.env("QWEN3_TTS_BENCH_TEMPERATURE", temperature);
    command.env("QWEN3_TTS_BENCH_TOP_K", top_k);
    command.env("QWEN3_TTS_BENCH_TOP_P", top_p);

    eprintln!(
        "running criterion bench: backend={backend} model_dir={} threads={threads} frames={frames}",
        model_dir.display()
    );

    let status = command.status()?;
    if !status.success() {
        return Err(format!("criterion benchmark failed for backend={backend}").into());
    }
    Ok(())
}

fn default_model_dir(workspace_root: &Path) -> PathBuf {
    workspace_root.join("models/qwen3-tts-bundle")
}

fn release_main_types() -> [&'static str; 2] {
    ["f16", "q8_0"]
}

fn resolve_release_main_types(
    raw_values: &[String],
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    if raw_values.is_empty() {
        return Ok(release_main_types()
            .into_iter()
            .map(str::to_owned)
            .collect());
    }

    let supported = release_main_types();
    let mut resolved = Vec::new();
    for raw in raw_values {
        for part in raw.split(',') {
            let main_type = part.trim();
            if main_type.is_empty() {
                continue;
            }
            if !supported.contains(&main_type) {
                let choices = supported.join(", ");
                return Err(
                    format!("unknown --main-type {main_type:?}. Valid values: {choices}.").into(),
                );
            }
            if !resolved.iter().any(|existing| existing == main_type) {
                resolved.push(main_type.to_owned());
            }
        }
    }

    if resolved.is_empty() {
        return Err("at least one non-empty --main-type must be provided".into());
    }

    Ok(resolved)
}

fn run_export_model_artifacts(
    workspace_root: &Path,
    model: &str,
    artifacts_dir: &Path,
    main_types: &[String],
    local_files_only: bool,
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut command = Command::new("uv");
    command.current_dir(workspace_root);
    command.args([
        "run",
        "export-model-artifacts",
        "--model",
        model,
        "--out-dir",
    ]);
    command.arg(artifacts_dir);
    for main_type in main_types {
        command.args(["--main-type", main_type]);
    }
    if local_files_only {
        command.arg("--local-files-only");
    }
    if verbose {
        command.arg("--verbose");
    }

    eprintln!(
        "exporting release artifacts with uv: model={model} out_dir={}",
        artifacts_dir.display()
    );

    let status = command.status()?;
    if !status.success() {
        return Err("uv run export-model-artifacts failed".into());
    }
    Ok(())
}

fn resolve_git_commit(workspace_root: &Path) -> Result<String, Box<dyn std::error::Error>> {
    let output = Command::new("git")
        .current_dir(workspace_root)
        .args(["rev-parse", "HEAD"])
        .output()?;
    if !output.status.success() {
        return Err("failed to resolve git commit with git rev-parse HEAD".into());
    }
    Ok(String::from_utf8(output.stdout)?.trim().to_owned())
}

fn collect_release_artifacts(dir: &Path) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let mut artifacts = fs::read_dir(dir)?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| path.is_file())
        .filter(|path| {
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                return false;
            };
            name == "qwen3-tts-vocoder.onnx"
                || (name.starts_with("qwen3-tts-0.6b-") && name.ends_with(".gguf"))
        })
        .collect::<Vec<_>>();
    artifacts.sort();

    if artifacts.is_empty() {
        return Err(format!("no release artifacts found in {}", dir.display()).into());
    }
    if !artifacts.iter().any(|path| {
        path.file_name()
            .is_some_and(|n| n == "qwen3-tts-vocoder.onnx")
    }) {
        return Err(format!("missing qwen3-tts-vocoder.onnx in {}", dir.display()).into());
    }
    if !artifacts
        .iter()
        .any(|path| path.extension().is_some_and(|ext| ext == "gguf"))
    {
        return Err(format!("missing qwen3-tts-0.6b-*.gguf in {}", dir.display()).into());
    }

    Ok(artifacts)
}

fn file_name_string(path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(ToOwned::to_owned)
        .ok_or_else(|| format!("invalid file name for {}", path.display()).into())
}

fn quantization_name(path: &Path) -> Option<String> {
    let name = path.file_name()?.to_str()?;
    name.strip_prefix("qwen3-tts-0.6b-")
        .and_then(|rest| rest.strip_suffix(".gguf"))
        .map(ToOwned::to_owned)
}

fn sha256_hex(path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 8192];

    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }

    let digest = hasher.finalize();
    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        write!(&mut hex, "{byte:02x}")?;
    }
    Ok(hex)
}

fn render_sha256sums(checksums: &[(String, String)]) -> String {
    let mut out = String::new();
    for (file_name, digest) in checksums {
        out.push_str(digest);
        out.push_str("  ");
        out.push_str(file_name);
        out.push('\n');
    }
    out
}

fn render_hf_model_card(
    template: &str,
    source_commit: &str,
    copied_files: &[PathBuf],
    quantizations: &[String],
    checksums: &[(String, String)],
) -> Result<String, Box<dyn std::error::Error>> {
    let root_layout = copied_files
        .iter()
        .map(|path| file_name_string(path))
        .collect::<Result<Vec<_>, _>>()?
        .join("\n");
    let quantization_list = quantizations
        .iter()
        .map(|q| format!("- `{q}`"))
        .collect::<Vec<_>>()
        .join("\n");
    let checksum_list = checksums
        .iter()
        .map(|(file_name, digest)| format!("- `{file_name}`\n  `{digest}`"))
        .collect::<Vec<_>>()
        .join("\n");

    Ok(template
        .replace("{{SOURCE_COMMIT}}", source_commit)
        .replace("{{ROOT_LAYOUT}}", &root_layout)
        .replace("{{QUANTIZATION_LIST}}", &quantization_list)
        .replace("{{CHECKSUM_LIST}}", &checksum_list))
}

fn hf_xet_gitattributes() -> &'static str {
    "*.gguf filter=lfs diff=lfs merge=lfs -text\n*.onnx filter=lfs diff=lfs merge=lfs -text\n"
}

fn same_existing_path(left: &Path, right: &Path) -> Result<bool, Box<dyn std::error::Error>> {
    if !left.exists() || !right.exists() {
        return Ok(left == right);
    }
    Ok(fs::canonicalize(left)? == fs::canonicalize(right)?)
}

fn sync_release_to_hf_repo(
    staged_release_dir: &Path,
    hf_repo_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    ensure_git_repo_root(hf_repo_dir)?;
    let staged_release_dir = fs::canonicalize(staged_release_dir)?;
    let hf_repo_dir = fs::canonicalize(hf_repo_dir)?;
    if staged_release_dir == hf_repo_dir {
        return Ok(());
    }
    remove_managed_release_files(&hf_repo_dir)?;

    for entry in fs::read_dir(&staged_release_dir)? {
        let path = entry?.path();
        if !path.is_file() {
            continue;
        }
        let file_name = file_name_string(&path)?;
        fs::copy(&path, hf_repo_dir.join(file_name))?;
    }

    Ok(())
}

fn remove_managed_release_files(dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if !path.is_file() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if is_managed_release_file_name(name) {
            fs::remove_file(path)?;
        }
    }
    Ok(())
}

fn is_managed_release_file_name(name: &str) -> bool {
    matches!(
        name,
        ".gitattributes" | "README.md" | "SHA256SUMS" | "qwen3-tts-vocoder.onnx"
    ) || (name.starts_with("qwen3-tts-0.6b-") && name.ends_with(".gguf"))
}

fn ensure_git_repo_root(dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if !dir.is_dir() {
        return Err(format!("hf repo dir does not exist: {}", dir.display()).into());
    }

    let output = Command::new("git")
        .current_dir(dir)
        .args(["rev-parse", "--show-toplevel"])
        .output()?;
    if !output.status.success() {
        return Err(format!("hf repo dir is not a git repository: {}", dir.display()).into());
    }

    let repo_root = String::from_utf8(output.stdout)?.trim().to_owned();
    let repo_root = PathBuf::from(repo_root);
    let requested = fs::canonicalize(dir)?;
    let actual = fs::canonicalize(repo_root)?;
    if requested != actual {
        return Err(format!(
            "--hf-repo-dir must point at the repository root, got {}",
            dir.display()
        )
        .into());
    }

    Ok(())
}

fn workspace_root() -> Result<PathBuf, Box<dyn std::error::Error>> {
    Ok(PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .ok_or("xtask manifest has no workspace parent")?
        .to_path_buf())
}

fn value_arg(
    args: &[String],
    idx: &mut usize,
    flag: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    *idx += 1;
    let value = args
        .get(*idx)
        .ok_or_else(|| format!("missing value for {flag}"))?
        .clone();
    *idx += 1;
    Ok(value)
}

fn print_usage() {
    eprintln!(
        "usage:\n  cargo xtask bench [cpu|metal|vulkan|all|both] [--model-dir PATH] [--text TEXT] [--threads N] [--frames N] [--temperature F] [--top-k N] [--top-p F] [-- <criterion args>]\n  cargo xtask profile [cpu|metal|vulkan|all|both] [--model-dir PATH] [--text TEXT] [--runs N] [--out OUT.wav] [--voice-clone-prompt] [... same flags as synthesize ...]\n  cargo xtask hf-release --model MODEL [--main-type TYPE] [--artifacts-dir PATH] [--out-dir PATH] [--hf-repo-dir PATH] [--readme-template PATH] [--source-commit SHA] [--local-files-only] [--verbose] [--skip-export]\n\nTry: cargo xtask profile --help\n     cargo xtask hf-release --help"
    );
}

fn print_profile_help() {
    eprintln!(
        "cargo xtask profile — run qts_cli profile with optional backend feature\n\n\
         usage:\n  cargo xtask profile [cpu|metal|vulkan|all|both] [-- ARGS_FOR_CLI...]\n\n\
         The first token selects both `cargo --features` and sets QWEN3_TTS_BACKEND for the child \
         (so macOS + vulkan actually uses the Vulkan GGML backend, not only links it).\n\n\
         Examples:\n  cargo xtask profile cpu --model-dir models/qwen3-tts-bundle --text hello --frames 32 --runs 3\n  cargo xtask profile metal --model-dir \"$QWEN3_TTS_MODEL_DIR\" --text hello --frames 64 --out /tmp/p.wav\n  cargo xtask profile vulkan --model-dir \"$QWEN3_TTS_MODEL_DIR\" --text hello --frames 64\n\n\
         All flags after the optional backend token are passed to `qts_cli profile`."
    );
}

fn print_hf_release_help() {
    eprintln!(
        "cargo xtask hf-release — prepare a Hugging Face release directory\n\n\
         usage:\n  cargo xtask hf-release --model MODEL [--main-type TYPE] [--artifacts-dir PATH] [--out-dir PATH] [--hf-repo-dir PATH] [--readme-template PATH] [--source-commit SHA] [--local-files-only] [--verbose] [--skip-export]\n\n\
         Defaults:\n  --artifacts-dir models/qwen3-tts-bundle\n  --out-dir target/hf-qts-release\n  --readme-template docs/huggingface-model-card.md\n  --source-commit <git rev-parse HEAD>\n\n\
         By default this command exports both GGUF variants:\n  uv run export-model-artifacts --model MODEL --out-dir <artifacts-dir> --main-type f16 --main-type q8_0\n\n\
         Repeat --main-type or pass a comma-separated list to export only the variants you need.\n\n\
         Then it copies `qwen3-tts-0.6b-*.gguf` and `qwen3-tts-vocoder.onnx`, writes `README.md`, \
         `SHA256SUMS`, and `.gitattributes`, and marks `.gguf` / `.onnx` for Hugging Face Xet in the prepared release directory.\n\n\
         If --hf-repo-dir is set, those managed release files are also synced into that existing cloned git repository root.\n\n\
         Use --skip-export only when you intentionally want to package already-generated artifacts."
    );
}
