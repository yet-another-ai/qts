use std::cmp::max;
use std::collections::BTreeMap;
use std::ffi::CStr;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, MutexGuard};

use qts_ggml::sys;

use crate::Qwen3TtsError;

/// `ggml_backend_reg_by_name` argument for the Metal backend (device names are `MTL0`, … — not `"Metal"`).
#[cfg(all(feature = "metal", target_vendor = "apple"))]
fn ggml_reg_mtl() -> &'static CStr {
    c"MTL"
}

/// `ggml_backend_reg_by_name` argument for the Vulkan backend (per-device names are `Vulkan0`, …).
#[cfg(feature = "vulkan")]
fn ggml_reg_vulkan() -> &'static CStr {
    c"Vulkan"
}

#[cfg(any(all(feature = "metal", target_vendor = "apple"), feature = "vulkan"))]
fn gpu_device_index() -> Result<usize, Qwen3TtsError> {
    let Ok(s) = std::env::var("QWEN3_TTS_GPU_DEVICE") else {
        return Ok(0);
    };
    let s = s.trim();
    if s.is_empty() {
        return Ok(0);
    }
    s.parse().map_err(|_| {
        Qwen3TtsError::InvalidInput(format!(
            "QWEN3_TTS_GPU_DEVICE must be a non-negative integer, got {s:?}"
        ))
    })
}

/// Parsed `QWEN3_TTS_BACKEND` — which GGML primary backend to use (independent of which backends
/// were linked into the binary).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BackendChoice {
    Auto,
    Explicit(BackendPreference),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BackendPreference {
    Cpu,
    Metal,
    Vulkan,
}

impl BackendPreference {
    fn as_env_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Metal => "metal",
            Self::Vulkan => "vulkan",
        }
    }
}

fn parse_qts_backend() -> Result<BackendChoice, Qwen3TtsError> {
    let var = match std::env::var("QWEN3_TTS_BACKEND") {
        Ok(s) if !s.trim().is_empty() => s,
        _ => return Ok(BackendChoice::Auto),
    };
    match var.trim().to_ascii_lowercase().as_str() {
        "auto" => Ok(BackendChoice::Auto),
        "cpu" => Ok(BackendChoice::Explicit(BackendPreference::Cpu)),
        "metal" => Ok(BackendChoice::Explicit(BackendPreference::Metal)),
        "vulkan" => Ok(BackendChoice::Explicit(BackendPreference::Vulkan)),
        other => Err(Qwen3TtsError::InvalidInput(format!(
            "QWEN3_TTS_BACKEND: unknown value '{other}' (expected auto, cpu, metal, vulkan)"
        ))),
    }
}

fn default_auto_backend_order() -> Vec<BackendPreference> {
    #[cfg(target_vendor = "apple")]
    {
        vec![
            BackendPreference::Metal,
            BackendPreference::Vulkan,
            BackendPreference::Cpu,
        ]
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        vec![BackendPreference::Vulkan, BackendPreference::Cpu]
    }
}

fn parse_auto_backend_order() -> Result<Vec<BackendPreference>, Qwen3TtsError> {
    let var = match std::env::var("QWEN3_TTS_BACKEND_FALLBACK") {
        Ok(s) if !s.trim().is_empty() => s,
        _ => return Ok(default_auto_backend_order()),
    };
    let mut order = Vec::new();
    for token in var.split(',') {
        let value = token.trim().to_ascii_lowercase();
        if value.is_empty() {
            continue;
        }
        let pref = match value.as_str() {
            "cpu" => BackendPreference::Cpu,
            "metal" => BackendPreference::Metal,
            "vulkan" => BackendPreference::Vulkan,
            other => {
                return Err(Qwen3TtsError::InvalidInput(format!(
                    "QWEN3_TTS_BACKEND_FALLBACK: unknown backend '{other}' (expected cpu, metal, vulkan)"
                )));
            }
        };
        if !order.contains(&pref) {
            order.push(pref);
        }
    }
    if order.is_empty() {
        return Err(Qwen3TtsError::InvalidInput(
            "QWEN3_TTS_BACKEND_FALLBACK must contain at least one backend".into(),
        ));
    }
    Ok(order)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    Cpu,
    #[cfg(all(feature = "metal", target_vendor = "apple"))]
    Metal,
    #[cfg(feature = "vulkan")]
    Vulkan,
}

impl BackendKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            #[cfg(all(feature = "metal", target_vendor = "apple"))]
            Self::Metal => "Metal",
            #[cfg(feature = "vulkan")]
            Self::Vulkan => "Vulkan",
        }
    }
}

#[derive(Clone)]
pub(crate) struct BackendSet(Arc<BackendSetInner>);

// SAFETY: BackendSet wraps ggml backends whose weight data is immutable after loading.
// Each graph execution creates its own allocation via the Mutex-protected gallocr.
// The CPU backend's thread pool is per-graph-execution (set_n_threads is called before compute).
unsafe impl Send for BackendSet {}
unsafe impl Sync for BackendSet {}

struct BackendSetInner {
    primary: OwnedBackend,
    primary_kind: BackendKind,
    cpu_fallback: Option<OwnedBackend>,
    primary_galloc: Mutex<OwnedGallocr>,
}

impl BackendSet {
    pub(crate) fn cpu() -> Result<Self, Qwen3TtsError> {
        Self::with_primary(OwnedBackend::cpu()?, BackendKind::Cpu, None)
    }

    pub(crate) fn new() -> Result<Self, Qwen3TtsError> {
        unsafe {
            sys::ggml_backend_load_all();
            sys::ggml_cpu_init();
        }

        match parse_qts_backend()? {
            BackendChoice::Auto => Self::auto_select_backend(&parse_auto_backend_order()?),
            BackendChoice::Explicit(choice) => Self::require_backend(choice),
        }
    }

    fn auto_select_backend(order: &[BackendPreference]) -> Result<Self, Qwen3TtsError> {
        if backend_debug_enabled() {
            let chain = order
                .iter()
                .map(|choice| choice.as_env_str())
                .collect::<Vec<_>>()
                .join(" -> ");
            eprintln!("[backend-debug] auto fallback chain: {chain}");
        }
        for choice in order {
            if let Some(backends) = Self::try_backend(*choice)? {
                return Ok(backends);
            }
        }
        Err(Qwen3TtsError::InvalidInput(
            "QWEN3_TTS_BACKEND_FALLBACK did not contain a usable backend".into(),
        ))
    }

    fn require_backend(choice: BackendPreference) -> Result<Self, Qwen3TtsError> {
        match choice {
            BackendPreference::Cpu => {
                let primary = OwnedBackend::cpu()?;
                if backend_debug_enabled() {
                    eprintln!("[backend-debug] selected {}", BackendKind::Cpu.as_str());
                }
                Self::with_primary(primary, BackendKind::Cpu, None)
            }
            BackendPreference::Metal => {
                #[cfg(all(feature = "metal", target_vendor = "apple"))]
                {
                    Self::require_reg_backend(ggml_reg_mtl(), BackendKind::Metal)
                }
                #[cfg(not(all(feature = "metal", target_vendor = "apple")))]
                {
                    Err(Qwen3TtsError::InvalidInput(
                        "QWEN3_TTS_BACKEND=metal is only valid on Apple targets with the `metal` feature"
                            .into(),
                    ))
                }
            }
            BackendPreference::Vulkan => {
                #[cfg(feature = "vulkan")]
                {
                    Self::require_reg_backend(ggml_reg_vulkan(), BackendKind::Vulkan)
                }
                #[cfg(not(feature = "vulkan"))]
                {
                    Err(Qwen3TtsError::InvalidInput(
                        "QWEN3_TTS_BACKEND=vulkan requires building with --features vulkan".into(),
                    ))
                }
            }
        }
    }

    fn try_backend(choice: BackendPreference) -> Result<Option<Self>, Qwen3TtsError> {
        match choice {
            BackendPreference::Cpu => {
                let primary = OwnedBackend::cpu()?;
                if backend_debug_enabled() {
                    eprintln!("[backend-debug] selected {}", BackendKind::Cpu.as_str());
                }
                Ok(Some(Self::with_primary(primary, BackendKind::Cpu, None)?))
            }
            BackendPreference::Metal => {
                #[cfg(all(feature = "metal", target_vendor = "apple"))]
                {
                    Self::try_optional_reg(ggml_reg_mtl(), BackendKind::Metal)
                }
                #[cfg(not(all(feature = "metal", target_vendor = "apple")))]
                {
                    if backend_debug_enabled() {
                        eprintln!(
                            "[backend-debug] skipping metal (not supported by this build/target)"
                        );
                    }
                    Ok(None)
                }
            }
            BackendPreference::Vulkan => {
                #[cfg(feature = "vulkan")]
                {
                    Self::try_optional_reg(ggml_reg_vulkan(), BackendKind::Vulkan)
                }
                #[cfg(not(feature = "vulkan"))]
                {
                    if backend_debug_enabled() {
                        eprintln!("[backend-debug] skipping vulkan (not enabled in this build)");
                    }
                    Ok(None)
                }
            }
        }
    }

    #[cfg(any(all(feature = "metal", target_vendor = "apple"), feature = "vulkan"))]
    fn try_optional_reg(reg: &CStr, kind: BackendKind) -> Result<Option<Self>, Qwen3TtsError> {
        let label = kind.as_str();
        if let Some(primary) = OwnedBackend::init_from_reg(reg, label, false)? {
            if backend_debug_enabled() {
                eprintln!("[backend-debug] selected {label}");
            }
            return Ok(Some(Self::with_primary(
                primary,
                kind,
                Some(OwnedBackend::cpu()?),
            )?));
        }

        if backend_debug_enabled() {
            eprintln!("[backend-debug] {label} unavailable, falling back");
        }
        Ok(None)
    }

    #[cfg(any(all(feature = "metal", target_vendor = "apple"), feature = "vulkan"))]
    fn require_reg_backend(reg: &CStr, kind: BackendKind) -> Result<Self, Qwen3TtsError> {
        let label = kind.as_str();
        let primary = OwnedBackend::init_from_reg(reg, label, true)?.ok_or_else(|| {
            Qwen3TtsError::InvalidInput(format!(
                "QWEN3_TTS_BACKEND requested {label}, but GPU backend init failed (see QWEN3_TTS_GPU_DEVICE, drivers, or SDK)"
            ))
        })?;
        if backend_debug_enabled() {
            eprintln!("[backend-debug] selected {label} (QWEN3_TTS_BACKEND)");
        }
        Self::with_primary(primary, kind, Some(OwnedBackend::cpu()?))
    }

    #[allow(clippy::arc_with_non_send_sync)]
    fn with_primary(
        primary: OwnedBackend,
        primary_kind: BackendKind,
        cpu_fallback: Option<OwnedBackend>,
    ) -> Result<Self, Qwen3TtsError> {
        let primary_galloc = Mutex::new(OwnedGallocr::new(primary.as_ptr())?);
        Ok(Self(Arc::new(BackendSetInner {
            primary,
            primary_kind,
            cpu_fallback,
            primary_galloc,
        })))
    }

    pub(crate) fn primary_ptr(&self) -> sys::ggml_backend_t {
        self.0.primary.as_ptr()
    }

    pub(crate) fn primary_kind(&self) -> BackendKind {
        self.0.primary_kind
    }

    pub(crate) fn configure_threads(&self, thread_count: usize) {
        self.0.primary.set_threads(thread_count);
        if let Some(cpu_fallback) = &self.0.cpu_fallback {
            cpu_fallback.set_threads(thread_count);
        }
    }

    fn primary_galloc(&self) -> MutexGuard<'_, OwnedGallocr> {
        self.0.primary_galloc.lock().unwrap()
    }
}

struct OwnedBackend {
    raw: NonNull<sys::ggml_backend>,
    is_cpu: bool,
}

impl OwnedBackend {
    fn cpu() -> Result<Self, Qwen3TtsError> {
        let raw = unsafe { sys::ggml_backend_cpu_init() };
        let raw = NonNull::new(raw).ok_or_else(|| {
            Qwen3TtsError::InvalidInput("failed to initialize ggml CPU backend".into())
        })?;
        Ok(Self { raw, is_cpu: true })
    }

    /// Initialize via `ggml_backend_reg_by_name` + `ggml_backend_dev_init`.
    ///
    /// `ggml_backend_init_by_name` matches **per-device** ids (`Vulkan0`, `MTL0`, …), not registry
    /// names (`Vulkan`, `MTL`), so we use the registry API instead.
    #[cfg(any(all(feature = "metal", target_vendor = "apple"), feature = "vulkan"))]
    fn init_from_reg(
        reg: &CStr,
        label: &str,
        require: bool,
    ) -> Result<Option<Self>, Qwen3TtsError> {
        let device_index = gpu_device_index()?;

        let reg_ptr = unsafe { sys::ggml_backend_reg_by_name(reg.as_ptr()) };
        if reg_ptr.is_null() {
            return Ok(None);
        }

        let n = unsafe { sys::ggml_backend_reg_dev_count(reg_ptr) };
        if n == 0 {
            return Ok(None);
        }

        if device_index >= n {
            return Err(Qwen3TtsError::InvalidInput(format!(
                "QWEN3_TTS_GPU_DEVICE={device_index} is out of range for {label} (available: 0..{})",
                n.saturating_sub(1)
            )));
        }

        let dev = unsafe { sys::ggml_backend_reg_dev_get(reg_ptr, device_index) };
        if dev.is_null() {
            return Ok(None);
        }
        let raw = unsafe { sys::ggml_backend_dev_init(dev, std::ptr::null()) };
        let Some(raw) = NonNull::new(raw) else {
            if require {
                return Err(Qwen3TtsError::InvalidInput(format!(
                    "QWEN3_TTS_BACKEND requested {label}, but ggml_backend_dev_init failed for device index {device_index}"
                )));
            }
            return Ok(None);
        };

        Ok(Some(Self { raw, is_cpu: false }))
    }

    fn as_ptr(&self) -> sys::ggml_backend_t {
        self.raw.as_ptr()
    }

    fn set_threads(&self, thread_count: usize) {
        if self.is_cpu {
            unsafe {
                sys::ggml_backend_cpu_set_n_threads(
                    self.raw.as_ptr(),
                    normalize_threads(thread_count),
                );
            }
        }
    }
}

impl Drop for OwnedBackend {
    fn drop(&mut self) {
        unsafe {
            sys::ggml_backend_free(self.raw.as_ptr());
        }
    }
}

struct OwnedGallocr {
    raw: NonNull<sys::ggml_gallocr>,
}

impl OwnedGallocr {
    fn new(backend: sys::ggml_backend_t) -> Result<Self, Qwen3TtsError> {
        let raw =
            unsafe { sys::ggml_gallocr_new(sys::ggml_backend_get_default_buffer_type(backend)) };
        let raw = NonNull::new(raw).ok_or_else(|| {
            Qwen3TtsError::InvalidInput("failed to initialize ggml graph allocator".into())
        })?;
        Ok(Self { raw })
    }
}

impl Drop for OwnedGallocr {
    fn drop(&mut self) {
        unsafe {
            sys::ggml_gallocr_free(self.raw.as_ptr());
        }
    }
}

pub(crate) struct TensorUpload<'a> {
    pub(crate) tensor: *mut sys::ggml_tensor,
    pub(crate) bytes: &'a [u8],
}

pub(crate) struct TensorDownload<'a> {
    pub(crate) tensor: *mut sys::ggml_tensor,
    pub(crate) bytes: &'a mut [u8],
}

pub(crate) struct OwnedBuffer {
    raw: NonNull<sys::ggml_backend_buffer>,
}

impl OwnedBuffer {
    pub(crate) fn alloc(
        ctx: *mut sys::ggml_context,
        backend: sys::ggml_backend_t,
    ) -> Result<Self, Qwen3TtsError> {
        let raw = unsafe { sys::ggml_backend_alloc_ctx_tensors(ctx, backend) };
        let raw = NonNull::new(raw).ok_or_else(|| {
            Qwen3TtsError::InvalidInput("failed to allocate ggml backend tensor buffer".into())
        })?;
        Ok(Self { raw })
    }
}

impl Drop for OwnedBuffer {
    fn drop(&mut self) {
        unsafe {
            sys::ggml_backend_buffer_free(self.raw.as_ptr());
        }
    }
}

pub(crate) fn graph_metadata_mem_size(max_nodes: usize) -> usize {
    let tensor_overhead = unsafe { sys::ggml_tensor_overhead() };
    let graph_overhead = unsafe { sys::ggml_graph_overhead_custom(max_nodes, false) };
    max(
        1024 * 1024,
        graph_overhead + tensor_overhead * max_nodes * 16,
    )
}

pub(crate) fn slice_as_bytes<T>(slice: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(slice.as_ptr().cast::<u8>(), std::mem::size_of_val(slice)) }
}

pub(crate) fn slice_as_bytes_mut<T>(slice: &mut [T]) -> &mut [u8] {
    unsafe {
        std::slice::from_raw_parts_mut(
            slice.as_mut_ptr().cast::<u8>(),
            std::mem::size_of_val(slice),
        )
    }
}

/// F32 bias for [`sys::ggml_soft_max_ext`], matching GGML CPU `ggml_compute_forward_diag_mask_f32` / `ggml_diag_mask_inf`.
/// Metal does not implement `GGML_OP_DIAG_MASK_INF`; fusing into softmax avoids that op.
pub(crate) fn packed_softmax_diag_mask_f32(ne: [i64; 4], n_past: i32) -> Vec<f32> {
    let n0 = ne[0].max(0) as usize;
    let n1 = ne[1].max(0) as usize;
    let n2 = ne[2].max(0) as usize;
    let n3 = ne[3].max(0) as usize;
    let total = n0.saturating_mul(n1).saturating_mul(n2).saturating_mul(n3);
    let mut out = vec![0.0f32; total];
    let np = n_past.max(0) as usize;
    for i3 in 0..n3 {
        for i2 in 0..n2 {
            for j in 0..n1 {
                for i in 0..n0 {
                    let v = if i >= np && i > np.saturating_add(j) {
                        f32::NEG_INFINITY
                    } else {
                        0.0
                    };
                    let idx = i + n0 * (j + n1 * (i2 + n2 * i3));
                    if let Some(slot) = out.get_mut(idx) {
                        *slot = v;
                    }
                }
            }
        }
    }
    out
}

/// One shared mask leaf per graph (same `kq` layout for every layer). Caller must [`TensorUpload`] the packed bytes.
pub(crate) unsafe fn ggml_soft_max_ext_with_diag_mask_cache(
    ctx: *mut sys::ggml_context,
    kq: *mut sys::ggml_tensor,
    n_past: i32,
    cache: &mut Option<(*mut sys::ggml_tensor, Vec<f32>)>,
) -> *mut sys::ggml_tensor {
    let mask_ptr = match cache {
        None => {
            let ne = (*kq).ne;
            let data = packed_softmax_diag_mask_f32(ne, n_past);
            let t = sys::ggml_new_tensor_4d(
                ctx,
                sys::ggml_type_GGML_TYPE_F32,
                ne[0],
                ne[1],
                ne[2],
                ne[3],
            );
            assert!(
                !t.is_null(),
                "ggml_new_tensor_4d failed for softmax causal mask"
            );
            *cache = Some((t, data));
            t
        }
        Some((mask_t, _)) => *mask_t,
    };
    sys::ggml_soft_max_ext(ctx, kq, mask_ptr, 1.0, 0.0)
}

pub(crate) fn execute_graph(
    backends: &BackendSet,
    graph: NonNull<sys::ggml_cgraph>,
    uploads: &[TensorUpload<'_>],
    downloads: &mut [TensorDownload<'_>],
    thread_count: usize,
    error_message: &str,
) -> Result<(), Qwen3TtsError> {
    maybe_log_backend_support(backends, graph, error_message);
    backends.configure_threads(thread_count);
    let debug_enabled = backend_debug_enabled();
    let alloc_start = debug_enabled.then(std::time::Instant::now);
    let galloc = backends.primary_galloc();
    let allocated = unsafe { sys::ggml_gallocr_alloc_graph(galloc.raw.as_ptr(), graph.as_ptr()) };
    let alloc_elapsed = alloc_start.map(|start| start.elapsed());
    if !allocated {
        return Err(Qwen3TtsError::InvalidInput(format!(
            "failed to allocate backend graph for {error_message}"
        )));
    }
    if let Some(alloc_elapsed) = alloc_elapsed {
        let n_nodes = unsafe { sys::ggml_graph_n_nodes(graph.as_ptr()) };
        eprintln!(
            "[graph_alloc] alloc={:.2}ms nodes={n_nodes} ({error_message})",
            alloc_elapsed.as_secs_f64() * 1000.0,
        );
    }
    run_graph_impl(backends, graph, uploads, downloads, error_message)
}

fn run_graph_impl(
    backends: &BackendSet,
    graph: NonNull<sys::ggml_cgraph>,
    uploads: &[TensorUpload<'_>],
    downloads: &mut [TensorDownload<'_>],
    error_message: &str,
) -> Result<(), Qwen3TtsError> {
    let _t0 = std::time::Instant::now();
    for upload in uploads {
        unsafe {
            sys::ggml_backend_tensor_set(
                upload.tensor,
                upload.bytes.as_ptr().cast(),
                0,
                upload.bytes.len(),
            );
        }
    }
    let _t1 = std::time::Instant::now();
    let status = unsafe { sys::ggml_backend_graph_compute(backends.primary_ptr(), graph.as_ptr()) };
    if status != sys::ggml_status_GGML_STATUS_SUCCESS {
        return Err(Qwen3TtsError::InvalidInput(error_message.into()));
    }
    let _t2 = std::time::Instant::now();
    for download in downloads {
        unsafe {
            sys::ggml_backend_tensor_get(
                download.tensor,
                download.bytes.as_mut_ptr().cast(),
                0,
                download.bytes.len(),
            );
        }
    }
    let _t3 = std::time::Instant::now();
    if backend_debug_enabled() {
        eprintln!(
            "[graph_impl] upload={:.2}ms  compute={:.2}ms  download={:.2}ms  ({error_message})",
            (_t1 - _t0).as_secs_f64() * 1000.0,
            (_t2 - _t1).as_secs_f64() * 1000.0,
            (_t3 - _t2).as_secs_f64() * 1000.0,
        );
    }
    Ok(())
}

fn maybe_log_backend_support(backends: &BackendSet, graph: NonNull<sys::ggml_cgraph>, label: &str) {
    if !backend_debug_enabled() {
        return;
    }

    let n_nodes = unsafe { sys::ggml_graph_n_nodes(graph.as_ptr()) };
    let mut supported = 0usize;
    let mut offloaded = 0usize;
    let mut unsupported_ops = BTreeMap::<String, usize>::new();
    for idx in 0..n_nodes {
        let node = unsafe { sys::ggml_graph_node(graph.as_ptr(), idx) };
        if node.is_null() {
            continue;
        }
        let is_supported = unsafe { sys::ggml_backend_supports_op(backends.primary_ptr(), node) };
        let is_offloaded = unsafe { sys::ggml_backend_offload_op(backends.primary_ptr(), node) };
        if is_supported {
            supported += 1;
        } else {
            let op = unsafe {
                let desc = sys::ggml_op_desc(node);
                if desc.is_null() {
                    "<unknown>".to_string()
                } else {
                    CStr::from_ptr(desc).to_string_lossy().into_owned()
                }
            };
            *unsupported_ops.entry(op).or_default() += 1;
        }
        if is_offloaded {
            offloaded += 1;
        }
    }

    eprintln!(
        "[backend-debug] {label}: nodes={n_nodes} supported={supported} offloaded={offloaded}"
    );
    for (op, count) in unsupported_ops.into_iter().take(12) {
        eprintln!("[backend-debug] unsupported {op}: {count}");
    }
}

fn backend_debug_enabled() -> bool {
    std::env::var_os("QWEN3_TTS_DEBUG_BACKEND").is_some()
}

fn normalize_threads(thread_count: usize) -> i32 {
    max(1, thread_count) as i32
}
