use std::ffi::{c_char, c_double, c_uint, c_ulong, c_void, CStr};
use std::path::PathBuf;
use std::sync::OnceLock;

use libloading::{Library, Symbol};

use crate::Qwen3TtsError;

const SOXR_FLOAT32_I: i32 = 0;
const SOXR_HQ: c_ulong = 4;

type SoxrError = *const c_char;
type SoxrIn = *const c_void;
type SoxrOut = *mut c_void;

#[repr(C)]
#[derive(Clone, Copy)]
struct SoxrIoSpec {
    itype: i32,
    otype: i32,
    scale: c_double,
    e: *mut c_void,
    flags: c_ulong,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct SoxrQualitySpec {
    precision: c_double,
    phase_response: c_double,
    passband_end: c_double,
    stopband_begin: c_double,
    e: *mut c_void,
    flags: c_ulong,
}

struct SoxrApi {
    _lib: Library,
    oneshot: unsafe extern "C" fn(
        c_double,
        c_double,
        c_uint,
        SoxrIn,
        usize,
        *mut usize,
        SoxrOut,
        usize,
        *mut usize,
        *const SoxrIoSpec,
        *const SoxrQualitySpec,
        *const c_void,
    ) -> SoxrError,
    io_spec: unsafe extern "C" fn(i32, i32) -> SoxrIoSpec,
    quality_spec: unsafe extern "C" fn(c_ulong, c_ulong) -> SoxrQualitySpec,
}

impl SoxrApi {
    unsafe fn load() -> Result<Self, String> {
        let mut errors = Vec::new();
        for path in candidate_paths() {
            match Library::new(&path) {
                Ok(lib) => {
                    let oneshot = {
                        let symbol: Symbol<
                            unsafe extern "C" fn(
                                c_double,
                                c_double,
                                c_uint,
                                SoxrIn,
                                usize,
                                *mut usize,
                                SoxrOut,
                                usize,
                                *mut usize,
                                *const SoxrIoSpec,
                                *const SoxrQualitySpec,
                                *const c_void,
                            ) -> SoxrError,
                        > = lib.get(b"soxr_oneshot").map_err(|err| {
                            format!("{}: missing soxr_oneshot: {err}", path.display())
                        })?;
                        *symbol
                    };
                    let io_spec = {
                        let symbol: Symbol<unsafe extern "C" fn(i32, i32) -> SoxrIoSpec> =
                            lib.get(b"soxr_io_spec").map_err(|err| {
                                format!("{}: missing soxr_io_spec: {err}", path.display())
                            })?;
                        *symbol
                    };
                    let quality_spec = {
                        let symbol: Symbol<
                            unsafe extern "C" fn(c_ulong, c_ulong) -> SoxrQualitySpec,
                        > = lib.get(b"soxr_quality_spec").map_err(|err| {
                            format!("{}: missing soxr_quality_spec: {err}", path.display())
                        })?;
                        *symbol
                    };
                    return Ok(Self {
                        _lib: lib,
                        oneshot,
                        io_spec,
                        quality_spec,
                    });
                }
                Err(err) => errors.push(format!("{}: {err}", path.display())),
            }
        }
        Err(format!(
            "soxr.dll/libsoxr.dll not found or not loadable. Set QWEN3_TTS_SOXR_DLL to a libsoxr DLL, or put it next to qts_cli.exe. Tried: {}",
            errors.join("; ")
        ))
    }
}

pub(crate) fn resample_soxr_hq(
    input: &[f32],
    from_hz: u32,
    to_hz: u32,
) -> Result<Vec<f32>, Qwen3TtsError> {
    if input.is_empty() || from_hz == to_hz {
        return Ok(input.to_vec());
    }
    static API: OnceLock<Result<SoxrApi, String>> = OnceLock::new();
    let api = API
        .get_or_init(|| unsafe { SoxrApi::load() })
        .as_ref()
        .map_err(|err| Qwen3TtsError::InvalidInput(err.clone()))?;

    let out_len = ((input.len() as f64) * (to_hz as f64) / (from_hz as f64)).ceil() as usize;
    let mut output = vec![0.0f32; out_len + 1024];
    let mut idone = 0usize;
    let mut odone = 0usize;
    let io_spec = unsafe { (api.io_spec)(SOXR_FLOAT32_I, SOXR_FLOAT32_I) };
    let quality_spec = unsafe { (api.quality_spec)(SOXR_HQ, 0) };
    let err = unsafe {
        (api.oneshot)(
            from_hz as c_double,
            to_hz as c_double,
            1,
            input.as_ptr().cast(),
            input.len(),
            &mut idone,
            output.as_mut_ptr().cast(),
            output.len(),
            &mut odone,
            &io_spec,
            &quality_spec,
            std::ptr::null(),
        )
    };
    if !err.is_null() {
        let message = unsafe { CStr::from_ptr(err) }
            .to_string_lossy()
            .into_owned();
        return Err(Qwen3TtsError::InvalidInput(format!(
            "soxr resample failed: {message}"
        )));
    }
    if idone != input.len() {
        return Err(Qwen3TtsError::InvalidInput(format!(
            "soxr consumed {idone}/{} input samples",
            input.len()
        )));
    }
    output.truncate(odone.min(out_len));
    Ok(output)
}

fn candidate_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Some(path) = std::env::var_os("QWEN3_TTS_SOXR_DLL") {
        paths.push(PathBuf::from(path));
    }
    if let Some(path) = option_env!("QWEN3_TTS_BUNDLED_SOXR_DLL") {
        paths.push(PathBuf::from(path));
    }
    #[cfg(windows)]
    {
        if let Ok(exe) = std::env::current_exe() {
            if let Some(dir) = exe.parent() {
                paths.push(dir.join("soxr.dll"));
                paths.push(dir.join("libsoxr.dll"));
            }
        }
        if let Ok(dir) = std::env::current_dir() {
            paths.push(dir.join("soxr.dll"));
            paths.push(dir.join("libsoxr.dll"));
            paths.push(dir.join("target").join("release").join("soxr.dll"));
            paths.push(dir.join("target").join("release").join("libsoxr.dll"));
        }
    }
    #[cfg(not(windows))]
    {
        paths.push(PathBuf::from("libsoxr.so"));
        paths.push(PathBuf::from("libsoxr.dylib"));
    }
    paths
}
