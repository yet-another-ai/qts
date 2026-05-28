use std::ffi::{CStr, CString};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::ptr::NonNull;

use qts_ggml::sys;

use super::paths::ModelPaths;
use crate::Qwen3TtsError;

pub struct GgufFile {
    ctx: NonNull<sys::gguf_context>,
    meta_ctx: Option<NonNull<sys::ggml_context>>,
    path: PathBuf,
}

#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    pub name: String,
    pub ty: sys::ggml_type,
    pub dims: Vec<usize>,
    pub offset: usize,
    pub size: usize,
}

impl GgufFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, Qwen3TtsError> {
        let path = path.as_ref().to_path_buf();
        let path_str = path.to_str().ok_or(Qwen3TtsError::InvalidPath)?;
        let path_c = CString::new(path_str)?;

        let mut meta_ctx = std::ptr::null_mut();
        let params = sys::gguf_init_params {
            no_alloc: true,
            ctx: &mut meta_ctx,
        };

        let ctx = unsafe { sys::gguf_init_from_file(path_c.as_ptr(), params) };
        let ctx = NonNull::new(ctx).ok_or_else(|| Qwen3TtsError::InvalidGguf(path.clone()))?;

        Ok(Self {
            ctx,
            meta_ctx: NonNull::new(meta_ctx),
            path,
        })
    }

    #[must_use]
    pub fn ctx_ptr(&self) -> *mut sys::gguf_context {
        self.ctx.as_ptr()
    }

    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    #[must_use]
    pub fn meta_ctx_ptr(&self) -> Option<*mut sys::ggml_context> {
        self.meta_ctx.map(NonNull::as_ptr)
    }

    #[must_use]
    pub fn n_tensors(&self) -> i64 {
        unsafe { sys::gguf_get_n_tensors(self.ctx.as_ptr()) }
    }

    #[must_use]
    pub fn data_offset(&self) -> usize {
        unsafe { sys::gguf_get_data_offset(self.ctx.as_ptr()) }
    }

    #[must_use]
    pub fn key_index(&self, key: &str) -> Option<i64> {
        let key = CString::new(key).ok()?;
        let idx = unsafe { sys::gguf_find_key(self.ctx.as_ptr(), key.as_ptr()) };
        (idx >= 0).then_some(idx)
    }

    #[must_use]
    pub fn get_u32(&self, key: &str, default: u32) -> u32 {
        self.key_index(key)
            .map(|idx| unsafe { sys::gguf_get_val_u32(self.ctx.as_ptr(), idx) })
            .unwrap_or(default)
    }

    #[must_use]
    pub fn get_f32(&self, key: &str, default: f32) -> f32 {
        self.key_index(key)
            .map(|idx| unsafe { sys::gguf_get_val_f32(self.ctx.as_ptr(), idx) })
            .unwrap_or(default)
    }

    #[must_use]
    pub fn get_arr_str(&self, key: &str) -> Option<Vec<String>> {
        let idx = self.key_index(key)?;
        let n = unsafe { sys::gguf_get_arr_n(self.ctx.as_ptr(), idx) };
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let s = unsafe { sys::gguf_get_arr_str(self.ctx.as_ptr(), idx, i) };
            if s.is_null() {
                continue;
            }
            out.push(unsafe { CStr::from_ptr(s) }.to_string_lossy().into_owned());
        }
        Some(out)
    }

    pub fn tensor_info(&self, name: &str) -> Result<GgufTensorInfo, Qwen3TtsError> {
        let meta_ctx = self
            .meta_ctx_ptr()
            .ok_or_else(|| Qwen3TtsError::InvalidGguf(self.path.clone()))?;
        let name_c = CString::new(name)?;
        let tensor = unsafe { sys::ggml_get_tensor(meta_ctx, name_c.as_ptr()) };
        let tensor =
            NonNull::new(tensor).ok_or_else(|| Qwen3TtsError::MissingTensor(name.into()))?;

        let idx = self
            .tensor_index(name)
            .ok_or_else(|| Qwen3TtsError::MissingTensor(name.into()))?;
        let n_dims = unsafe { sys::ggml_n_dims(tensor.as_ptr()) } as usize;
        let dims = (0..n_dims)
            .map(|dim| unsafe { (*tensor.as_ptr()).ne[dim] as usize })
            .collect::<Vec<_>>();

        Ok(GgufTensorInfo {
            name: name.into(),
            ty: unsafe { sys::gguf_get_tensor_type(self.ctx.as_ptr(), idx) },
            dims,
            offset: unsafe { sys::gguf_get_tensor_offset(self.ctx.as_ptr(), idx) },
            size: unsafe { sys::gguf_get_tensor_size(self.ctx.as_ptr(), idx) },
        })
    }

    pub fn read_tensor_f32(&self, name: &str) -> Result<(GgufTensorInfo, Vec<f32>), Qwen3TtsError> {
        let info = self.tensor_info(name)?;
        let raw = self.read_tensor_bytes_by_info(&info)?;

        let element_count = info
            .dims
            .iter()
            .copied()
            .try_fold(1usize, |acc, dim| acc.checked_mul(dim))
            .ok_or_else(|| Qwen3TtsError::InvalidTensor(name.into()))?;

        let data = match info.ty {
            sys::ggml_type_GGML_TYPE_F32 => {
                if info.size != element_count * std::mem::size_of::<f32>() {
                    return Err(Qwen3TtsError::InvalidTensor(name.into()));
                }
                raw.chunks_exact(4)
                    .map(|chunk| {
                        let bytes: [u8; 4] = chunk.try_into().expect("f32 chunk");
                        f32::from_le_bytes(bytes)
                    })
                    .collect()
            }
            sys::ggml_type_GGML_TYPE_F16 => {
                if info.size != element_count * std::mem::size_of::<sys::ggml_fp16_t>() {
                    return Err(Qwen3TtsError::InvalidTensor(name.into()));
                }
                let mut out = Vec::with_capacity(element_count);
                for chunk in raw.chunks_exact(2) {
                    let bytes: [u8; 2] = chunk.try_into().expect("f16 chunk");
                    let value = u16::from_le_bytes(bytes);
                    out.push(unsafe { sys::ggml_fp16_to_fp32(value) });
                }
                out
            }
            sys::ggml_type_GGML_TYPE_Q8_0 => {
                const QK8_0: usize = 32;
                const BLOCK_Q8_0_SIZE: usize = 2 + QK8_0;
                if element_count % QK8_0 != 0
                    || info.size != element_count / QK8_0 * BLOCK_Q8_0_SIZE
                {
                    return Err(Qwen3TtsError::InvalidTensor(name.into()));
                }
                let mut out = Vec::with_capacity(element_count);
                for block in raw.chunks_exact(BLOCK_Q8_0_SIZE) {
                    let scale = u16::from_le_bytes(block[0..2].try_into().expect("q8_0 scale"));
                    let scale = unsafe { sys::ggml_fp16_to_fp32(scale) };
                    for &q in &block[2..] {
                        out.push(scale * (q as i8 as f32));
                    }
                }
                out
            }
            _ => return Err(Qwen3TtsError::UnsupportedTensorType(name.into())),
        };

        Ok((info, data))
    }

    pub fn read_tensor_bytes(
        &self,
        name: &str,
    ) -> Result<(GgufTensorInfo, Vec<u8>), Qwen3TtsError> {
        let info = self.tensor_info(name)?;
        let raw = self.read_tensor_bytes_by_info(&info)?;
        Ok((info, raw))
    }

    fn tensor_index(&self, name: &str) -> Option<i64> {
        for idx in 0..self.n_tensors() {
            let tensor_name = unsafe { sys::gguf_get_tensor_name(self.ctx.as_ptr(), idx) };
            if tensor_name.is_null() {
                continue;
            }
            if unsafe { CStr::from_ptr(tensor_name) }.to_string_lossy() == name {
                return Some(idx);
            }
        }
        None
    }

    fn read_tensor_bytes_by_info(&self, info: &GgufTensorInfo) -> Result<Vec<u8>, Qwen3TtsError> {
        let mut file = File::open(&self.path)?;
        file.seek(SeekFrom::Start((self.data_offset() + info.offset) as u64))?;
        let mut raw = vec![0u8; info.size];
        file.read_exact(&mut raw)?;
        Ok(raw)
    }
}

impl Drop for GgufFile {
    fn drop(&mut self) {
        unsafe {
            sys::gguf_free(self.ctx.as_ptr());
            if let Some(meta_ctx) = self.meta_ctx {
                sys::ggml_free(meta_ctx.as_ptr());
            }
        }
    }
}

/// Validate that required model artifacts exist and the main checkpoint parses as GGUF.
pub fn load_and_validate(paths: &ModelPaths) -> Result<(), Qwen3TtsError> {
    if !paths.main_gguf.is_file() {
        return Err(Qwen3TtsError::ModelFile(paths.main_gguf.clone()));
    }
    if !paths.vocoder_onnx.is_file() {
        return Err(Qwen3TtsError::ModelFile(paths.vocoder_onnx.clone()));
    }
    validate_gguf_file(&paths.main_gguf)?;
    Ok(())
}

fn validate_gguf_file(path: &Path) -> Result<(), Qwen3TtsError> {
    GgufFile::open(path).map(|_| ())
}
