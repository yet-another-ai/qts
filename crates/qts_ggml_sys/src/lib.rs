#![allow(
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    clippy::missing_safety_doc,
    clippy::useless_transmute
)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_add_graph() {
        unsafe {
            let params = ggml_init_params {
                mem_size: 16 * 1024 * 1024,
                mem_buffer: core::ptr::null_mut(),
                no_alloc: false,
            };
            let ctx = ggml_init(params);
            assert!(!ctx.is_null());

            let a = ggml_new_tensor_1d(ctx, ggml_type_GGML_TYPE_F32, 1);
            let b = ggml_new_tensor_1d(ctx, ggml_type_GGML_TYPE_F32, 1);
            assert!(!ggml_set_f32(a, 2.0).is_null());
            assert!(!ggml_set_f32(b, 3.0).is_null());
            let sum = ggml_add(ctx, a, b);

            let gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, sum);
            let status = ggml_graph_compute_with_ctx(ctx, gf, 1);
            assert_eq!(status, ggml_status_GGML_STATUS_SUCCESS);
            let v = ggml_get_f32_1d(sum, 0);
            assert!((v - 5.0).abs() < 1e-5);

            ggml_free(ctx);
        }
    }

    #[test]
    fn smoke_quantized_cast_roundtrip() {
        unsafe {
            let params = ggml_init_params {
                mem_size: 16 * 1024 * 1024,
                mem_buffer: core::ptr::null_mut(),
                no_alloc: false,
            };
            let ctx = ggml_init(params);
            assert!(!ctx.is_null());

            let q = ggml_new_tensor_1d(ctx, ggml_type_GGML_TYPE_Q8_0, 32);
            assert!(!q.is_null());

            let src = (0..32).map(|i| i as f32 * 0.125).collect::<Vec<_>>();
            let q_nbytes = ggml_nbytes(q);
            let mut q_data = vec![0u8; q_nbytes];
            let written = ggml_quantize_chunk(
                ggml_type_GGML_TYPE_Q8_0,
                src.as_ptr(),
                q_data.as_mut_ptr().cast(),
                0,
                1,
                32,
                core::ptr::null(),
            );
            assert_eq!(written, q_data.len());
            core::ptr::copy_nonoverlapping(q_data.as_ptr(), ggml_get_data(q).cast(), q_data.len());

            let out = ggml_cast(ctx, q, ggml_type_GGML_TYPE_F32);
            let gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, out);
            let status = ggml_graph_compute_with_ctx(ctx, gf, 1);
            assert_eq!(status, ggml_status_GGML_STATUS_SUCCESS);

            let mut recovered = vec![0.0f32; 32];
            core::ptr::copy_nonoverlapping(ggml_get_data_f32(out), recovered.as_mut_ptr(), 32);
            for (expected, actual) in src.iter().zip(recovered.iter()) {
                assert!((expected - actual).abs() < 0.25);
            }

            ggml_free(ctx);
        }
    }
}
