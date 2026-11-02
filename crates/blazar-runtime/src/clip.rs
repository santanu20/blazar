//! In-process CLIP image-embedding lane (candle, CPU).
//!
//! Serves models pulled with HF architecture `CLIPModel` (for example
//! `blazar pull openai/clip-vit-base-patch32 --revision refs/pr/15`)
//! at whatever size their own `config.json` declares - ViT-B/32,
//! B/16, L/14 and any other CLIP tower load from the same code path.
//! There is no server child to supervise: weights are mmap'd once and
//! kept resident, and every request computes on a blocking thread so
//! the async gateway never stalls on the forward pass.
//!
//! Weights always come from a blazar-pulled models directory so every
//! embedding is attributable to a stored receipt (repo, sha256, bytes).

use blazar_core::dirs::BlazarDirs;
use blazar_core::store::Store;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::clip::{
    ClipConfig, ClipModel, div_l2_norm, text_model::ClipTextConfig, vision_model::ClipVisionConfig,
};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// One resolved model layout: the candle config the weights were
/// trained with plus the preprocessing size it expects.
#[derive(Debug)]
struct ClipSpec {
    config: ClipConfig,
    image_size: u32,
}

/// HF `hidden_act` string -> candle's CLIP activation. candle 0.11
/// implements only QuickGelu for CLIP towers; the OpenAI CLIP family
/// ships quick_gelu throughout, so anything else refuses with the
/// value named rather than silently loading a numerically different
/// gelu.
fn activation_from_hf(
    name: &str,
) -> Result<candle_transformers::models::clip::text_model::Activation, String> {
    match name {
        "quick_gelu" | "quickgelu" => {
            Ok(candle_transformers::models::clip::text_model::Activation::QuickGelu)
        }
        other => Err(format!(
            "hidden_act \"{other}\" is not loadable - the candle CLIP tower implements only quick_gelu"
        )),
    }
}

fn field<'a>(section: &'a serde_json::Value, key: &str) -> Result<&'a serde_json::Value, String> {
    section
        .get(key)
        .ok_or_else(|| format!("config field \"{key}\" missing"))
}

/// Translate an HF CLIP `config.json` (verified against
/// openai/clip-vit-base-patch32, where `vision_config`/`text_config`
/// carry the tower dimensions and `hidden_size` is the tower embed
/// dim) into candle's `ClipConfig`. Legacy generation-time keys in
/// the sections are ignored; only the fields candle constructs from
/// are read.
fn spec_from_hf_config(config: &serde_json::Value) -> Result<ClipSpec, String> {
    let vision = config
        .get("vision_config")
        .ok_or("config section \"vision_config\" missing")?;
    let text = config
        .get("text_config")
        .ok_or("config section \"text_config\" missing")?;

    let act_v = activation_from_hf(
        field(vision, "hidden_act")?
            .as_str()
            .unwrap_or("quick_gelu"),
    )?;
    let act_t = activation_from_hf(field(text, "hidden_act")?.as_str().unwrap_or("quick_gelu"))?;
    let image_size = field(vision, "image_size")?
        .as_u64()
        .ok_or("vision_config.image_size not an integer")? as u32;

    let vision_config = ClipVisionConfig {
        embed_dim: field(vision, "hidden_size")?
            .as_u64()
            .ok_or("vision_config.hidden_size not an integer")? as usize,
        activation: act_v,
        intermediate_size: field(vision, "intermediate_size")?
            .as_u64()
            .ok_or("vision_config.intermediate_size not an integer")?
            as usize,
        num_hidden_layers: field(vision, "num_hidden_layers")?
            .as_u64()
            .ok_or("vision_config.num_hidden_layers not an integer")?
            as usize,
        num_attention_heads: field(vision, "num_attention_heads")?
            .as_u64()
            .ok_or("vision_config.num_attention_heads not an integer")?
            as usize,
        projection_dim: field(vision, "projection_dim")?
            .as_u64()
            .ok_or("vision_config.projection_dim not an integer")? as usize,
        // every CLIP tower ships RGB inputs; the field only appears in newer configs
        num_channels: vision
            .get("num_channels")
            .and_then(|v| v.as_u64())
            .unwrap_or(3) as usize,
        image_size: image_size as usize,
        patch_size: field(vision, "patch_size")?
            .as_u64()
            .ok_or("vision_config.patch_size not an integer")? as usize,
    };
    let text_config = ClipTextConfig {
        vocab_size: field(text, "vocab_size")?
            .as_u64()
            .ok_or("text_config.vocab_size not an integer")? as usize,
        embed_dim: field(text, "hidden_size")?
            .as_u64()
            .ok_or("text_config.hidden_size not an integer")? as usize,
        activation: act_t,
        intermediate_size: field(text, "intermediate_size")?
            .as_u64()
            .ok_or("text_config.intermediate_size not an integer")?
            as usize,
        max_position_embeddings: field(text, "max_position_embeddings")?
            .as_u64()
            .ok_or("text_config.max_position_embeddings not an integer")?
            as usize,
        // HF has no padding-token override; only the text tower would consume it
        pad_with: None,
        num_hidden_layers: field(text, "num_hidden_layers")?
            .as_u64()
            .ok_or("text_config.num_hidden_layers not an integer")?
            as usize,
        num_attention_heads: field(text, "num_attention_heads")?
            .as_u64()
            .ok_or("text_config.num_attention_heads not an integer")?
            as usize,
        projection_dim: field(text, "projection_dim")?
            .as_u64()
            .ok_or("text_config.projection_dim not an integer")? as usize,
    };
    Ok(ClipSpec {
        config: ClipConfig {
            text_config,
            vision_config,
            // contrastive-logit scale; only the paired forward pass reads it
            logit_scale_init_value: config
                .get("logit_scale_init_value")
                .and_then(|v| v.as_f64())
                .unwrap_or(2.6592) as f32,
            image_size: image_size as usize,
        },
        image_size,
    })
}

/// Read the layout from the `config.json` shipped beside the weights.
/// Rows without one (legacy stubs) fall back to the ViT-B/32 preset
/// with a note threaded into any load error, so a mismatch is never
/// mis-attributed to the weights themselves.
fn spec_for_weights(weights: &Path) -> Result<(ClipSpec, Option<String>), String> {
    let config_path = weights
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("config.json");
    match std::fs::read(&config_path) {
        Ok(bytes) => {
            let value: serde_json::Value = serde_json::from_slice(&bytes)
                .map_err(|e| format!("parse {}: {e}", config_path.display()))?;
            Ok((spec_from_hf_config(&value)?, None))
        }
        Err(_) => Ok((
            ClipSpec {
                config: ClipConfig::vit_base_patch32(),
                image_size: 224,
            },
            Some(format!(
                "config.json missing beside {}, assumed ViT-B/32 layout",
                weights.display()
            )),
        )),
    }
}

/// One resident weights set. Re-pulling the model changes the
/// directory, which invalidates this cache without a restart; the
/// layout travels with the resident because it was resolved from the
/// same directory the weights were loaded from.
struct Resident {
    weights_path: PathBuf,
    model: ClipModel,
}

static RESIDENT: Mutex<Option<Resident>> = Mutex::new(None);

/// Rows in the store whose HF config declares architecture
/// `CLIPModel` - the only kind this lane can embed.
fn clip_rows(store: &Store) -> Result<Vec<blazar_core::store::ModelRow>, String> {
    store
        .list_models()
        .map_err(|e| format!("store: {e}"))
        .map(|rows| {
            rows.into_iter()
                .filter(|r| r.arch.as_deref() == Some("CLIPModel"))
                .collect()
        })
}

/// Resolve the CLIP model to serve: the requested name when given
/// (exact match, mirroring the rest of the gateway), otherwise the
/// single installed CLIP model. Every failure teaches the remedy.
pub fn resolve(dirs: &BlazarDirs, requested: Option<&str>) -> Result<(String, PathBuf), String> {
    let store = Store::open(dirs).map_err(|e| format!("store: {e}"))?;
    let rows = clip_rows(&store)?;
    if let Some(name) = requested {
        let row = rows.iter().find(|r| r.name == name).ok_or_else(|| {
            format!(
                "no CLIP model named \"{name}\" - installed: {}",
                names(&rows)
            )
        })?;
        Ok((row.name.clone(), weights_file(&row.path)?))
    } else {
        if rows.len() != 1 {
            return Err(format!(
                "expected exactly one CLIP model, found {} - pass \"model\" explicitly; installed: {}",
                rows.len(),
                names(&rows)
            ));
        }
        let row = &rows[0];
        Ok((row.name.clone(), weights_file(&row.path)?))
    }
}

fn names(rows: &[blazar_core::store::ModelRow]) -> String {
    if rows.is_empty() {
        "none - pull one: blazar pull openai/clip-vit-base-patch32 --revision refs/pr/15"
            .to_string()
    } else {
        rows.iter()
            .map(|r| r.name.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    }
}

fn weights_file(model_dir: &str) -> Result<PathBuf, String> {
    let path = std::path::Path::new(model_dir).join("model.safetensors");
    if path.exists() {
        Ok(path)
    } else {
        Err(format!(
            "model.safetensors missing under {model_dir} - re-pull the model"
        ))
    }
}

/// Decode + resize one image into a `[1,3,size,size]` f32 tensor in
/// CLIP's expected `[-1,1]` pixel range (pure function - unit tested).
pub fn preprocess(bytes: &[u8], image_size: u32) -> Result<Tensor, String> {
    let img = image::load_from_memory(bytes)
        .map_err(|e| format!("decode image: {e}"))?
        .resize_to_fill(
            image_size,
            image_size,
            image::imageops::FilterType::Triangle,
        )
        .to_rgb8();
    let (w, h) = img.dimensions();
    let mut planar = Vec::with_capacity((image_size * image_size * 3) as usize);
    // affine 2/255*px - 1 maps [0,255] -> [-1,1]; emitted planar
    // (channel-major) so no permute is needed afterwards
    for c in 0..3 {
        for px in img.pixels() {
            let v = f32::from(px.0[c]);
            planar.push(v * (2.0 / 255.0) - 1.0);
        }
    }
    Tensor::from_vec(planar, (1, 3, h as usize, w as usize), &Device::Cpu)
        .map_err(|e| format!("image tensor: {e}"))
}

fn load(
    weights: &std::path::Path,
    spec: &ClipSpec,
    spec_note: Option<&str>,
) -> Result<ClipModel, String> {
    // SAFETY: mmap'd safetensors read. The weights file lives in a
    // blazar-managed models directory that only the pull lane rewrites
    // (atomically, via re-pull); no other process truncates it mid-read.
    // Re-pulls write a fresh directory, and the resident cache keys on
    // the path so a swapped file re-mmaps instead of aliasing the old map.
    #[allow(unsafe_code)]
    let vb = unsafe {
        let paths = vec![weights.to_path_buf()];
        VarBuilder::from_mmaped_safetensors(&paths, DType::F32, &Device::Cpu)
    }
    .map_err(|e| format!("mmap {}: {e}", weights.display()))?;
    ClipModel::new(vb, &spec.config).map_err(|e| {
        // when the layout was assumed rather than read from config.json,
        // say so - the mismatch is the missing file, not the weights
        match spec_note {
            Some(note) => format!("CLIP weights incompatible ({note}): {e}"),
            None => format!("CLIP weights incompatible with config.json layout: {e}"),
        }
    })
}

/// Embed a batch of images (raw encoded bytes, png/jpeg) with the
/// resolved CLIP model, returning one L2-normalized vector per image
/// (512-d for B-class towers, 768-d for L/14, per the model's own
/// config). Runs the forward pass on a blocking thread; safe to call
/// from async handlers.
pub async fn embed_images(weights: PathBuf, images: &[Vec<u8>]) -> Result<Vec<Vec<f32>>, String> {
    let (spec, spec_note) = spec_for_weights(&weights)?;
    if let Some(note) = &spec_note {
        tracing::warn!("{note}");
    }
    let prepped: Vec<Tensor> = images
        .iter()
        .map(|b| preprocess(b, spec.image_size))
        .collect::<Result<_, _>>()?;
    let batch = Tensor::cat(&prepped, 0).map_err(|e| format!("batch: {e}"))?;
    tokio::task::spawn_blocking(move || -> Result<Vec<Vec<f32>>, String> {
        let mut guard = RESIDENT.lock().expect("clip resident lock");
        let resident = match guard.as_ref() {
            Some(r) if r.weights_path == weights => r,
            _ => {
                let model = load(&weights, &spec, spec_note.as_deref())?;
                *guard = Some(Resident {
                    weights_path: weights.clone(),
                    model,
                });
                guard.as_ref().expect("just stored")
            }
        };
        let features = resident
            .model
            .get_image_features(&batch)
            .map_err(|e| format!("clip forward: {e}"))?;
        let normalized = div_l2_norm(&features).map_err(|e| format!("l2 norm: {e}"))?;
        normalized
            .to_vec2::<f32>()
            .map_err(|e| format!("read embeddings: {e}"))
    })
    .await
    .map_err(|e| format!("clip task: {e}"))?
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]
    use super::*;

    fn red_png(w: u32, h: u32) -> Vec<u8> {
        let img = image::RgbImage::from_pixel(w, h, image::Rgb([255, 0, 0]));
        let mut out = std::io::Cursor::new(Vec::new());
        img.write_to(&mut out, image::ImageFormat::Png)
            .expect("encode png");
        out.into_inner()
    }

    #[test]
    fn unit__preprocess__shape_range_and_channel_order() {
        let t = preprocess(&red_png(2, 2), 224).expect("preprocess");
        assert_eq!(t.dims(), &[1, 3, 224, 224]);
        let v = t
            .flatten_all()
            .expect("flatten")
            .to_vec1::<f32>()
            .expect("f32");
        assert_eq!(v.len(), 3 * 224 * 224);
        // planar layout: channel 0 all red -> +1, channels 1 and 2 -> -1
        let per = 224 * 224;
        for (i, x) in v.iter().enumerate() {
            let expected = if i / per == 0 { 1.0 } else { -1.0 };
            assert!((x - expected).abs() < 1e-5, "idx {i}: {x} != {expected}");
        }
    }

    #[test]
    fn unit__preprocess__resizes_to_the_models_own_image_size() {
        let t = preprocess(&red_png(2, 2), 336).expect("preprocess");
        assert_eq!(t.dims(), &[1, 3, 336, 336], "L/14 towers preprocess at 336");
    }

    #[test]
    fn unit__preprocess__rejects_garbage_bytes() {
        let err = preprocess(&[1, 2, 3, 4], 224).expect_err("must fail");
        assert!(err.contains("decode image"), "{err}");
    }

    #[test]
    fn unit__div_l2_norm__rows_become_unit_length() {
        let t = Tensor::from_vec(vec![3.0f32, 4.0f32, 0.0f32, 0.0f32], (1, 4), &Device::Cpu)
            .expect("tensor");
        let n = div_l2_norm(&t).expect("norm");
        let v = n.to_vec2::<f32>().expect("f32")[0].clone();
        let l2: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((l2 - 1.0).abs() < 1e-6, "l2 = {l2}");
        assert!((v[0] - 0.6).abs() < 1e-6 && (v[1] - 0.8).abs() < 1e-6);
    }

    /// Minimal but faithful HF CLIP config.json shape - field names and
    /// nesting match the live openai/clip-vit-base-patch32 config
    /// (legacy generation-time keys omitted; the reader ignores them).
    fn hf_config(vision: serde_json::Value, text: serde_json::Value) -> serde_json::Value {
        serde_json::json!({ "vision_config": vision, "text_config": text })
    }

    #[test]
    fn unit__spec_from_hf_config__parses_the_real_b32_shape() {
        let config = hf_config(
            serde_json::json!({
                "hidden_act": "quick_gelu", "hidden_size": 768, "image_size": 224,
                "intermediate_size": 3072, "num_attention_heads": 12, "num_hidden_layers": 12,
                "patch_size": 32, "projection_dim": 512
            }),
            serde_json::json!({
                "hidden_act": "quick_gelu", "hidden_size": 512,
                "intermediate_size": 2048, "max_position_embeddings": 77,
                "num_attention_heads": 8, "num_hidden_layers": 12,
                "projection_dim": 512, "vocab_size": 49408
            }),
        );
        let spec = spec_from_hf_config(&config).expect("spec");
        assert_eq!(spec.image_size, 224);
        assert_eq!(spec.config.image_size, 224);
        let preset = ClipConfig::vit_base_patch32();
        assert_eq!(
            spec.config.vision_config.embed_dim,
            preset.vision_config.embed_dim
        );
        assert_eq!(
            spec.config.vision_config.patch_size,
            preset.vision_config.patch_size
        );
        assert_eq!(
            spec.config.vision_config.num_channels, 3,
            "num_channels defaults to RGB"
        );
        assert_eq!(
            spec.config.text_config.vocab_size,
            preset.text_config.vocab_size
        );
        assert_eq!(spec.config.text_config.pad_with, None);
    }

    #[test]
    fn unit__spec_from_hf_config__parses_vit_large_patch14() {
        let config = hf_config(
            serde_json::json!({
                "hidden_act": "quick_gelu", "hidden_size": 1024, "image_size": 336,
                "intermediate_size": 4096, "num_attention_heads": 16, "num_hidden_layers": 24,
                "patch_size": 14, "projection_dim": 768, "num_channels": 3
            }),
            serde_json::json!({
                "hidden_act": "quick_gelu", "hidden_size": 768,
                "intermediate_size": 3072, "max_position_embeddings": 77,
                "num_attention_heads": 12, "num_hidden_layers": 12,
                "projection_dim": 768, "vocab_size": 49408
            }),
        );
        let spec = spec_from_hf_config(&config).expect("spec");
        assert_eq!(
            spec.image_size, 336,
            "L/14@336 preprocesses at its own size"
        );
        assert_eq!(spec.config.vision_config.embed_dim, 1024);
        assert_eq!(spec.config.vision_config.patch_size, 14);
        assert_eq!(spec.config.vision_config.num_hidden_layers, 24);
        assert_eq!(spec.config.text_config.embed_dim, 768);
    }

    #[test]
    fn unit__spec_from_hf_config__refuses_unknown_activation_and_missing_fields() {
        let mut config = hf_config(
            serde_json::json!({
                "hidden_act": "swiglu_shazam", "hidden_size": 768, "image_size": 224,
                "intermediate_size": 3072, "num_attention_heads": 12, "num_hidden_layers": 12,
                "patch_size": 32, "projection_dim": 512
            }),
            serde_json::json!({
                "hidden_act": "quick_gelu", "hidden_size": 512,
                "intermediate_size": 2048, "max_position_embeddings": 77,
                "num_attention_heads": 8, "num_hidden_layers": 12,
                "projection_dim": 512, "vocab_size": 49408
            }),
        );
        let err = spec_from_hf_config(&config).expect_err("unknown act");
        assert!(err.contains("swiglu_shazam"), "{err}");

        config["vision_config"]
            .as_object_mut()
            .expect("vision")
            .remove("patch_size");
        // the activation must be valid so the missing-field path is
        // what refuses (activation is validated first)
        config["vision_config"]["hidden_act"] = serde_json::json!("quick_gelu");
        let err = spec_from_hf_config(&config).expect_err("missing field");
        assert!(err.contains("patch_size"), "{err}");
    }

    #[test]
    fn unit__spec_for_weights__missing_config_falls_back_with_note() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let weights = tmp.path().join("model.safetensors");
        std::fs::write(&weights, b"stub").expect("write");
        let (spec, note) = spec_for_weights(&weights).expect("fallback spec");
        assert_eq!(spec.image_size, 224);
        let note = note.expect("fallback must carry the attribution note");
        assert!(note.contains("assumed ViT-B/32"), "{note}");
    }

    #[test]
    fn unit__resolve__teaches_on_missing_and_unknown_and_picks_sole_model() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let dirs = blazar_core::BlazarDirs {
            config_dir: tmp.path().join("cfg"),
            data_dir: tmp.path().join("data"),
        };
        std::fs::create_dir_all(&dirs.data_dir).expect("mkdir");

        let store = Store::open(&dirs).expect("store");
        let seed = |name: &str, dir: &str| {
            std::fs::create_dir_all(dir).expect("mkdir");
            std::fs::write(std::path::Path::new(dir).join("model.safetensors"), b"stub")
                .expect("write");
            store
                .upsert_model(&blazar_core::store::ModelRow {
                    name: name.to_string(),
                    repo: format!("openai/{name}"),
                    quant: "SAFETENSORS".into(),
                    path: dir.to_string(),
                    bytes: 1,
                    sha256: None,
                    mmproj_path: None,
                    components: vec![],
                    shards: 1,
                    arch: Some("CLIPModel".into()),
                    params: None,
                    ctx_train: None,
                    pulled_at: 0,
                    last_used_at: 0,
                })
                .expect("upsert");
        };

        let err = resolve(&dirs, None).expect_err("no clip models");
        assert!(err.contains("none - pull one"), "{err}");
        assert!(err.contains("clip-vit-base-patch32"), "{err}");

        seed("clip-a", tmp.path().join("model-a").to_str().unwrap());
        let (name, path) = resolve(&dirs, None).expect("sole model");
        assert_eq!(name, "clip-a");
        assert!(path.ends_with("model.safetensors"));

        let err = resolve(&dirs, Some("clip-b")).expect_err("unknown");
        assert!(
            err.contains("no CLIP model named \"clip-b\" - installed: clip-a"),
            "{err}"
        );

        seed("clip-c", tmp.path().join("model-c").to_str().unwrap());
        let err = resolve(&dirs, None).expect_err("now ambiguous");
        assert!(err.contains("pass \"model\" explicitly"), "{err}");
        assert!(err.contains("clip-a, clip-c"), "{err}");

        std::fs::remove_dir_all(tmp.path()).ok();
        std::fs::remove_dir_all(tmp.path().join("model-a").to_str().unwrap()).ok();
        std::fs::remove_dir_all(tmp.path().join("model-c").to_str().unwrap()).ok();
    }
}
