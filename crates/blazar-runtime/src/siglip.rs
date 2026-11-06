//! In-process `SigLIP` image-embedding lane (candle, CPU).
//!
//! Serves models pulled with HF architecture `SiglipModel` (for example
//! `blazar pull google/siglip-base-patch16-224`) at whatever size their
//! own `config.json` declares - the google configs are sparse and omit
//! most fields, so the reader leans on candle's serde defaults, which
//! mirror HF's `configuration_siglip.py` defaults. There is no server
//! child to supervise: weights are mmap'd once and kept resident, and
//! every request computes on a blocking thread so the async gateway
//! never stalls on the forward pass.
//!
//! Weights always come from a blazar-pulled models directory so every
//! embedding is attributable to a stored receipt (repo, sha256, bytes).

use blazar_core::dirs::BlazarDirs;
use blazar_core::store::Store;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::clip::div_l2_norm;
use candle_transformers::models::siglip::{Config, Model, TextConfig, VisionConfig};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// One resolved model layout: the candle config the weights were
/// trained with plus the preprocessing size it expects. Debug is
/// derived for test assertions (`expect_err` needs it on the Ok side).
#[derive(Debug)]
struct SiglipSpec {
    config: Config,
    image_size: u32,
}

/// Candle 0.11 deserializes `Activation` by its serde names (the
/// lowercase spellings its rename attributes declare - the accepted
/// list comes straight from candle's own deserialize error). HF
/// configs use those same spellings almost everywhere; the one alias
/// that needs translating is `new_gelu`. Anything else is refused
/// rather than silently loading a numerically different activation.
fn activation_variant_name(name: &str) -> Result<&'static str, String> {
    const ACCEPTED: &[&str] = &[
        "gelu",
        "gelu_new",
        "newgelu",
        "relu",
        "relu2",
        "relu6",
        "silu",
        "sigmoid",
        "hardsigmoid",
        "swiglu",
        "swish",
        "mish",
        "hardswish",
        "elu",
        "leakyrelu",
        "gelu_pytorch_tanh",
        "gelupytorchtanh",
    ];
    Ok(match name {
        "new_gelu" => "gelu_new",
        // return the 'static table entry itself, not the borrowed input
        n => match ACCEPTED.iter().find(|accepted| *accepted == &n) {
            Some(accepted) => *accepted,
            None => {
                return Err(format!(
                    "hidden_act \"{n}\" has no candle implementation - the SigLIP lane refuses \
                     rather than silently loading a numerically different activation"
                ));
            }
        },
    })
}

/// Patch one tower section's `hidden_act` into candle's variant-name
/// spelling; every other field passes through untouched so serde's
/// defaults (which mirror HF's) apply exactly as candle intends.
fn patch_activation(section: &serde_json::Value) -> Result<serde_json::Value, String> {
    let mut patched = section.clone();
    if let Some(act) = patched.get("hidden_act").and_then(|v| v.as_str()) {
        let variant = activation_variant_name(act)?;
        patched["hidden_act"] = serde_json::Value::String(variant.to_string());
    }
    Ok(patched)
}

/// Translate an HF `SigLIP` `config.json` into candle's siglip `Config`.
/// Verified against google/siglip-base-patch16-224, whose sections
/// carry only a handful of fields (`patch_size`, dims, `vocab_size`) -
/// the sparse rest is exactly what candle's serde defaults exist for.
fn spec_from_hf_config(config: &serde_json::Value) -> Result<SiglipSpec, String> {
    let vision = config
        .get("vision_config")
        .ok_or("config section \"vision_config\" missing")?;
    let text = config
        .get("text_config")
        .ok_or("config section \"text_config\" missing")?;
    let vision_config: VisionConfig = serde_json::from_value(patch_activation(vision)?)
        .map_err(|e| format!("vision_config: {e}"))?;
    let text_config: TextConfig =
        serde_json::from_value(patch_activation(text)?).map_err(|e| format!("text_config: {e}"))?;
    let image_size = u32::try_from(vision_config.image_size).map_err(|_| {
        "vision_config.image_size exceeds u32 - not a real SigLIP config".to_string()
    })?;
    Ok(SiglipSpec {
        config: Config {
            text_config,
            vision_config,
        },
        image_size,
    })
}

/// Read the layout from the `config.json` shipped beside the weights.
/// Rows without one (legacy stubs) fall back to the base preset with a
/// note threaded into any load error, so a mismatch is never
/// mis-attributed to the weights themselves.
fn spec_for_weights(weights: &Path) -> Result<(SiglipSpec, Option<String>), String> {
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
            SiglipSpec {
                config: Config::base_patch16_224(),
                image_size: 224,
            },
            Some(format!(
                "config.json missing beside {}, assumed siglip-base-patch16-224 layout",
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
    model: Model,
}

static RESIDENT: Mutex<Option<Resident>> = Mutex::new(None);

/// Rows in the store whose HF config declares architecture
/// `SiglipModel` - the only kind this lane can embed.
fn siglip_rows(store: &Store) -> Result<Vec<blazar_core::store::ModelRow>, String> {
    store
        .list_models()
        .map_err(|e| format!("store: {e}"))
        .map(|rows| {
            rows.into_iter()
                .filter(|r| r.arch.as_deref() == Some("SiglipModel"))
                .collect()
        })
}

/// Resolve the `SigLIP` model to serve: the requested name when given
/// (exact match, mirroring the rest of the gateway), otherwise the
/// single installed `SigLIP` model. Every failure teaches the remedy.
pub fn resolve(dirs: &BlazarDirs, requested: Option<&str>) -> Result<(String, PathBuf), String> {
    let store = Store::open(dirs).map_err(|e| format!("store: {e}"))?;
    let rows = siglip_rows(&store)?;
    if let Some(name) = requested {
        let row = rows.iter().find(|r| r.name == name).ok_or_else(|| {
            format!(
                "no SigLIP model named \"{name}\" - installed: {}",
                names(&rows)
            )
        })?;
        Ok((row.name.clone(), weights_file(&row.path)?))
    } else {
        if rows.len() != 1 {
            return Err(format!(
                "expected exactly one SigLIP model, found {} - pass \"model\" explicitly; installed: {}",
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
        "none - pull one: blazar pull google/siglip-base-patch16-224".to_string()
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

/// Preprocessing is shared with the CLIP lane: `SigLIP`'s HF processor
/// normalizes with mean 0.5 / std 0.5, which is exactly the
/// `[-1,1]` affine the CLIP path already emits - one decoder, one
/// resize, correct pixels for both towers.
fn preprocess(bytes: &[u8], image_size: u32) -> Result<Tensor, String> {
    crate::clip::preprocess(bytes, image_size)
}

fn load(
    weights: &std::path::Path,
    spec: &SiglipSpec,
    spec_note: Option<&str>,
) -> Result<Model, String> {
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
    // candle's siglip Model::new also reads the root `logit_scale` and
    // `logit_bias` tensors (the paired-forward contrastive head); the
    // google distributions carry both, and a distribution without them
    // surfaces candle's own error rather than a silent substitute.
    Model::new(&spec.config, vb).map_err(|e| match spec_note {
        Some(note) => format!("SigLIP weights incompatible ({note}): {e}"),
        None => format!("SigLIP weights incompatible with config.json layout: {e}"),
    })
}

/// Embed a batch of images (raw encoded bytes, png/jpeg) with the
/// resolved `SigLIP` model, returning one L2-normalized vector per image
/// (768-d for the base towers, per the model's own config). Runs the
/// forward pass on a blocking thread; safe to call from async handlers.
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
        let mut guard = RESIDENT.lock().expect("siglip resident lock");
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
            .map_err(|e| format!("siglip forward: {e}"))?;
        let normalized = div_l2_norm(&features).map_err(|e| format!("l2 norm: {e}"))?;
        normalized
            .to_vec2::<f32>()
            .map_err(|e| format!("read embeddings: {e}"))
    })
    .await
    .map_err(|e| format!("siglip task: {e}"))?
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]
    use super::*;

    /// The exact sparse shape of the live google/siglip-base-patch16-224
    /// config.json: vision carries only `model_type` + `patch_size`, text a
    /// handful of dims - everything else rides candle's serde defaults.
    fn google_sparse_config() -> serde_json::Value {
        serde_json::json!({
            "architectures": ["SiglipModel"],
            "model_type": "siglip",
            "vision_config": { "model_type": "siglip_vision_model", "patch_size": 16 },
            "text_config": {
                "model_type": "siglip_text_model", "hidden_size": 768,
                "intermediate_size": 3072, "num_attention_heads": 12, "vocab_size": 32000
            }
        })
    }

    #[test]
    fn unit__spec_from_hf_config__parses_the_sparse_google_config() {
        let spec = spec_from_hf_config(&google_sparse_config()).expect("spec");
        assert_eq!(spec.image_size, 224, "default image_size applies");
        assert_eq!(spec.config.vision_config.patch_size, 16);
        assert_eq!(
            spec.config.vision_config.hidden_size, 768,
            "default dims apply"
        );
        assert_eq!(spec.config.vision_config.num_hidden_layers, 12);
        assert_eq!(spec.config.text_config.vocab_size, 32000);
        assert_eq!(spec.config.text_config.max_position_embeddings, 64);
        assert_eq!(
            spec.config.vision_config.hidden_act,
            candle_nn::Activation::GeluPytorchTanh,
            "absent act rides candle's default (the SigLIP family act)"
        );
    }

    #[test]
    fn unit__spec_from_hf_config__explicit_fields_and_activations() {
        let config = serde_json::json!({
            "vision_config": {
                "hidden_act": "gelu_pytorch_tanh", "hidden_size": 1156, "image_size": 384,
                "intermediate_size": 4304, "num_attention_heads": 16, "num_hidden_layers": 27,
                "patch_size": 14, "num_channels": 3
            },
            "text_config": {
                "hidden_act": "gelu_pytorch_tanh", "hidden_size": 1156,
                "intermediate_size": 4304, "max_position_embeddings": 64,
                "num_attention_heads": 16, "num_hidden_layers": 27, "vocab_size": 32000
            }
        });
        let spec = spec_from_hf_config(&config).expect("spec");
        assert_eq!(
            spec.image_size, 384,
            "so400m-class towers preprocess at 384"
        );
        assert_eq!(spec.config.vision_config.hidden_size, 1156);
        assert_eq!(spec.config.vision_config.num_hidden_layers, 27);
        assert_eq!(spec.config.text_config.hidden_size, 1156);
    }

    #[test]
    fn unit__spec_from_hf_config__unknown_activation_refuses() {
        let mut config = google_sparse_config();
        config["vision_config"]["hidden_act"] = serde_json::json!("gelu_shazam");
        let err = spec_from_hf_config(&config).expect_err("unknown act");
        assert!(err.contains("gelu_shazam"), "{err}");
        assert!(err.contains("no candle implementation"), "{err}");
    }

    #[test]
    fn unit__activation_variant_names__gelu_family_maps_to_candle_spelling() {
        // candle's serde names are the lowercase spellings; the HF
        // `gelu_pytorch_tanh` (the SigLIP family act) passes as-is.
        assert_eq!(
            activation_variant_name("gelu_pytorch_tanh").expect("tanh"),
            "gelu_pytorch_tanh"
        );
        assert_eq!(activation_variant_name("gelu").expect("gelu"), "gelu");
        // the one real alias: HF spells it new_gelu, serde wants gelu_new
        assert_eq!(
            activation_variant_name("new_gelu").expect("new_gelu"),
            "gelu_new"
        );
        assert!(activation_variant_name("quick_gelu_shazam").is_err());
    }

    #[test]
    fn unit__spec_for_weights__missing_config_falls_back_with_note() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let weights = tmp.path().join("model.safetensors");
        std::fs::write(&weights, b"stub").expect("write");
        let (spec, note) = spec_for_weights(&weights).expect("fallback spec");
        assert_eq!(spec.image_size, 224);
        let note = note.expect("fallback must carry the attribution note");
        assert!(note.contains("assumed siglip-base-patch16-224"), "{note}");
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
                    repo: format!("google/{name}"),
                    quant: "SAFETENSORS".into(),
                    path: dir.to_string(),
                    bytes: 1,
                    sha256: None,
                    mmproj_path: None,
                    components: vec![],
                    shards: 1,
                    arch: Some("SiglipModel".into()),
                    params: None,
                    ctx_train: None,
                    pulled_at: 0,
                    last_used_at: 0,
                })
                .expect("upsert");
        };

        let err = resolve(&dirs, None).expect_err("no siglip models");
        assert!(err.contains("none - pull one"), "{err}");
        assert!(err.contains("siglip-base-patch16-224"), "{err}");

        seed("siglip-a", tmp.path().join("model-a").to_str().unwrap());
        let (name, path) = resolve(&dirs, None).expect("sole model");
        assert_eq!(name, "siglip-a");
        assert!(path.ends_with("model.safetensors"));

        let err = resolve(&dirs, Some("siglip-b")).expect_err("unknown");
        assert!(
            err.contains("no SigLIP model named \"siglip-b\" - installed: siglip-a"),
            "{err}"
        );

        seed("siglip-c", tmp.path().join("model-c").to_str().unwrap());
        let err = resolve(&dirs, None).expect_err("now ambiguous");
        assert!(err.contains("pass \"model\" explicitly"), "{err}");
        assert!(err.contains("siglip-a, siglip-c"), "{err}");

        std::fs::remove_dir_all(tmp.path()).ok();
    }
}
