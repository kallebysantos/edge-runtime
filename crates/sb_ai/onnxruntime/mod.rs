use core::panic;
use std::{
    borrow::Cow,
    collections::HashMap,
    hash::Hasher,
    sync::{Arc, Mutex},
};

use anyhow::{anyhow, Result};
use deno_core::{error::AnyError, op2, serde_json};
use once_cell::sync::Lazy;
use ort::{
    GraphOptimizationLevel, Session, SessionBuilder, SessionInputValue, TensorElementType, Value,
    ValueType,
};
use serde::{Deserialize, Serialize};
use xxhash_rust::xxh3::Xxh3;

use crate::onnx::ensure_onnx_env_init;

type OnnxSessionMap = Mutex<HashMap<String, Arc<Session>>>;
static ONNX_SESSIONS: Lazy<OnnxSessionMap> = Lazy::new(OnnxSessionMap::default);

#[derive(Serialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
struct SessionInfo {
    id: String,
    inputs: Vec<String>,
    outputs: Vec<String>,
}

// TODO: Better tensor convertion
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
struct TensorInt64 {
    #[serde(rename = "type")]
    data_type: String,
    dims: Vec<i64>,
    data: Vec<i64>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
struct TensorFloat32 {
    #[serde(rename = "type")]
    data_type: String,
    dims: Vec<i64>,
    data: Vec<f32>,
}
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
enum Tensor {
    Int64(TensorInt64),
    Float32(TensorFloat32),
}

fn session_builder() -> Result<SessionBuilder> {
    let orm_threads = std::env::var("OMP_NUM_THREADS")
        .map_or(None, |val| val.parse::<usize>().ok())
        .unwrap_or(1);

    Ok(Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        // NOTE(Nyannyacha): This is set to prevent memory leaks caused by different input
        // shapes.
        //
        // Backgrounds:
        // [1]: https://github.com/microsoft/onnxruntime/issues/11118
        // [2]: https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/framework/session_options.h#L95-L110
        .with_memory_pattern(false)?
        .with_intra_threads(orm_threads)?)
}

#[op2]
#[serde]
pub fn op_sb_ai_ort_init_session(#[buffer] model_bytes: &[u8]) -> Result<SessionInfo> {
    println!("Hello from ORT");

    let mut sessions = ONNX_SESSIONS.lock().unwrap();
    println!(
        "op_sb_ai_init_ort_session: received {} bytes.",
        model_bytes.len()
    );

    if let Some(err) = ensure_onnx_env_init() {
        return Err(anyhow!("failed to create onnx environment: {err}"));
    }

    let checksum = {
        let mut model_bytes = model_bytes;
        let mut hasher = Xxh3::new();
        let _ = std::io::copy(&mut model_bytes, &mut hasher);

        let hash = hasher.finish().to_be_bytes();
        faster_hex::hex_string(&hash)
    };

    let session = session_builder()?.commit_from_memory(model_bytes)?;

    let session_info = SessionInfo {
        id: checksum.to_owned(),
        inputs: session.inputs.iter().map(|i| i.name.to_owned()).collect(),
        outputs: session.outputs.iter().map(|o| o.name.to_owned()).collect(),
    };

    sessions.insert(checksum.to_owned(), Arc::new(session));

    Ok(session_info)
}

#[op2]
#[serde]
pub fn op_sb_ai_ort_run_session(
    #[string] session_id: String,
    #[serde] inputs: HashMap<String, Tensor>,
) -> Result<HashMap<String, Tensor>> {
    // TODO: take inputs a hash map and pass then to session.
    println!("op_sb_ai_run_ort_session: got {inputs:?}");

    let sessions = ONNX_SESSIONS.lock().unwrap();
    let session = sessions.get(&session_id).unwrap();
    println!("op_sb_ai_run_ort_session: loaded {session_id} -> {session:?}");

    // Prepare input values
    let mut inputs = inputs
        .iter()
        .map(|(key, value)| {
            // TODO: Proper conversion
            let raw_tensor = match value {
                Tensor::Int64(value) => {
                    Value::from_array((value.dims.to_owned(), value.data.to_owned())).unwrap()
                }
                Tensor::Float32(_) => {
                    panic!("invalid TensorFloat32")
                }
            };

            (key, raw_tensor)
        })
        .collect::<HashMap<_, _>>();

    // Create input session map
    let input_values = session
        .inputs
        .iter()
        .map(|input| {
            (
                Cow::from(&input.name),
                SessionInputValue::from(inputs.remove(&input.name).unwrap()),
            )
        })
        .collect::<Vec<_>>();

    let outputs = session.run(input_values)?;
    println!("op_sb_ai_run_ort_session: outputs {outputs:?}");

    // Prepare outputs
    let output_map = session
        .outputs
        .iter()
        .map(|output| {
            // TODO: Proper pattern matching
            let ValueType::Tensor { ty, .. } = output.output_type else {
                panic!("Invalid output_type");
            };
            let tensor = if let TensorElementType::Float32 = ty {
                let (dims, data) = outputs
                    .get(output.name.as_str())
                    .unwrap()
                    .try_extract_raw_tensor::<f32>()
                    .unwrap();

                Tensor::Float32(TensorFloat32 {
                    data_type: "f32".into(),
                    dims,
                    data: data.to_vec(),
                })
            } else {
                let (dims, data) = outputs
                    .get(&output.name.as_str())
                    .unwrap()
                    .try_extract_raw_tensor::<i64>()
                    .unwrap();

                Tensor::Int64(TensorInt64 {
                    data_type: "int64".into(),
                    dims,
                    data: data.to_vec(),
                })
            };

            (output.name.to_owned(), tensor)
        })
        .collect();

    Ok(output_map)
}
