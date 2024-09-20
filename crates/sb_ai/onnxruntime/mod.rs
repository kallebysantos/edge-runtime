mod model_session;

use core::panic;
use std::{
    borrow::Cow,
    collections::HashMap,
    hash::Hasher,
    sync::{Arc, Mutex},
};

use anyhow::{anyhow, Result};
use deno_core::{error::AnyError, op2, serde_json};
use model_session::{ModelInfo, ModelSession};
use once_cell::sync::Lazy;
use ort::{
    GraphOptimizationLevel, Session, SessionBuilder, SessionInputValue, TensorElementType, Value,
    ValueType,
};
use serde::{Deserialize, Serialize};

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

#[op2]
#[to_v8]
pub fn op_sb_ai_ort_init_session(#[buffer] model_bytes: &[u8]) -> Result<ModelInfo> {
    let model_info = ModelSession::from_bytes(model_bytes)?;

    Ok(model_info.info())
}

#[op2]
#[serde]
pub fn op_sb_ai_ort_run_session(
    #[string] model_id: String,
    #[serde] inputs: HashMap<String, Tensor>,
) -> Result<HashMap<String, Tensor>> {
    let model = ModelSession::from_id(&model_id).unwrap();

    println!("op_sb_ai_run_ort_session: loaded {model_id} -> {model:?}");

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

    let model_session = model.inner();

    // Create input session map
    let input_values = model_session
        .inputs
        .iter()
        .map(|input| {
            (
                Cow::from(input.name.to_owned()),
                SessionInputValue::from(inputs.remove(&input.name).unwrap()),
            )
        })
        .collect::<Vec<_>>();

    let outputs = model_session.run(input_values)?;
    println!("op_sb_ai_run_ort_session: outputs {outputs:?}");

    // Prepare outputs
    let output_map = model_session
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
