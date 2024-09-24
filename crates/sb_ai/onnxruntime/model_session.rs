use std::{
    collections::HashMap,
    hash::Hasher,
    sync::{Arc, Mutex},
};

use anyhow::{anyhow, Result};
use deno_core::{serde_v8::to_v8, ToV8};
use once_cell::sync::Lazy;
use ort::{GraphOptimizationLevel, Session, SessionBuilder};
use xxhash_rust::xxh3::Xxh3;

use crate::onnx::ensure_onnx_env_init;

type OnnxSessionMap = Mutex<HashMap<String, Arc<ModelSession>>>;
static ONNX_SESSIONS: Lazy<OnnxSessionMap> = Lazy::new(OnnxSessionMap::default);

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

fn get_model_checksum(model_bytes: &[u8]) -> String {
    let checksum = {
        let mut model_bytes = model_bytes;
        let mut hasher = Xxh3::new();
        let _ = std::io::copy(&mut model_bytes, &mut hasher);

        let hash = hasher.finish().to_be_bytes();
        faster_hex::hex_string(&hash)
    };

    checksum
}

fn load_session_from_bytes(model_bytes: &[u8]) -> Result<Arc<ModelSession>> {
    let model_id = get_model_checksum(model_bytes);

    let mut sessions = ONNX_SESSIONS.lock().unwrap();
    let model = sessions.get(&model_id);

    match model {
        Some(already_initialized_onnx_session) => Ok(already_initialized_onnx_session.clone()),
        None => {
            let session = {
                if let Some(err) = ensure_onnx_env_init() {
                    return Err(anyhow!("failed to create onnx environment: {err}"));
                }

                session_builder()?.commit_from_memory(model_bytes)?
            };

            let model = Arc::new(ModelSession::new(model_id.to_owned(), Arc::new(session)));

            sessions.insert(model_id.to_owned(), model.clone());

            Ok(model)
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub id: String,
    pub input_names: Vec<String>,
    pub output_names: Vec<String>,
}

#[derive(Debug)]
pub struct ModelSession {
    info: ModelInfo,
    inner: Arc<Session>,
}

impl ModelSession {
    fn new(id: String, session: Arc<Session>) -> Self {
        let input_names = session
            .inputs
            .iter()
            .map(|input| input.name.to_owned())
            .collect::<Vec<_>>();

        let output_names = session
            .outputs
            .iter()
            .map(|output| output.name.to_owned())
            .collect::<Vec<_>>();

        Self {
            info: ModelInfo {
                id,
                input_names,
                output_names,
            },
            inner: session,
        }
    }

    pub fn info(&self) -> ModelInfo {
        self.info.to_owned()
    }

    pub fn inner(&self) -> Arc<Session> {
        self.inner.clone()
    }

    pub fn from_id(id: &String) -> Option<Arc<Self>> {
        let sessions = ONNX_SESSIONS.lock().unwrap();
        let model = sessions.get(id);

        match model {
            Some(model) => Some(model.clone()),
            None => None,
        }
    }

    pub fn from_bytes(model_bytes: &[u8]) -> Result<Arc<Self>> {
        load_session_from_bytes(model_bytes)
    }
}

impl<'a> ToV8<'a> for ModelInfo {
    type Error = std::convert::Infallible;

    fn to_v8(
        self,
        scope: &mut deno_core::v8::HandleScope<'a>,
    ) -> std::result::Result<deno_core::v8::Local<'a, deno_core::v8::Value>, Self::Error> {
        let v8_values = to_v8(scope, (self.id, self.input_names, self.output_names));

        Ok(v8_values.unwrap())
    }
}
