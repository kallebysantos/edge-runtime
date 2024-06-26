use anyhow::Error;
use ndarray::{Array1, Axis, Ix3};
use ndarray_linalg::norm::{normalize, NormalizeAxis};
use ort::{inputs, Session};
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::mpsc::{self, UnboundedReceiver, UnboundedSender};
use tokio::sync::Mutex;

use crate::tensor_ops::mean_pool;

use super::{Pipeline, PipelineInput, PipelineRequest};

pub type FeatureExtractionResult = Vec<f32>;

#[derive(Debug)]
pub(crate) struct FeatureExtractionPipelineInput {
    pub prompt: String,
    pub mean_pool: bool,
    pub normalize: bool,
}
impl PipelineInput for FeatureExtractionPipelineInput {}

#[derive(Debug)]
pub struct FeatureExtractionPipeline {
    sender:
        UnboundedSender<PipelineRequest<FeatureExtractionPipelineInput, FeatureExtractionResult>>,
    receiver: Arc<
        Mutex<
            UnboundedReceiver<
                PipelineRequest<FeatureExtractionPipelineInput, FeatureExtractionResult>,
            >,
        >,
    >,
}
impl FeatureExtractionPipeline {
    pub fn init() -> Self {
        let (sender, receiver) = mpsc::unbounded_channel::<
            PipelineRequest<FeatureExtractionPipelineInput, FeatureExtractionResult>,
        >();

        Self {
            sender,
            receiver: Arc::new(Mutex::new(receiver)),
        }
    }
}
impl Pipeline<FeatureExtractionPipelineInput, FeatureExtractionResult>
    for FeatureExtractionPipeline
{
    fn get_sender(
        &self,
    ) -> UnboundedSender<PipelineRequest<FeatureExtractionPipelineInput, FeatureExtractionResult>>
    {
        self.sender.to_owned()
    }

    fn get_receiver(
        &self,
    ) -> Arc<
        Mutex<
            UnboundedReceiver<
                PipelineRequest<FeatureExtractionPipelineInput, FeatureExtractionResult>,
            >,
        >,
    > {
        Arc::clone(&self.receiver)
    }

    fn run(
        &self,
        session: &Session,
        tokenizer: &Tokenizer,
        input: &FeatureExtractionPipelineInput,
    ) -> Result<FeatureExtractionResult, Error> {
        let encoded_prompt = tokenizer
            .encode(input.prompt.to_owned(), true)
            .map_err(anyhow::Error::msg)?;

        let input_ids = encoded_prompt
            .get_ids()
            .iter()
            .map(|i| *i as i64)
            .collect::<Vec<_>>();

        let attention_mask = encoded_prompt
            .get_attention_mask()
            .iter()
            .map(|i| *i as i64)
            .collect::<Vec<_>>();

        let token_type_ids = encoded_prompt
            .get_type_ids()
            .iter()
            .map(|i| *i as i64)
            .collect::<Vec<_>>();

        let input_ids_array = Array1::from_iter(input_ids.iter().cloned());
        let input_ids_array = input_ids_array.view().insert_axis(Axis(0));

        let attention_mask_array = Array1::from_iter(attention_mask.iter().cloned());
        let attention_mask_array = attention_mask_array.view().insert_axis(Axis(0));

        let token_type_ids_array = Array1::from_iter(token_type_ids.iter().cloned());
        let token_type_ids_array = token_type_ids_array.view().insert_axis(Axis(0));

        let outputs = session.run(inputs! {
            "input_ids" => input_ids_array,
            "token_type_ids" => token_type_ids_array,
            "attention_mask" => attention_mask_array,
        }?)?;

        let embeddings = outputs["last_hidden_state"].extract_tensor()?;
        let embeddings = embeddings.into_dimensionality::<Ix3>()?;

        let result = if input.mean_pool {
            mean_pool(embeddings, attention_mask_array.insert_axis(Axis(2)))
        } else {
            embeddings.into_owned().remove_axis(Axis(0))
        };

        let result = if input.normalize {
            let (normalized, _) = normalize(result, NormalizeAxis::Row);
            normalized
        } else {
            result
        };

        Ok(result.view().to_slice().unwrap().to_vec())
    }
}
