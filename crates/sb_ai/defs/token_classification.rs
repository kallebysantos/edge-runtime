use crate::{
    pipeline::try_get_config_url_from_env,
    serde_json::{Map, Value},
    tensor_ops::argmax,
};
use anyhow::anyhow;
use deno_core::error::AnyError;
use ndarray::{Array1, Axis, Ix3, Zip};
use ort::{inputs, ArrayExtensions};
use reqwest::Url;
use serde::{Deserialize, Serialize};

use crate::pipeline::{
    try_get_model_url_from_env, try_get_tokenizer_url_from_env, PipelineDefinition,
};

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum TokenClassificationAggregationStrategy {
    None,
    Simple,
    Max,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(default)]
pub struct TokenClassificationOptions {
    pub aggregation_strategy: TokenClassificationAggregationStrategy,
    pub ignore_labels: Vec<String>,
}
impl Default for TokenClassificationOptions {
    fn default() -> Self {
        Self {
            aggregation_strategy: TokenClassificationAggregationStrategy::Simple,
            ignore_labels: vec![String::from("O")],
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct RawClassifiedToken {
    pub entity: String,
    pub score: f32,
    pub index: usize,
    pub word: String,
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct ClassifiedTokenGroup {
    pub group_entity: String,
    pub score: f32,
    pub word: String,
    pub start: usize,
    pub end: usize,
    pub tokens: Vec<RawClassifiedToken>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum TokenClassificationOutput {
    Raw(Vec<RawClassifiedToken>),
    Grouped(Vec<ClassifiedTokenGroup>),
}
#[derive(Debug, Clone)]
pub struct TokenClassificationPipeline;

impl PipelineDefinition for TokenClassificationPipeline {
    type Input = String;
    type InputOptions = TokenClassificationOptions;
    type Output = TokenClassificationOutput;

    fn make() -> Self {
        TokenClassificationPipeline
    }

    fn name(&self) -> std::borrow::Cow<'static, str> {
        "token-classification".into()
    }

    fn model_url(&self, requested_variation: Option<&str>) -> Result<Url, AnyError> {
        try_get_model_url_from_env(self, requested_variation).unwrap_or(Err(anyhow!(
            "{}: no default or variation found",
            self.name()
        )))
    }

    fn tokenizer_url(&self, requested_variation: Option<&str>) -> Option<Result<Url, AnyError>> {
        try_get_tokenizer_url_from_env(self, requested_variation)
    }

    fn config_url(&self, requested_variation: Option<&str>) -> Option<Result<Url, AnyError>> {
        try_get_config_url_from_env(self, requested_variation)
    }

    fn run(
        &self,
        session: &ort::Session,
        tokenizer: Option<&tokenizers::Tokenizer>,
        config: Option<&Map<String, Value>>,
        input: &Self::Input,
        options: Option<&Self::InputOptions>,
    ) -> Result<Self::Output, AnyError> {
        let tokenizer = tokenizer.unwrap();
        let options = options
            .cloned()
            .unwrap_or(TokenClassificationOptions::default());

        let encodings = tokenizer.encode(input.to_owned(), true).unwrap();

        let input_ids = encodings
            .get_ids()
            .iter()
            .map(|v| i64::from(*v))
            .collect::<Vec<_>>();

        let attention_mask = encodings
            .get_attention_mask()
            .iter()
            .map(|v| i64::from(*v))
            .collect::<Vec<_>>();

        let token_type_ids = encodings
            .get_type_ids()
            .iter()
            .map(|v| i64::from(*v))
            .collect::<Vec<_>>();

        let input_ids_array = Array1::from_vec(input_ids.to_owned()).insert_axis(Axis(0));
        let attention_mask_array = Array1::from_vec(attention_mask.to_owned()).insert_axis(Axis(0));
        let token_type_ids_array = Array1::from_vec(token_type_ids.to_owned()).insert_axis(Axis(0));

        let outputs = session.run(inputs! {
            "input_ids" => input_ids_array,
            "token_type_ids" => token_type_ids_array,
            "attention_mask" => attention_mask_array,
        }?)?;

        let outputs = outputs["logits"].try_extract_tensor::<f32>()?;
        let outputs = outputs.into_dimensionality::<Ix3>().unwrap();
        let outputs = outputs.softmax(Axis(2));

        let labels = config.unwrap().get("id2label").unwrap();
        let vocab = tokenizer.get_added_vocabulary();

        let tokens = encodings.get_tokens();
        let offsets = encodings.get_offsets();

        let outputs = Zip::from(outputs.lanes(Axis(2))).map_collect(argmax);

        let outputs = outputs.map_axis(Axis(1), |row| {
            let classifications = row.iter().enumerate().filter_map(|(idx, predict)| {
                let token = tokens.get(idx).unwrap();

                if vocab.is_special_token(token) {
                    return None;
                }

                let entity = labels
                    .get(predict.0.to_string())
                    .map(|label| label.as_str().unwrap().into())
                    .unwrap();

                let (start, end) = offsets.get(idx).unwrap();

                Some(RawClassifiedToken {
                    index: idx,
                    score: predict.1,
                    entity,
                    word: token.to_owned(),
                    start: *start,
                    end: *end,
                })
            });

            match options.aggregation_strategy {
                TokenClassificationAggregationStrategy::None => {
                    TokenClassificationOutput::Raw(none_aggregation(classifications))
                }
                TokenClassificationAggregationStrategy::Simple => {
                    TokenClassificationOutput::Grouped(simple_aggregation(
                        classifications,
                        &input,
                        &options.ignore_labels,
                    ))
                }
                TokenClassificationAggregationStrategy::Max => TokenClassificationOutput::Grouped(
                    max_aggregation(classifications, &input, &options.ignore_labels),
                ),
            }
        });

        Ok(outputs.first().unwrap().to_owned())
    }
}

fn none_aggregation(iter: impl IntoIterator<Item = RawClassifiedToken>) -> Vec<RawClassifiedToken> {
    iter.into_iter().collect()
}

fn simple_aggregation(
    iter: impl IntoIterator<Item = RawClassifiedToken>,
    original_text: &str,
    ignore_labels: &[String],
) -> Vec<ClassifiedTokenGroup> {
    let mut iter = iter.into_iter().peekable();
    let mut result = vec![];

    loop {
        let Some(item) = iter.next() else {
            break;
        };

        let (_, label) = {
            if item.entity.is_char_boundary(2) {
                item.entity.split_at(2)
            } else {
                continue;
            }
        };

        if ignore_labels.iter().any(|to_ignore| to_ignore == label) {
            continue;
        }

        let child_label = format!("I-{label}");

        // Peeking take_while()
        let mut group = vec![item.to_owned()];
        loop {
            let Some(child) = iter.next_if(|other| other.entity == child_label) else {
                break;
            };

            group.push(child)
        }

        let start = item.start;
        let end = group.last().map_or(item.end, |last| last.end);
        let word = String::from(original_text.get(start..end).unwrap());

        // Apply mean score
        let score = group.iter().map(|i| i.score).sum::<f32>() / group.len() as f32;

        result.push(ClassifiedTokenGroup {
            group_entity: label.to_owned(),
            tokens: group,
            score,
            word,
            start,
            end,
        })
    }

    result
}

fn max_aggregation(
    iter: impl IntoIterator<Item = RawClassifiedToken>,
    original_text: &str,
    ignore_labels: &[String],
) -> Vec<ClassifiedTokenGroup> {
    let mut iter = iter.into_iter().peekable();
    let mut result = vec![];

    loop {
        let Some(item) = iter.next() else {
            break;
        };

        // TODO: rust >= v1.80: replace by split_at_checked
        let (_, label) = {
            if item.entity.is_char_boundary(2) {
                item.entity.split_at(2)
            } else {
                continue;
            }
        };

        if ignore_labels.iter().any(|to_ignore| to_ignore == label) {
            continue;
        }

        let child_label = format!("I-{label}");

        let start = item.start;
        let mut end = item.end;

        // Peeking take_while()
        let mut group = vec![item.to_owned()];

        loop {
            let Some(child) = iter.next_if(|other| {
                other.entity == child_label
                    || (other.start == end
                        && !ignore_labels
                            .iter()
                            .any(|to_ignore| to_ignore.contains(&other.entity)))
            }) else {
                break;
            };

            end = child.end;
            group.push(child);
        }

        // let end = group.last().map_or(item.end, |last| last.end);
        let word = String::from(original_text.get(start..end).unwrap());

        // Apply max score
        let max = group
            .iter()
            .max_by(|a, b| a.score.total_cmp(&b.score))
            .unwrap();

        // TODO: rust >= v1.80: replace by split_at_checked
        let group_entity = {
            if max.entity.is_char_boundary(2) {
                max.entity.split_at(2).1
            } else {
                label
            }
        };

        result.push(ClassifiedTokenGroup {
            group_entity: group_entity.to_owned(),
            tokens: group.to_owned(),
            score: max.score,
            word,
            start,
            end,
        })
    }

    result
}
