declare namespace Supabase {
  /**
   * Provides AI related APIs
   */
  export interface Ai {
    readonly Pipeline: typeof Pipeline;
  }

  /**
   * Provides AI related APIs
   */
  export const ai: Ai;

  /** Parameters specific to feature extraction pipelines. */
  export type FeatureExtractionOptions = {
    /** Pool embeddings by taking their mean. */
    mean_pool?: boolean;
    /** Whether or not to normalize the embeddings in the last dimension. */
    normalize?: boolean;
  };

  /** The features computed by the model.*/
  export type FeatureExtractionOutput = number[];

  /** The classifications computed by the model.*/
  export type TextClassificationOutput = {
    /** The label predicted. */
    label: string;
    /** The corresponding probability. */
    score: number;
  };

  export type ZeroShotClassificationInput = {
    /** The text to classify.*/
    text: string;
    /** The set of possible class labels to classify.*/
    candidate_labels: string[];
  };

  /** Parameters specific to zero-shot classification pipelines.*/
  export type ZeroShotClassificationOptions = {
    /**
     * Whether or not multiple candidate labels can be true.
     * If `false`, the scores are normalized such that the sum of the label likelihoods for each sequence is 1.
     * If `true`, the labels are considered independent and probabilities are normalized for each candidate by doing a softmax
     * of the entailment score vs. the contradiction score.
     */
    multi_label?: boolean;
    /**
     * The template used to turn each label into an NLI-style hypothesis.
     * This template must include a `{}` or similar syntax for the candidate label to be inserted into the template.
     * For example, the default template is `"This example is {}."` With the candidate label `"sports"`, this would be fed into the model like `"<cls> sequence to classify <sep> This example is sports . <sep>".`
     * The default template works well in many cases, but it may be worthwhile to experiment with different templates depending on their task setting.
     */
    hypothesis_template?: string;
  };

  /** Parameters specific to token classification pipelines.*/
  export type TokenClassificationOptions = {
    /**  The strategy to fuse (or not) tokens based on the model prediction.
     *
     * - `'none'`: Will simply not do any aggregation and simply return raw results from the model.
     * - `'simple'`: Will attempt to group entities following the default schema:
     *      (A, B-TAG), (B, I-TAG), (C, I-TAG), (D, B-TAG2) (E, B-TAG2) will end up being
     *      ``[{“word”: ABC, “entity”: “TAG”}, {“word”: “D”, “entity”: “TAG2”}, {“word”: “E”, “entity”: “TAG2”}]``.
     *      Notice that two consecutive B tags will end up as different entities. On word based languages, we might end up splitting words undesirably:
     *      Imagine Microsoft being tagged as [{“word”: “Micro”, “entity”: “ENTERPRISE”}, {“word”: “soft”, “entity”: “NAME”}].
     *      Look for `MAX` to mitigate that and disambiguate words (on languages that support that meaning, which is basically tokens separated by a space).
     *      These mitigations will only work on real words, “New york” might still be tagged with two different entities.
     *
     * - `'max'`: (works only on word based models) Will use the `SIMPLE` strategy except that words, cannot end up with different tags. Word entity will simply be the token with the maximum score.
     *
     * @default 'simple'
     */
    aggregation_strategy?: 'none' | 'simple' | 'max';
    /**  A list of labels to ignore.
     *
     * @default ['O']
     */
    ignore_labels?: string[];
  };

  /** The raw token classification computed by the model.*/
  export type RawTokenClassificationOutput = {
    /** The entity label predicted for that token/word. */
    entity: string;
    /** The corresponding probability. */
    score: number;
    /** The index of the corresponding token in the sentence. */
    index: number;
    /** The token/word classified. This is obtained by decoding the selected tokens. If you want to have the exact string in the original sentence, use start and end. */
    word: string;
    /** The index of the start of the corresponding entity in the sentence. */
    start: number;
    /** The index of the end of the corresponding entity in the sentence. */
    end: number;
  };

  /** A grouped version of the classified tokens after applying `aggregation_strategy`. */
  export type GroupedTokenClassificationOutput = {
    /** The entity label predicted for that token/word based on the selected `aggregation_strategy`. */
    group_entity: string;
    /** The corresponding probability based on the selected `aggregation_strategy`. */
    score: number;
    /** The exact string from original sentence after grouping. */
    word: string;
    /** The index of the start of the corresponding entity in the sentence. */
    start: number;
    /** The index of the end of the corresponding entity in the sentence. */
    end: number;
    /** A list of all grouped tokens in their raw version. */
    tokens: RawTokenClassificationOutput[];
  };

  type PipelineTasks =
    | 'feature-extraction'
    | 'supabase-gte'
    | 'gte-small'
    | 'text-classification'
    | 'sentiment-analysis'
    | 'zero-shot-classification'
    | 'token-classification';

  /**
   * Pipelines provide a high-level, easy to use, API for running machine learning models.
   */
  export class Pipeline<K extends PipelineTasks> {
    /**
     * Create a new pipeline using given task
     * @param task The task of the pipeline.
     * @param variant Witch model variant to use by the pipeline.
     */
    constructor(task: K, variant?: string);

    /**
     * {@label pipeline-feature-extraction}
     *
     * Feature extraction pipeline using no model head. This pipeline extracts the hidden
     * states from the base transformer, which can be used as features in downstream tasks.
     *
     * **Example:** Instantiate pipeline using the `Pipeline` class.
     * ```javascript
     * const extractor = new Supabase.ai.Pipeline('feature-extraction');
     * const output = await extractor('This is a simple test.');
     *
     * // output: [0.05939, 0.02165, ...]
     *
     * ```
     *
     * **Example:** Batch inference, processing multiples in parallel
     * ```javascript
     * const extractor = new Supabase.ai.Pipeline('feature-extraction');
     * const output = await extractor(["I'd use Supabase in all of my projects", "Just a test for embedding"]);
     *
     * // output: [[0.07399, 0.01462, ...], [-0.08963, 0.01234, ...]]
     *
     * ```
     */
    run<I extends string | string[]>(
      input: K extends 'feature-extraction' ? I : never,
      opts?: FeatureExtractionOptions,
    ): Promise<I extends string[] ? FeatureExtractionOutput[] : FeatureExtractionOutput>;

    /**
     * Feature extraction pipeline using no model head.
     * This pipeline does the same as `'feature-extraction'` but using the `'Supabase/gte-small'` as default model.
     *
     * {@link run:pipeline-feature-extraction}
     */
    run<I extends string | string[]>(
      input: K extends 'supabase-gte' | 'gte-small' ? I : never,
      opts?: FeatureExtractionOptions,
    ): Promise<I extends string[] ? FeatureExtractionOutput[] : FeatureExtractionOutput>;

    /**
     * Text classification pipeline using any `ModelForSequenceClassification`.
     *
     * **Example:** Instantiate pipeline using the `Pipeline` class.
     * ```javascript
     * const classifier = new Supabase.ai.Pipeline('text-classification');
     * const output = await classifier('I love Supabase!');
     *
     * // output: {label: 'POSITIVE', score: 1.00}
     *
     * ```
     *
     * **Example:** Batch inference, processing multiples in parallel
     * ```javascript
     * const classifier = new Supabase.ai.Pipeline('sentiment-analysis');
     * const output = await classifier(['Cats are fun', 'Java is annoying']);
     *
     * // output: [{label: 'POSITIVE', score: 0.99 }, {label: 'NEGATIVE', score: 0.97}]
     *
     * ```
     */
    run<I extends string | string[]>(
      input: K extends 'text-classification' | 'sentiment-analysis' ? I : never,
    ): Promise<I extends string[] ? TextClassificationOutput[] : TextClassificationOutput>;

    /**
     * NLI-based zero-shot classification pipeline using a `ModelForSequenceClassification`.
     * Equivalent of `text-classification` pipelines, but these models don’t require a hardcoded number of potential classes, they can be chosen at runtime. It usually means it’s slower but it is much more flexible.
     *
     * **Example:** Instantiate pipeline using the `Pipeline` class.
     * ```javascript
     * const classifier = new Supabase.ai.Pipeline('zero-shot-classification');
     * const output = await classifier({
     *  text: 'one day I will see the world',
     *  candidate_labels: ['travel', 'cooking', 'exploration']
     * });
     *
     * // output: [{label: "travel", score: 0.797}, {label: "exploration", score: 0.199}, {label: "cooking", score: 0.002}]
     *
     * ```
     *
     * **Example:** Handling multiple correct labels
     * ```javascript
     * const classifier = new Supabase.ai.Pipeline('zero-shot-classification');
     * const input = {
     *  text: 'one day I will see the world',
     *  candidate_labels: ['travel', 'cooking', 'exploration']
     * };
     * const output = await classifier(input, { multi_label: true });
     *
     * // output: [{label: "travel", score: 0.994}, {label: "exploration", score: 0.938}, {label: "cooking", score: 0.001}]
     *
     * ```
     *
     * **Example:** Custom hypothesis template
     * ```javascript
     * const classifier = new Supabase.ai.Pipeline('zero-shot-classification');
     * const input = {
     *  text: 'one day I will see the world',
     *  candidate_labels: ['travel', 'cooking', 'exploration']
     * };
     * const output = await classifier(input, { hypothesis_template: "This example is NOT about {}");
     *
     * // output: [{label: "cooking", score: 0.47}, {label: "exploration", score: 0.26}, {label: "travel", score: 0.26}]
     *
     * ```
     */
    run<I extends ZeroShotClassificationInput>(
      input: K extends 'zero-shot-classification' ? I : never,
      opts: ZeroShotClassificationOptions,
    ): Promise<TextClassificationOutput[]>;

    /**
     * Named Entity Recognition pipeline using any `ModelForTokenClassification`
     *
     * **Example:** Instantiate pipeline using the `Pipeline` class.
     * ```javascript
     * const classifier = new Supabase.ai.Pipeline('token-classification');
     * const output = await classifier("My name is Kalleby and I'm from Brazil.");
     *
     * // output: [
     *    {group_entity: "PER", score: 0.99, word: "Kalleby", start: 11, end: 15, tokens: [...]},
     *    {group_entity: "LOC", score: 0.99, word: "Brazil", start: 32, end: 38, tokens: [...]}
     *  ]
     *
     * ```
     *
     * **Example:** Getting the raw outputs
     * ```javascript
     * const classifier = new Supabase.ai.Pipeline('token-classification');
     * const output = await classifier("My name is Kalleby and I'm from Brazil.", { aggregation_strategy: 'none'});
     *
     * // output: [
     *    {entity: "O", score: 0.99, index: 1, word: "My", start: 0, end: 2},
     *    {entity: "O", score: 0.99, index: 2, word: "name", start: 3, end: 7},
     *    {entity: "O", score: 0.99, index: 3, word: "is", start: 8, end: 10},
     *    {entity: "B-PER", score: 0.99, index: 4, word: "kalle", start: 11, end: 16},
     *    {entity: "I-PER", score: 0.99, index: 5, word: "##by", start: 16, end: 18},
     *    ...
     *    {entity: "LOC", score: 0.99, index: 11, word: "Brazil", start: 32, end: 38}
     *  ]
     *
     * ```
     */
    run<I extends string, O extends TokenClassificationOptions>(
      input: K extends 'token-classification' ? I : never,
      opts?: O,
    ): Promise<
      O extends { aggregation_strategy: 'None' } ? RawTokenClassificationOutput[]
        : GroupedTokenClassificationOutput[]
    >;
  }
}
