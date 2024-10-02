import { assertAlmostEquals, assertEquals } from 'jsr:@std/assert';
import {
  env,
  pipeline,
} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0-alpha.20/dist/transformers.min.js';

// Ensure we do not use browser cache
env.useBrowserCache = false;
env.allowLocalModels = false;

const pipe = await pipeline('zero-shot-classification');

Deno.serve(async () => {
  const sequences_to_classify = 'I love making pizza';
  const candidate_labels = ['travel', 'cooking', 'dancing'];

  const output = await pipe(sequences_to_classify, candidate_labels);

  assertEquals(output.labels, ['cooking', 'travel', 'dancing']);

  [0.9986959357301496, 0.0007607675703779872, 0.0005432966994724542]
    .map((expected, idx) => {
      assertAlmostEquals(output.scores[idx], expected);
    });

  return new Response();
});
