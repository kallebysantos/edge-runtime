import { assertEquals, assertAlmostEquals } from 'jsr:@std/assert';
import {
  env,
  pipeline,
} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0-alpha.20/dist/transformers.min.js';

// Ensure we do not use browser cache
env.useBrowserCache = false;
env.allowLocalModels = false;

const pipe = await pipeline('feature-extraction', 'supabase/gte-small'); // 384 dims model

Deno.serve(async () => {
  const input = [
    'This framework generates embeddings for each input sentence',
    'Sentences are passed as a list of string.',
    'The quick brown fox jumps over the lazy dog.',
  ];

  const output = await pipe(input, { pooling: 'mean', normalize: true });

  assertEquals(output.size, 3 * 384);
  assertEquals(output.dims.length, 2);

  // Comparing first 3 predictions
  [-0.050141554325819016, -0.00658650230616331, 0.0057992227375507355]
    .map((expected, idx) => {
      assertAlmostEquals(output.data[idx], expected);
    });

  return new Response();
});
