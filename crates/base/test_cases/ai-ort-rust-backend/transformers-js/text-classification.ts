
import { assertAlmostEquals, assertEquals } from 'jsr:@std/assert';
import {
  env,
  pipeline,
} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0-alpha.20/dist/transformers.min.js';

// Ensure we do not use browser cache
env.useBrowserCache = false;
env.allowLocalModels = false;

const pipe = await pipeline('text-classification');

Deno.serve(async () => {
  const input = [
    'I love supabase',
    'I hated the movie',
  ];

  const output = await pipe(input);

  assertEquals(output.length, 2);

  [
    { label: 'POSITIVE', score: 0.9993272423744202 },
    { label: 'NEGATIVE', score: 0.9997110962867737 },
  ].map((expected, idx) => {
    assertEquals(output[idx].label, expected.label);
    assertAlmostEquals(output[idx].score, expected.score);
  });

  return new Response();
});
