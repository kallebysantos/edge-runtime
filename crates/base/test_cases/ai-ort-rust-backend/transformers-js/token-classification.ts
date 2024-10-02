import { assertAlmostEquals, assertEquals } from 'jsr:@std/assert';
import {
  env,
  pipeline,
} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0-alpha.20/dist/transformers.min.js';

// Ensure we do not use browser cache
env.useBrowserCache = false;
env.allowLocalModels = false;

const pipe = await pipeline('token-classification');

Deno.serve(async () => {
  const input = "My name is Kalleby and I'm from Brazil.";

  const output = await pipe(input);

  assertEquals(output.length, 3);

  [
    {
      entity: 'B-PER',
      score: 0.9937819242477417,
      word: 'Kalle',
    },
    {
      entity: 'I-PER',
      score: 0.9965004920959473,
      word: '##by',
    },
    {
      entity: 'B-LOC',
      score: 0.9998262524604797,
      word: 'Brazil',
    },
  ].map((expected, idx) => {
    assertEquals(output[idx].entity, expected.entity);
    assertAlmostEquals(output[idx].score, expected.score);
    assertEquals(output[idx].word, expected.word);
  });

  return new Response();
});
