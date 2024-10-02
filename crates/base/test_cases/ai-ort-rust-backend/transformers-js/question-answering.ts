import { assertAlmostEquals, assertEquals } from 'jsr:@std/assert';
import {
  env,
  pipeline,
} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0-alpha.20/dist/transformers.min.js';

// Ensure we do not use browser cache
env.useBrowserCache = false;
env.allowLocalModels = false;

const pipe = await pipeline('question-answering');

Deno.serve(async () => {
  const input = 'Who was Jim Henson?';
  const context = 'Jim Henson was a nice puppet.';

  const output = await pipe(input, context);

  assertEquals(output.answer, 'a nice puppet');
  assertAlmostEquals(output.score, 0.80401875204943);

  return new Response();
});
