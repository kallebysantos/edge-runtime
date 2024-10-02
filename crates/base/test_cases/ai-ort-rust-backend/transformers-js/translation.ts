import { assertEquals } from 'jsr:@std/assert';
import {
  env,
  pipeline,
} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0-alpha.20/dist/transformers.min.js';

// Ensure we do not use browser cache
env.useBrowserCache = false;
env.allowLocalModels = false;

const pipe = await pipeline('translation_en_to_de');

Deno.serve(async () => {
  const input = [
    'Hello, how are you?',
    'My name is Maria.',
  ];

  const output = await pipe(input);

  const expected = [
    { 'translation_text': 'Hallo, wie sind Sie?' },
    { 'translation_text': 'Mein Name ist Maria.' },
  ];

  assertEquals(output, expected);

  return new Response();
});
