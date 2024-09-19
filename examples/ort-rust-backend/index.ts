import { env, pipeline } from '@xenova/transformers';

// Ensure we do not use browser cache
env.useBrowserCache = false;
env.allowLocalModels = false;

const pipe = await pipeline('sentiment-analysis');

Deno.serve(async (req: Request) => {
  const { input } = await req.json();

  const output = await pipe(input);
  return new Response(
    JSON.stringify(
      output,
    ),
    {
      headers: {
        'Content-Type': 'application/json',
        'Connection': 'keep-alive',
      },
    },
  );
});
