/* Using default: Gte-small */
const pipe = new Supabase.ai.Pipeline('token-classification');

/* Using custom model
const pipe = new Supabase.ai.Pipeline(
	'feature-extraction',
	'paraphrase-multilingual-MiniLM-L12-v2',
);
*/

// Using different tasks
// const pipe = new Supabase.ai.Pipeline('sentiment-analysis');

Deno.serve(async (req: Request) => {
  const { input, options } = await req.json();

  const output = await pipe.run(input, options);
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
