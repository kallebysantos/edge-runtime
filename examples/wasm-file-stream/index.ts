// import { add } from "./add-wasm/pkg/add_wasm.js";

import { readFile } from 'node:fs/promises';

Deno.serve(async (req) => {
  //const { a, b } = await req.json();
  //const result = add(a, b);
  /*
  const file = await Deno.readFile(
    new URL("./add_wasm_bg.wasm", import.meta.url),
  );
  const file = await fetch(
    new URL('./add_wasm_bg.wasm', import.meta.url),
  );
  const file = await readFile(
    new URL('./add_wasm_bg.wasm', import.meta.url),
  );
  */
  const file = await fetch(
    new URL('./add_wasm_bg.wasm', import.meta.url),
  );

  return Response.json({ result: true }, {
    headers: { 'Content-Type': 'application/json' },
  });
});
