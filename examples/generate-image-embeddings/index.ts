import {
  Gravity,
  ImageMagick,
  initializeImageMagick,
  MagickColors,
  MagickFormat,
  MagickGeometry,
} from 'npm:@imagemagick/magick-wasm@0.0.30';
import { AutoProcessor, CLIPVisionModelWithProjection, env, RawImage } from '@xenova/transformers';

// Ensure we do not use browser cache
env.useBrowserCache = false;
env.allowLocalModels = false;

const wasmBytes = await Deno.readFile(
  new URL(
    'magick.wasm',
    import.meta.resolve('npm:@imagemagick/magick-wasm@0.0.30'),
  ),
);

await initializeImageMagick(
  wasmBytes,
);

// Load processor and vision model
const processor = await AutoProcessor.from_pretrained('Xenova/clip-vit-base-patch32');
const model = await CLIPVisionModelWithProjection.from_pretrained(
  'Xenova/clip-vit-base-patch32',
);

export async function fetchImage(url: string) {
  const imageRes = await fetch(new URL(url));
  const imageBlob = await imageRes.blob();
  const buffer = await imageBlob.arrayBuffer();

  return new Uint8Array(buffer);
}

Deno.serve(async () => {
  const image = await fetchImage(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png',
  );

  const rawImg = ImageMagick.read(image, (img) => {
    const { width, height } = processor.feature_extractor.crop_size;
    // We need to resize to fit model dims
    // https://legacy.imagemagick.org/Usage/resize/#space_fill
    img.resize(new MagickGeometry(width, height));
    img.extent(new MagickGeometry(width, height), Gravity.Center, MagickColors.Transparent);

    return img
      .write(
        MagickFormat.Png,
        (buffer) =>
          new RawImage(
            buffer,
            img.width,
            img.height,
            img.channels.length,
          ),
      );
  });

  // Disable pre-processor transformations
  processor.feature_extractor.do_resize = false;
  processor.feature_extractor.do_rescale = false;
  processor.feature_extractor.do_center_crop = false;

  const inputs = await processor(rawImg);
  const output = await model(inputs);

  //return new Response(rawImg, { headers: { 'Content-Type': 'image/png' } });
  return new Response(JSON.stringify(output));
});
