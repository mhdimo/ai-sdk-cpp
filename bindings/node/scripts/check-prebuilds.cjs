// Refuses to publish a package nobody could install.
//
// The failure this guards against is silent and total. `npm publish` is happy
// with no prebuilds/ directory, and every user then gets "No native build was
// found" from node-gyp-build. Nothing upstream catches it either, because the
// publisher's own machine has build/Release from `npm run build` and loads
// fine -- the package only breaks on machines that are not this one, which is
// all of them.
//
// Run by prepublishOnly. CI assembles prebuilds/ from the per-platform
// artifacts before publishing; this checks that assembly actually happened.

const fs = require('node:fs');
const path = require('node:path');

const DIR = path.join(__dirname, '..', 'prebuilds');

const SHARED_LIB = {
  darwin: 'libai_sdk.dylib',
  linux: 'libai_sdk.so',
  win32: 'ai_sdk.dll',
};

/** The addon and its shared library, both of which an install needs. */
function expectedFiles(tuple) {
  const platform = tuple.split('-')[0];
  const lib = SHARED_LIB[platform];
  if (!lib) return null; // a platform we have no recipe for
  return ['node.napi.node', lib];
}

if (!fs.existsSync(DIR)) {
  console.error(
    'refusing to publish: no prebuilds/ directory.\n' +
      'Run `npm run build:prebuild` for this platform, or publish from CI,\n' +
      'which builds every platform and assembles them here.'
  );
  process.exit(1);
}

const tuples = fs
  .readdirSync(DIR, { withFileTypes: true })
  .filter((e) => e.isDirectory())
  .map((e) => e.name);

if (tuples.length === 0) {
  console.error('refusing to publish: prebuilds/ is empty.');
  process.exit(1);
}

let failed = false;
for (const tuple of tuples) {
  const wanted = expectedFiles(tuple);
  if (!wanted) {
    console.error(`  ${tuple}: unrecognised platform, cannot check`);
    failed = true;
    continue;
  }
  const missing = wanted.filter((f) => !fs.existsSync(path.join(DIR, tuple, f)));
  if (missing.length > 0) {
    console.error(`  ${tuple}: missing ${missing.join(', ')}`);
    failed = true;
  } else {
    console.log(`  ${tuple}: ok`);
  }
}

if (failed) {
  console.error('refusing to publish: a prebuild is incomplete (see above).');
  process.exit(1);
}

console.log(`prebuilds ok: ${tuples.join(', ')}`);
