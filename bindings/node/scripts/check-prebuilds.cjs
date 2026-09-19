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

// The platforms this release means to support, checked as a set.
//
// Checking only the directories that happen to be present answers a different
// question -- "is what arrived complete?" -- and answers it yes for a tarball
// that arrived with one platform in it. Such a tarball installs perfectly for
// whoever built it, and for everyone else `npm install` exits 1: install.cjs
// finds no matching prebuild and has no source fallback to fall back to. That
// is the failure this script's own header describes, and it does not catch it.
//
// Adding a platform here is a commitment: the matrix has to build it, and this
// will refuse to publish until it does. Windows has no recipe yet.
const SUPPORTED = ['darwin-arm64', 'darwin-x64', 'linux-arm64', 'linux-x64'];

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

// Completeness of what arrived, then whether what arrived is the whole set.
// Separate checks because they fail for different reasons -- an upload that was
// cut short, versus a matrix leg that never ran -- and the second is the one
// that decides whether the package is installable.
const absent = SUPPORTED.filter((t) => !tuples.includes(t));
if (absent.length > 0) {
  console.error(
    `refusing to publish: no prebuild for ${absent.join(', ')}.\n` +
      'Those platforms are claimed by this release, and a package without them\n' +
      'fails at install time there -- see the matrix in prebuilds.yml. If a\n' +
      'platform is being dropped, remove it from SUPPORTED deliberately.'
  );
  failed = true;
}

if (failed) {
  console.error('refusing to publish: a prebuild is missing or incomplete (see above).');
  process.exit(1);
}

console.log(`prebuilds ok: ${tuples.join(', ')}`);
