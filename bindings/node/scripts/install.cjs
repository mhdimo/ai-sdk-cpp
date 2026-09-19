// Install step for the Node binding.
//
// npm runs `node-gyp rebuild` by default for any package containing a
// binding.gyp, and that default is wrong for this one. The addon links
// libai_sdk, which exists only in a source checkout that has already built the
// C++ library -- so a user installing from npm would watch a compiler fail on
// a file they do not have. Shipping prebuilds is the whole point of the
// package, and declaring an install script is the only way to stop npm from
// compiling instead.
//
// Three cases:
//
//   1. A source checkout. Build the addon if the C++ library is already there,
//      which is what npm used to do by default; if it is not, say so and
//      succeed anyway, because failing `npm install` on a fresh clone would
//      block a contributor before they have read the README.
//
//      This is checked before the prebuild, deliberately. A checkout can have
//      both -- make-prebuild.sh leaves prebuilds/ behind -- and node-gyp-build
//      prefers build/Release at require time. Deciding on the prebuild here
//      would leave a developer who is editing addon.cpp running a stale
//      binary from their last release build, with nothing to say so.
//
//   2. A prebuild matches this platform. Nothing to do -- every normal
//      install.
//
//   3. Neither. Fail loudly and name the platforms that do have prebuilds,
//      rather than letting the require() fail later with node-gyp-build's
//      platform list and no hint about what to do.

const fs = require('node:fs');
const path = require('node:path');
const { spawnSync } = require('node:child_process');

const ROOT = path.join(__dirname, '..');
const REPO = path.join(ROOT, '..', '..');

const tuple = `${process.platform}-${process.arch}`;
const prebuild = path.join(ROOT, 'prebuilds', tuple, 'node.napi.node');

/** The C++ library the addon links, in either its shared or static form. */
function findBuiltLibrary() {
  const dir = path.join(REPO, 'build', 'bindings', 'c');
  if (!fs.existsSync(dir)) return null;
  const wanted = ['libai_sdk.dylib', 'libai_sdk.so', 'ai_sdk.dll', 'libai_sdk.a'];
  for (const name of wanted) {
    const full = path.join(dir, name);
    if (fs.existsSync(full)) return full;
  }
  return null;
}

/** Platform tuples this package actually carries a prebuild for. */
function availablePlatforms() {
  const dir = path.join(ROOT, 'prebuilds');
  if (!fs.existsSync(dir)) return [];
  return fs
    .readdirSync(dir, { withFileTypes: true })
    .filter((e) => e.isDirectory())
    .map((e) => e.name);
}

const inSourceCheckout =
  fs.existsSync(path.join(ROOT, 'binding.gyp')) &&
  fs.existsSync(path.join(REPO, 'CMakeLists.txt'));

if (inSourceCheckout) {
  // Match what npm would have done on its own, but only when it can succeed.
  const lib = findBuiltLibrary();
  if (!lib) {
    console.log(
      'ai-sdk-cpp: the C++ library is not built yet, so the native addon was skipped.\n' +
        '  Build it, then the addon:\n' +
        '    cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j\n' +
        '    (cd bindings/node && npm run build)'
    );
    process.exit(0);
  }

  const gyp = spawnSync(
    process.execPath,
    [require.resolve('node-gyp/bin/node-gyp.js'), 'rebuild'],
    { cwd: ROOT, stdio: 'inherit' }
  );
  process.exit(gyp.status ?? 1);
}

if (fs.existsSync(prebuild)) {
  process.exit(0);
}

const have = availablePlatforms();
console.error(
  `\nai-sdk-cpp has no prebuilt binary for ${tuple}.\n\n` +
    (have.length
      ? `Prebuilds are available for: ${have.join(', ')}\n\n`
      : 'This package was published without any prebuilds, which is a packaging bug.\n\n') +
    'To build it for this platform, install from source:\n' +
    '  git clone https://github.com/mhdimo/ai-sdk-cpp && cd ai-sdk-cpp/bindings/node\n'
);
process.exit(1);
