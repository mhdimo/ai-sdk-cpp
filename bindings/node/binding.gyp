{
  # gyp parses this file as a Python literal: comments are #, not //.
  "variables": {
    # Where libai_sdk is linked from. A source checkout uses the in-repo build.
    # The prebuild script sets AI_SDK_LIBDIR to a static-OpenSSL build instead,
    # so the shipped dylib has no Homebrew or distro path baked into it.
    "ai_sdk_libdir": "<!(node -e \"process.stdout.write(process.env.AI_SDK_LIBDIR || '<(module_root_dir)/../../build/bindings/c')\")",
    # The second rpath entry. A source checkout needs the in-repo path, because
    # there the dylib sits in build/bindings/c rather than beside the addon. In
    # a prebuild build that path would be the build machine's -- useless at the
    # user's end and nobody else's business -- so it degrades to a duplicate of
    # @loader_path, which is inert.
    "ai_sdk_extra_rpath": "<!(node -e \"process.stdout.write(process.env.AI_SDK_LIBDIR ? '@loader_path' : '<(module_root_dir)/../../build/bindings/c')\")"
  },
  "targets": [
    {
      "target_name": "ai_sdk_native",
      "cflags!": ["-fno-exceptions"],
      "cflags_cc!": ["-fno-exceptions"],
      # C++17 here against the core's C++20 is deliberate, not drift. This
      # target compiles one file, and that file includes <napi.h> and the C
      # header `ai_sdk.h` -- it never mentions ai::Task or any other C++20-only
      # core type, because it links libai_sdk and crosses the C ABI. So 17 costs
      # nothing, and it is what lets `node-gyp rebuild` succeed on a user's
      # machine with a toolchain whose C++20 support is partial (VS2019, older
      # Xcode). Raising this to 20 would narrow the set of machines that can
      # build the addon from source without buying anything.
      "cflags_cc": ["-std=c++17"],
      "sources": ["src/addon.cpp"],
      "include_dirs": [
        "<!@(node -p \"require('node-addon-api').include\")",
        "../c"
      ],
      "libraries": [
        "-L<(ai_sdk_libdir)",
        "-lai_sdk",
        # @loader_path is what makes a shipped addon relocatable: the dynamic
        # loader looks for libai_sdk in the same directory as the .node file,
        # wherever the package manager happened to put it.
        # Order matters to dyld, which takes the first rpath that resolves. The
        # specific location goes first so a source checkout always finds the
        # dylib it was built against, not some other copy that happens to be
        # beside the addon.
        "-Wl,-rpath,<(ai_sdk_extra_rpath)",
        "-Wl,-rpath,@loader_path"
      ],
      "defines": ["NAPI_DISABLE_CPP_EXCEPTIONS"],
      "conditions": [
        ["OS=='mac'", {
          "xcode_settings": {
            "GCC_ENABLE_CPP_EXCEPTIONS": "YES",
            "CLANG_CXX_LIBRARY": "libc++",
            "MACOSX_DEPLOYMENT_TARGET": "12.0",
            "OTHER_CPLUSPLUSFLAGS": ["-std=c++17"]
          }
        }],
        ["OS=='linux'", {
          "cflags_cc": ["-std=c++17", "-fexceptions"]
        }],
        ["OS=='win'", {
          "msvs_settings": {
            "VCCLCompilerTool": {
              "ExceptionHandling": 1,
              "AdditionalOptions": ["/std:c++17"]
            }
          }
        }]
      ]
    }
  ]
}
