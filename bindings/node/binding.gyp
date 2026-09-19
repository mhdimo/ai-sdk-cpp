{
  # gyp parses this file as a Python literal: comments are #, not //.
  "variables": {
    # Where libai_sdk is linked from. A source checkout uses the in-repo build.
    # The prebuild script sets AI_SDK_LIBDIR to a static-OpenSSL build instead,
    # so the shipped dylib has no Homebrew or distro path baked into it.
    "ai_sdk_libdir": "<!(node -e \"process.stdout.write(process.env.AI_SDK_LIBDIR || '<(module_root_dir)/../../build/bindings/c')\")",
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
      # Every rpath lives in the per-OS conditions below, because the token is
      # not portable: dyld spells it @loader_path and the ELF loader spells it
      # $ORIGIN, and to the *other* loader each is just a relative directory
      # name that does not exist. Written here it links on both and resolves on
      # one, which is the kind of failure a build cannot report.
      "libraries": [
        "-L<(ai_sdk_libdir)",
        "-lai_sdk"
      ],
      "defines": ["NAPI_DISABLE_CPP_EXCEPTIONS"],
      "conditions": [
        ["OS=='mac'", {
          "libraries": [
            # The in-repo build tree, for a source checkout: there the dylib is
            # in build/bindings/c rather than beside the addon. First in the
            # list, because dyld takes the first entry that resolves and a
            # checkout should load the dylib it was built against.
            #
            # In a prebuild build this expands to *nothing* rather than to an
            # empty rpath argument. That distinction is the whole reason for
            # `<!@()`: an empty element in an rpath list is not ignored, it
            # means the current working directory, which is the loader offering
            # to take the library from wherever the process happens to be.
            "<!@(node -e \"process.stdout.write(process.env.AI_SDK_LIBDIR ? '' : '-Wl,-rpath,<(module_root_dir)/../../build/bindings/c')\")",
            "-Wl,-rpath,@loader_path"
          ],
          "xcode_settings": {
            "GCC_ENABLE_CPP_EXCEPTIONS": "YES",
            "CLANG_CXX_LIBRARY": "libc++",
            "MACOSX_DEPLOYMENT_TARGET": "12.0",
            "OTHER_CPLUSPLUSFLAGS": ["-std=c++17"]
          }
        }],
        ["OS=='linux'", {
          "libraries": [
            # The in-repo build tree, for a source checkout -- same reasoning as
            # the macOS branch above, and dropped entirely in a prebuild build
            # so that the shipped addon records one rpath and it is correct.
            "<!@(node -e \"process.stdout.write(process.env.AI_SDK_LIBDIR ? '' : '-Wl,-rpath,<(module_root_dir)/../../build/bindings/c')\")",
            # Quoted because the ELF loader, not the shell, has to expand it.
            # Both of those matter and both are easy to get wrong:
            #
            #   `$$` because this string travels through a Makefile on the way.
            #   make collapses `$$` to `$`, so the linker is handed $ORIGIN.
            #
            #   the single quotes because without them the *shell* sees `$O` as
            #   an unset variable and expands it to nothing. The flag then ends
            #   as a bare `-Wl,-rpath,`, which the linker records as an empty
            #   entry -- the current working directory. Measured, not reasoned:
            #   the generated Makefile carries both spellings, and only the
            #   quoted one ever reaches the linker intact.
            #
            # Nothing reports either mistake, because an addon with a broken
            # rpath still loads on the machine that built it.
            "-Wl,-rpath,'$$ORIGIN'"
          ],
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
