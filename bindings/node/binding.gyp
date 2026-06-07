{
  "targets": [
    {
      "target_name": "ai_sdk_native",
      "cflags!": ["-fno-exceptions"],
      "cflags_cc!": ["-fno-exceptions"],
      "cflags_cc": ["-std=c++17"],
      "sources": ["src/addon.cpp"],
      "include_dirs": [
        "<!@(node -p \"require('node-addon-api').include\")",
        "../c"
      ],
      "libraries": [
        "-L<(module_root_dir)/../../build/bindings/c",
        "-lai_sdk",
        "-Wl,-rpath,@loader_path/../../build/bindings/c"
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
          "cflags_cc": ["-std=c++17", "-fexceptions"],
          "libraries": [
            "-Wl,-rpath,$ORIGIN/../../build/bindings/c"
          ]
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
