#pragma once

// The version the runtime reports — ai_sdk_version() in the C API, the MCP
// clientInfo handshake, and anything else that has to tell a peer what it is
// talking to.
//
// The build defines AI_SDK_VERSION from the CMake project version
// (CMakeLists.txt), so the native string is derived rather than retyped. The
// fallback only applies outside that build system.
#ifndef AI_SDK_VERSION
#define AI_SDK_VERSION "0.0.0-unknown"
#endif

namespace ai {

constexpr const char* version() noexcept { return AI_SDK_VERSION; }

} // namespace ai
