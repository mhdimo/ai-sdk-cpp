// Vendored Boost.JSON implementation.
//
// Boost.JSON's non-inline symbols (boost::json::serialize, the object
// initializer_list constructor, ...) are inconsistently available across
// platforms: the system libboost_json links fine on macOS (Homebrew) but not on
// some Linux distros, producing undefined-reference errors at link time.
// Including <boost/json/src.hpp> in exactly one core translation unit compiles
// the entire Boost.JSON library into libai-sdk-core, so we never depend on a
// system libboost_json. Do NOT also link Boost::json (would be duplicate defs).
#include <boost/json/src.hpp>
