#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_FETCH_CONTROL_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_FETCH_CONTROL_H_

#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"

#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler {

using FetchControlMemoryReader = bool (*)(void*, uint64_t, std::span<uint32_t>);

struct ShaderFetchControl {
	// These are native branch outcomes, not guessed default attribute values.
	std::vector<std::pair<uint32_t, bool>> decisions;
	std::vector<std::pair<uint64_t, uint64_t>> reads;
};

// Reads only scalar dependencies needed to prove a native branch outcome.
// Unknown execution masks, divergent values, loops and conflicting definitions
// retain the original branch. The reader must observe current GPU-backed data.
ShaderFetchControl AnalyzeShaderFetchControl(
    const Decoder::Program& decoded, std::span<const uint32_t> user_data,
    uint32_t user_data_base, FetchControlMemoryReader reader, void* userdata = nullptr,
    uint32_t wave_size = 32);

// Preserve native PCs and code data while making proved branches unconditional.
// The ordinary CFG builder then removes their unreachable instructions.
Decoder::Program ApplyShaderFetchControl(const Decoder::Program& decoded,
                                        const ShaderFetchControl& control);

} // namespace Libs::Graphics::ShaderRecompiler

#endif
