#include "graphics/shader/shaderFetchControl.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <map>
#include <optional>
#include <set>

namespace Libs::Graphics::ShaderRecompiler {
namespace {

using Decoder::Family;
using Decoder::Instruction;
using Decoder::Opcode;
using Decoder::Operand;
using Decoder::OperandKind;
using Known = std::optional<uint32_t>;
constexpr uint32_t SccCode = 253;

bool ScalarCode(const Operand& operand, uint32_t& code) {
	switch (operand.kind) {
		case OperandKind::Sgpr: code = operand.reg; return true;
		case OperandKind::VccLo: code = 106; return true;
		case OperandKind::VccHi: code = 107; return true;
		case OperandKind::M0: code = 124; return true;
		case OperandKind::ExecLo: code = 126; return true;
		case OperandKind::ExecHi: code = 127; return true;
		case OperandKind::Scc: code = SccCode; return true;
		default: return false;
	}
}

bool Plain(const Operand& operand) {
	return operand.sdwa_sel == 6 && !operand.sdwa_sext && !operand.op_sel &&
	       !operand.op_sel_hi && !operand.dpp && !operand.dpp8 &&
	       operand.omod == 0 && !operand.clamp;
}

bool PairDestination(const Instruction& inst) {
	if (inst.family == Family::SMEM) return inst.data_dwords == 2;
	if (inst.family == Family::VOPC || inst.family == Family::VOP3 ||
	    (inst.opcode >= Opcode::V_CMP_F_F32 && inst.opcode <= Opcode::V_CMPX_TRU_F64)) {
		return true;
	}
	switch (inst.opcode) {
		case Opcode::S_MOV_B64: case Opcode::S_CMOV_B64:
		case Opcode::S_BREV_B64: case Opcode::S_BITREPLICATE_B64_B32:
		case Opcode::S_QUADMASK_B64: case Opcode::S_GETPC_B64:
		case Opcode::S_SWAPPC_B64: case Opcode::S_NOT_B64:
		case Opcode::S_WQM_B64: case Opcode::S_AND_B64:
		case Opcode::S_ANDN2_B64: case Opcode::S_OR_B64:
		case Opcode::S_ORN2_B64: case Opcode::S_XOR_B64:
		case Opcode::S_NAND_B64: case Opcode::S_NOR_B64:
		case Opcode::S_XNOR_B64: case Opcode::S_LSHL_B64:
		case Opcode::S_LSHR_B64: case Opcode::S_ASHR_I64:
		case Opcode::S_BFE_U64: case Opcode::S_BFM_B64:
		case Opcode::S_CSELECT_B64: case Opcode::S_AND_SAVEEXEC_B64:
		case Opcode::S_ORN2_SAVEEXEC_B64: case Opcode::S_ANDN1_SAVEEXEC_B64:
		case Opcode::S_BITSET0_B64: case Opcode::S_BITSET1_B64: return true;
		default: return false;
	}
}

bool MayWriteScc(const Instruction& inst) {
	if (inst.family == Family::SOPC || inst.family == Family::SOP2) {
		return inst.opcode != Opcode::S_CSELECT_B32 &&
		       inst.opcode != Opcode::S_CSELECT_B64;
	}
	// Unknown scalar operations stop the query; MOV and these prolog operations
	// are documented not to change SCC.
	if (inst.family == Family::SOP1) {
		switch (inst.opcode) {
			case Opcode::S_MOV_B32: case Opcode::S_MOV_B64:
			case Opcode::S_CMOV_B64: case Opcode::S_GETPC_B64:
			case Opcode::S_SETPC_B64: case Opcode::S_SWAPPC_B64:
			case Opcode::S_BREV_B32: case Opcode::S_BREV_B64: return false;
			default: return true;
		}
	}
	return inst.family == Family::SOPK && inst.opcode != Opcode::S_MOVK_I32;
}

bool Writes(const Instruction& inst, uint32_t code) {
	if (code == SccCode) return MayWriteScc(inst);
	const auto matches = [&](const Operand& destination, uint32_t count) {
		uint32_t first = 0;
		return ScalarCode(destination, first) && code >= first && code - first < count;
	};
	const auto count = inst.family == Family::SMEM ? std::max(inst.data_dwords, 1u)
	                                            : PairDestination(inst) ? 2u : 1u;
	return matches(inst.dst, count) || matches(inst.dst2, 2u);
}

bool IsScalarLoad(Opcode opcode, bool& buffer) {
	switch (opcode) {
		case Opcode::S_LOAD_DWORD: case Opcode::S_LOAD_DWORDX2:
		case Opcode::S_LOAD_DWORDX4: case Opcode::S_LOAD_DWORDX8:
		case Opcode::S_LOAD_DWORDX16: buffer = false; return true;
		case Opcode::S_BUFFER_LOAD_DWORD: case Opcode::S_BUFFER_LOAD_DWORDX2:
		case Opcode::S_BUFFER_LOAD_DWORDX4: case Opcode::S_BUFFER_LOAD_DWORDX8:
		case Opcode::S_BUFFER_LOAD_DWORDX16: buffer = true; return true;
		default: return false;
	}
}

class ControlQuery {
public:
	ControlQuery(const Decoder::Program& decoded, std::span<const uint32_t> user_data,
	             uint32_t base, FetchControlMemoryReader reader, void* userdata,
	             uint32_t wave_size): decoded(decoded), user_data(user_data), base(base),
	    reader(reader), userdata(userdata), wave_size(wave_size),
	    predecessors(decoded.instructions.size()), successors(decoded.instructions.size()) {
		for (uint32_t index = 0; index < decoded.instructions.size(); ++index) {
			indices.emplace(decoded.instructions[index].pc, index);
		}
		for (uint32_t index = 0; index < decoded.instructions.size(); ++index) {
			const auto& inst = decoded.instructions[index];
			const auto edge = [&](uint32_t to) {
				if (to < decoded.instructions.size()) {
					successors[index].push_back(to);
					predecessors[to].push_back(index);
				}
			};
			if (Decoder::IsDirectBranch(inst.opcode)) {
				const auto target = indices.find(inst.branch_target);
				if (target != indices.end()) edge(target->second);
				if (Decoder::IsConditionalBranch(inst.opcode)) edge(index + 1u);
			} else if (inst.opcode != Opcode::S_ENDPGM &&
			           inst.opcode != Opcode::S_SETPC_B64 && inst.opcode != Opcode::S_SWAPPC_B64) {
				edge(index + 1u);
			}
		}
	}

	ShaderFetchControl Run() {
		std::vector<uint32_t> queue;
		std::vector<bool> visited(decoded.instructions.size());
		if (!decoded.instructions.empty()) queue.push_back(0);
		for (size_t cursor = 0; cursor < queue.size(); ++cursor) {
			const auto index = queue[cursor];
			if (visited[index]) continue;
			visited[index] = true;
			const auto& inst = decoded.instructions[index];
			const auto outcome = Branch(index);
			if (outcome) {
				result.decisions.emplace_back(inst.pc, *outcome != 0u);
				if (*outcome != 0u) {
					if (const auto target = indices.find(inst.branch_target); target != indices.end()) {
						queue.push_back(target->second);
					}
				} else if (index + 1u < decoded.instructions.size()) {
					queue.push_back(index + 1u);
				}
			} else {
				queue.insert(queue.end(), successors[index].begin(), successors[index].end());
			}
		}
		std::sort(result.decisions.begin(), result.decisions.end());
		std::sort(result.reads.begin(), result.reads.end());
		result.reads.erase(std::unique(result.reads.begin(), result.reads.end()), result.reads.end());
		// Unproved conditions stay dynamic and do not create host-specialized dependencies.
		if (result.decisions.empty()) result.reads.clear();
		return std::move(result);
	}

private:
	Known Branch(uint32_t before) {
		const auto opcode = decoded.instructions[before].opcode;
		if (opcode == Opcode::S_CBRANCH_SCC0 || opcode == Opcode::S_CBRANCH_SCC1) {
			const auto scc = Register(before, SccCode);
			if (scc) return uint32_t{(*scc != 0u) == (opcode == Opcode::S_CBRANCH_SCC1)};
		}
		if (opcode == Opcode::S_CBRANCH_VCCZ || opcode == Opcode::S_CBRANCH_VCCNZ) {
			const auto lo = Register(before, 106u);
			const auto hi = wave_size == 64u ? Register(before, 107u) : Known{0u};
			if (lo && hi) {
				return uint32_t{((*lo | *hi) == 0u) == (opcode == Opcode::S_CBRANCH_VCCZ)};
			}
		}
		return {};
	}

	Known Register(uint32_t before, uint32_t code) {
		const auto key = std::pair{before, code};
		if (const auto found = values.find(key); found != values.end()) return found->second;
		if (!visiting.insert(key).second) return {};
		const auto finish = [&](Known value) {
			visiting.erase(key);
			values.emplace(key, value);
			return value;
		};
		std::set<uint32_t> walked;
		uint32_t cursor = before;
		while (cursor < predecessors.size()) {
			if (cursor == 0u && !predecessors[0].empty()) return finish({});
			if (cursor == 0u) {
				return finish(code >= base && code - base < user_data.size()
				                  ? Known{user_data[code - base]} : Known{});
			}
			if (predecessors[cursor].size() != 1u) return finish({});
			cursor = predecessors[cursor].front();
			if (!walked.insert(cursor).second) return finish({});
			const auto& producer = decoded.instructions[cursor];
			if (Writes(producer, code)) return finish(Produce(cursor, code));
		}
		return finish({});
	}

	Known OperandValue(uint32_t before, const Operand& operand, uint32_t component = 0u) {
		if (!Plain(operand) || operand.negate || operand.absolute || operand.negate_hi ||
		    operand.clamp) return {};
		uint32_t code = 0;
		if (ScalarCode(operand, code)) return Register(before, code + component);
		switch (operand.kind) {
			case OperandKind::LiteralConstant: case OperandKind::IntegerInlineConstant:
			case OperandKind::FloatInlineConstant:
				return component == 0u ? operand.value
				                      : uint32_t{static_cast<int32_t>(operand.value) < 0 ? UINT32_MAX : 0};
			case OperandKind::Null: return 0u;
			default: return {};
		}
	}

	Known FloatValue(uint32_t before, const Operand& operand) {
		auto plain = operand;
		plain.negate = false;
		plain.absolute = false;
		const auto raw = OperandValue(before, plain);
		if (!raw || operand.negate_hi || operand.op_sel || operand.op_sel_hi) return {};
		uint32_t bits = *raw;
		if (operand.absolute) bits &= 0x7fffffffu;
		if (operand.negate) bits ^= 0x80000000u;
		// Comparing a denormal may depend on the shader's input flush mode. Retain it.
		if ((bits & 0x7fffffffu) != 0u && (bits & 0x7f800000u) == 0u) return {};
		return bits;
	}

	Known Read(uint32_t index, uint32_t component, bool buffer) {
		const auto& inst = decoded.instructions[index];
		const auto low = OperandValue(index, inst.src0);
		const auto high = OperandValue(index, inst.src0, 1);
		const auto offset = OperandValue(index, inst.src1);
		if (!low || !high || !offset || reader == nullptr) return {};
		constexpr uint64_t AddressMask = 0x0000ffffffffffffull;
		const auto address_base = ((uint64_t{*high} << 32u) | *low) & AddressMask;
		const int64_t immediate = static_cast<int32_t>(inst.offset) & ~int64_t{3};
		const int64_t relative = immediate + int64_t{*offset & ~3u} + int64_t{component} * 4;
		if (buffer) {
			const auto records = OperandValue(index, inst.src0, 2);
			const auto word3 = OperandValue(index, inst.src0, 3);
			if (!records || !word3 || relative < 0) return {};
			const uint64_t stride = (*high >> 16u) & 0x3fffu;
			const uint64_t size = stride == 0u ? *records : stride * *records;
			if (static_cast<uint64_t>(relative) > size || size - static_cast<uint64_t>(relative) < sizeof(uint32_t)) return {};
		}
		const auto aligned = address_base & ~uint64_t{3};
		if (relative < 0 && static_cast<uint64_t>(-relative) > aligned) return {};
		if (relative >= 0 && static_cast<uint64_t>(relative) > UINT64_MAX - aligned) return {};
		const uint64_t address = relative < 0 ? aligned - static_cast<uint64_t>(-relative)
		                                      : aligned + static_cast<uint64_t>(relative);
		uint32_t value = 0;
		if (!reader(userdata, address, {&value, 1u})) return {};
		result.reads.emplace_back(address, sizeof(value));
		return value;
	}

	Known Produce(uint32_t index, uint32_t code) {
		const auto& inst = decoded.instructions[index];
		uint32_t destination = 0;
		const bool has_destination = ScalarCode(inst.dst, destination);
		const auto component = has_destination && code >= destination ? code - destination : 0u;
		bool buffer = false;
		if (IsScalarLoad(inst.opcode, buffer)) return Read(index, component, buffer);
		const auto a = OperandValue(index, inst.src0);
		const auto b = OperandValue(index, inst.src1);
		if (code == SccCode) {
			if (!a || !b) return {};
			switch (inst.opcode) {
				case Opcode::S_CMP_EQ_I32: case Opcode::S_CMP_EQ_U32: return uint32_t{*a == *b};
				case Opcode::S_CMP_LG_I32: case Opcode::S_CMP_LG_U32: return uint32_t{*a != *b};
				case Opcode::S_CMP_GT_I32: return uint32_t{std::bit_cast<int32_t>(*a) > std::bit_cast<int32_t>(*b)};
				case Opcode::S_CMP_GE_I32: return uint32_t{std::bit_cast<int32_t>(*a) >= std::bit_cast<int32_t>(*b)};
				case Opcode::S_CMP_LT_I32: return uint32_t{std::bit_cast<int32_t>(*a) < std::bit_cast<int32_t>(*b)};
				case Opcode::S_CMP_LE_I32: return uint32_t{std::bit_cast<int32_t>(*a) <= std::bit_cast<int32_t>(*b)};
				case Opcode::S_CMP_GT_U32: return uint32_t{*a > *b};
				case Opcode::S_CMP_GE_U32: return uint32_t{*a >= *b};
				case Opcode::S_CMP_LT_U32: return uint32_t{*a < *b};
				case Opcode::S_CMP_LE_U32: return uint32_t{*a <= *b};
				case Opcode::S_AND_B32: return uint32_t{(*a & *b) != 0u};
				case Opcode::S_OR_B32: return uint32_t{(*a | *b) != 0u};
				case Opcode::S_XOR_B32: return uint32_t{(*a ^ *b) != 0u};
				case Opcode::S_ADD_U32: return uint32_t{uint64_t{*a} + *b > UINT32_MAX};
				case Opcode::S_SUB_U32: return uint32_t{*a >= *b};
				default: return {};
			}
		}
		if (inst.opcode == Opcode::S_MOV_B32 || inst.opcode == Opcode::S_MOVK_I32) return a;
		if (inst.opcode == Opcode::S_MOV_B64) return OperandValue(index, inst.src0, component);
		if (inst.opcode == Opcode::S_CSELECT_B32 || inst.opcode == Opcode::S_CSELECT_B64) {
			const auto scc = Register(index, SccCode);
			return scc ? OperandValue(index, *scc ? inst.src0 : inst.src1, component) : Known{};
		}
		if (inst.opcode >= Opcode::V_CMP_F_F32 && inst.opcode <= Opcode::V_CMP_TRU_F32) {
			const auto lhs_bits = FloatValue(index, inst.src0);
			const auto rhs_bits = FloatValue(index, inst.src1);
			if (!lhs_bits || !rhs_bits) return {};
			const float lhs = std::bit_cast<float>(*lhs_bits);
			const float rhs = std::bit_cast<float>(*rhs_bits);
			const bool ordered = !std::isnan(lhs) && !std::isnan(rhs);
			bool predicate = false;
			switch (inst.opcode) {
				case Opcode::V_CMP_F_F32: predicate = false; break;
				case Opcode::V_CMP_LT_F32: predicate = lhs < rhs; break;
				case Opcode::V_CMP_EQ_F32: predicate = lhs == rhs; break;
				case Opcode::V_CMP_LE_F32: predicate = lhs <= rhs; break;
				case Opcode::V_CMP_GT_F32: predicate = lhs > rhs; break;
				case Opcode::V_CMP_LG_F32: predicate = ordered && lhs != rhs; break;
				case Opcode::V_CMP_GE_F32: predicate = lhs >= rhs; break;
				case Opcode::V_CMP_O_F32: predicate = ordered; break;
				case Opcode::V_CMP_U_F32: predicate = !ordered; break;
				case Opcode::V_CMP_NGE_F32: predicate = !(lhs >= rhs); break;
				case Opcode::V_CMP_NLG_F32: predicate = !ordered || lhs == rhs; break;
				case Opcode::V_CMP_NGT_F32: predicate = !(lhs > rhs); break;
				case Opcode::V_CMP_NLE_F32: predicate = !(lhs <= rhs); break;
				case Opcode::V_CMP_NEQ_F32: predicate = lhs != rhs; break;
				case Opcode::V_CMP_NLT_F32: predicate = !(lhs < rhs); break;
				case Opcode::V_CMP_TRU_F32: predicate = true; break;
				default: return {};
			}
			// A false uniform comparison clears every mask bit for any EXEC. A true
			// comparison inherits EXEC, whose non-emptiness is not assumed here.
			return predicate ? Known{} : Known{0u};
		}
		if (!a || (!b && inst.opcode != Opcode::S_NOT_B32)) return {};
		switch (inst.opcode) {
			case Opcode::S_ADD_U32: case Opcode::S_ADD_I32: return *a + *b;
			case Opcode::S_SUB_U32: case Opcode::S_SUB_I32: return *a - *b;
			case Opcode::S_AND_B32: return *a & *b;
			case Opcode::S_ANDN2_B32: return *a & ~*b;
			case Opcode::S_OR_B32: return *a | *b;
			case Opcode::S_ORN2_B32: return *a | ~*b;
			case Opcode::S_XOR_B32: return *a ^ *b;
			case Opcode::S_NOT_B32: return ~*a;
			case Opcode::S_LSHL_B32: return *a << (*b & 31u);
			case Opcode::S_LSHR_B32: return *a >> (*b & 31u);
			case Opcode::S_MUL_I32: return *a * *b;
			case Opcode::S_LSHL1_ADD_U32: return (*a << 1u) + *b;
			case Opcode::S_LSHL2_ADD_U32: return (*a << 2u) + *b;
			case Opcode::S_LSHL3_ADD_U32: return (*a << 3u) + *b;
			case Opcode::S_LSHL4_ADD_U32: return (*a << 4u) + *b;
			case Opcode::S_BFE_U32: {
				const uint32_t shift = *b & 31u;
				const uint32_t width = (*b >> 16u) & 0x7fu;
				if (width == 0u) return 0u;
				if (width > 32u) return {};
				return (*a >> shift) & (width == 32u ? UINT32_MAX : (1u << width) - 1u);
			}
			default: return {};
		}
	}

	const Decoder::Program& decoded;
	std::span<const uint32_t> user_data;
	uint32_t base;
	FetchControlMemoryReader reader;
	void* userdata;
	uint32_t wave_size;
	std::map<uint32_t, uint32_t> indices;
	std::vector<std::vector<uint32_t>> predecessors;
	std::vector<std::vector<uint32_t>> successors;
	std::map<std::pair<uint32_t, uint32_t>, Known> values;
	std::set<std::pair<uint32_t, uint32_t>> visiting;
	ShaderFetchControl result;
};

} // namespace

ShaderFetchControl AnalyzeShaderFetchControl(
    const Decoder::Program& decoded, std::span<const uint32_t> user_data,
    uint32_t user_data_base, FetchControlMemoryReader reader, void* userdata,
    uint32_t wave_size) {
	// Most vertex shaders are straight-line fetch prologs. Avoid constructing a
	// reaching-definition graph when there is no branch this query can specialize.
	const bool has_supported_branch = std::any_of(
	    decoded.instructions.begin(), decoded.instructions.end(), [](const auto& inst) {
		    return inst.opcode == Opcode::S_CBRANCH_SCC0 ||
		           inst.opcode == Opcode::S_CBRANCH_SCC1 ||
		           inst.opcode == Opcode::S_CBRANCH_VCCZ ||
		           inst.opcode == Opcode::S_CBRANCH_VCCNZ;
	    });
	if (!has_supported_branch) return {};
	// Indirect control flow needs more than this conservative reaching-definition
	// query. Keep it entirely dynamic rather than guessing its possible targets.
	if (std::any_of(decoded.instructions.begin(), decoded.instructions.end(), [](const auto& inst) {
		    const auto op = inst.opcode;
		    const bool writes_global =
		        (op >= Opcode::BUFFER_STORE_FORMAT_X && op <= Opcode::BUFFER_STORE_FORMAT_D16_X) ||
		        (op >= Opcode::BUFFER_STORE_BYTE && op <= Opcode::BUFFER_STORE_DWORDX4) ||
		        (op >= Opcode::TBUFFER_STORE_FORMAT_X && op <= Opcode::BUFFER_ATOMIC_FMAX) ||
		        (op >= Opcode::FLAT_STORE_BYTE && op <= Opcode::FLAT_STORE_DWORDX4) ||
		        (op >= Opcode::IMAGE_STORE && op <= Opcode::IMAGE_ATOMIC_FMAX);
		    return writes_global || op == Opcode::S_SETPC_B64 || op == Opcode::S_SWAPPC_B64 ||
		           op == Opcode::UNSUPPORTED;
	    })) return {};
	return ControlQuery(decoded, user_data, user_data_base, reader, userdata, wave_size).Run();
}

Decoder::Program ApplyShaderFetchControl(const Decoder::Program& decoded,
                                        const ShaderFetchControl& control) {
	auto specialized = decoded;
	for (auto& inst: specialized.instructions) {
		if (!Decoder::IsConditionalBranch(inst.opcode)) continue;
		const auto decision = std::find_if(control.decisions.begin(), control.decisions.end(),
		    [&](const auto& entry) { return entry.first == inst.pc; });
		if (decision == control.decisions.end()) continue;
		inst.opcode = decision->second ? Opcode::S_BRANCH : Opcode::S_NOP;
		inst.family = Family::SOPP;
		inst.opcode_id = decision->second ? 2u : 0u;
		inst.raw[0] = decision->second
		    ? 0xbf820000u | (uint32_t{(inst.branch_target - inst.pc - 4u) / 4u} & 0xffffu)
		    : 0xbf800000u;
	}
	return specialized;
}

} // namespace Libs::Graphics::ShaderRecompiler
