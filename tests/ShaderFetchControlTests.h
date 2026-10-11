// Included inside the shader test namespace, after its encoding/compilation helpers.
void CheckShaderFetchControl() {
  using ShaderRecompiler::AnalyzeShaderFetchControl;
  using ShaderRecompiler::ApplyShaderFetchControl;
  using ShaderRecompiler::ShaderFetchControl;
  constexpr const char *name = "ConditionalEmbeddedVertexFetch";
  constexpr uint64_t flag_address = 0x1058;
  constexpr uint64_t fifth_descriptor = 0x2040;
  struct Memory {
    std::vector<u32> words = std::vector<u32>(0x4000 / sizeof(u32));
    std::vector<std::pair<uint64_t, uint64_t>> observed;
    bool flag_readable = true;
    bool fifth_mapped = false;
    uint32_t fifth_reads = 0;
  } memory;
  const auto Read = +[](void *userdata, uint64_t address,
                        std::span<u32> values) -> bool {
    auto &memory = *static_cast<Memory *>(userdata);
    const auto end = address + values.size_bytes();
    memory.observed.emplace_back(address, values.size_bytes());
    if (address <= 0x1058 && end > 0x1058 && !memory.flag_readable) return false;
    if (address < 0x2050 && end > 0x2040) {
      ++memory.fifth_reads;
      if (!memory.fifth_mapped) return false;
    }
    if (address % sizeof(u32) != 0 || end < address ||
        end > memory.words.size() * sizeof(u32)) return false;
    std::copy_n(memory.words.begin() + address / sizeof(u32), values.size(), values.begin());
    return true;
  };

  std::array<u32, 20> user_data{};
  // Hardware s[12:15] is a native constant-buffer descriptor; s[24:25] and
  // s[26:27] contain the buffer and attribute tables, as in the rescue shader.
  user_data[4] = 0x1000;
  user_data[6] = 128;
  user_data[7] = BufferFormat(Prospero::BufferFormat::k32Float) << 12u | DstSel(4, 4, 4, 4);
  user_data[16] = 0x2000;
  user_data[18] = 0x3000;
  for (u32 i = 0; i < 5; ++i) {
    memory.words[0x3000 / sizeof(u32) + i] = i;
    memory.words[0x2000 / sizeof(u32) + i * 4] = 0x3600;
    memory.words[0x2000 / sizeof(u32) + i * 4 + 1] = 16u << 16u;
    memory.words[0x2000 / sizeof(u32) + i * 4 + 2] = 1;
    memory.words[0x2000 / sizeof(u32) + i * 4 + 3] =
        BufferFormat(Prospero::BufferFormat::k32_32_32_32Float) << 12u | DstSel(4, 5, 6, 7);
  }

  std::vector<u32> code;
  code.push_back(EncodeSMovB32(0, InlineU32(0)));
  code.push_back(EncodeVop2(0x01u, 0, Vgpr(8), 5));
  for (u32 i = 0; i < 4; ++i) {
    code.push_back(EncodeSmem0(0x02u, 4, 12));
    code.push_back(EncodeSmem1(i * 16));
    code.push_back(EncodeMubuf0(0x03u, 0, true, false));
    code.push_back(EncodeMubuf1(9, 1, 0));
    code.push_back(EncodeExp0(0x20u + i, 0xfu, false));
    code.push_back(EncodeExp1(9, 10, 11, 12));
  }
  code.push_back(EncodeSmem0(0x09u, 44, 6));
  code.push_back(EncodeSmem1(84));
  const auto compare_word = code.size();
  // VOP3 permits a scalar second source and an explicit scalar comparison dst.
  AppendVop3(&code, 0x0du, 62, InlineU32(0), 45);
  code.push_back(EncodeSop1(0x04u, 106, 62));
  const auto branch_word = code.size();
  code.push_back(EncodeSopp(0x06u));
  code.push_back(EncodeSmem0(0x00u, 28, 13));
  code.push_back(EncodeSmem1(16));
  code.push_back(EncodeSmem0(0x02u, 8, 12));
  code.push_back(EncodeSmem1(64));
  code.push_back(EncodeMubuf0(0x02u, 0, true, false));
  code.push_back(EncodeMubuf1(4, 2, 0));
  code.push_back(EncodeExp0(0x24u, 0x7u, false));
  code.push_back(EncodeExp1(4, 5, 6, 6));
  const auto join_word = code.size();
  code[branch_word] = EncodeSopp(0x06u, static_cast<u32>(join_word - branch_word - 1));
  AppendVMovU32(&code, 40, 0);
  AppendVMovU32(&code, 41, 0);
  AppendVMovU32(&code, 42, 0);
  AppendVMovU32(&code, 43, 0x3f800000u);
  code.push_back(EncodeExp0(0x0cu, 0xfu));
  code.push_back(EncodeExp1(40, 41, 42, 43));
  AppendEnd(&code);
  ShaderRecompiler::Decoder::Program decoded;
  ShaderRecompiler::Decoder::DecodeProgram(code, decoded);
  const auto branch_pc = static_cast<u32>(branch_word * sizeof(u32));
  Require(name, "native compare encoding",
          std::ranges::any_of(decoded.instructions, [&](const auto &inst) {
            return inst.pc == compare_word * sizeof(u32) &&
                   inst.opcode == ShaderRecompiler::Decoder::Opcode::V_CMP_NEQ_F32 &&
                   inst.dst.reg == 62 && inst.src1.reg == 45;
          }), "regression must encode the rescue's scalar-source vector comparison");

  ShaderVertexInputInfo vertex{};
  vertex.fetch_embedded = true;
  vertex.fetch_buffer_reg = 16;
  vertex.fetch_attrib_reg = 18;
  vertex.resources_num = 5;
  for (u32 i = 0; i < 5; ++i) {
    vertex.resources[i].fields[3] =
        BufferFormat(Prospero::BufferFormat::k32_32_32_32Float) << 12u | DstSel(4, 5, 6, 7);
    vertex.resources_dst[i].attr_id = static_cast<int>(i);
    vertex.resources_dst[i].registers_num = i == 4 ? 3 : 4;
  }

  const auto Analyze = [&] {
    memory.observed.clear();
    memory.fifth_reads = 0;
    return AnalyzeShaderFetchControl(decoded, user_data, 8, Read, &memory, 32);
  };
  const auto HasDecision = [&](const ShaderFetchControl &control, bool taken) {
    return std::ranges::find(control.decisions, std::pair{branch_pc, taken}) != control.decisions.end();
  };
  const auto Attributes = [&](const ShaderFetchControl &control) {
    const auto specialized = ApplyShaderFetchControl(decoded, control);
    return ShaderRecompiler::GetEmbeddedVertexFetchAttributes(
        specialized, vertex, 8, static_cast<u32>(user_data.size()));
  };
  const auto Compile = [&](const ShaderFetchControl &control, u32 required) {
    auto input = vertex;
    input.wave_size = 32;
    input.fetch_branches = control.decisions;
    input.fetch_reads = control.reads;
    input.resources_num = static_cast<int>(required);
    ShaderRecompiler::CompileOptions options;
    options.stage = ShaderType::Vertex;
    options.wave_size = 32;
    options.user_data_base = 8;
    options.user_data = user_data;
    options.input_info.vertex = &input;
    auto translated = ShaderRecompiler::TranslateProgram(code, options);
    auto plan = ShaderRecompiler::IR::ExtractResourcePlan(translated.program);
    ShaderRecompiler::IR::ResourceSnapshot snapshot;
    ShaderRecompiler::IR::ResourceSpecialization specialization;
    const ShaderRecompiler::IR::SrtRuntime runtime{
        .user_data = user_data,
        .shader_base = reinterpret_cast<uint64_t>(code.data()),
        .read_memory = Read,
        .userdata = &memory,
        .read_specialization_memory = Read};
    Require(name, "resource materialization",
            ShaderRecompiler::IR::MaterializeResources(plan, runtime, snapshot, specialization),
            "specialized fetch resources could not be materialized");
    auto result = ShaderRecompiler::CompileProgram(std::move(translated), options, specialization);
    ValidateSpirv(name, result.spirv);
    u32 parameters = 0;
    for (const auto &input : result.program.info.inputs) {
      if (input.kind == ShaderRecompiler::IR::StageInputKind::Parameter) {
        ++parameters;
        Require(name, "compact Vulkan location", input.location < required,
                "inactive vertex semantic leaked into the native input interface");
      }
    }
    Require(name, "live vertex interface", parameters == required,
            "SPIR-V must retain exactly the reachable embedded vertex fetches");
    return result.spirv;
  };

  memory.words[flag_address / sizeof(u32)] = 0;
  const auto inactive = Analyze();
  Require(name, "zero disables fifth fetch", HasDecision(inactive, true) &&
              Attributes(inactive) == std::vector<int>({0, 1, 2, 3}),
          "zero scalar flag must skip the conditional fifth vertex fetch");
  Require(name, "tracked uniform flag",
          std::ranges::any_of(inactive.reads, [&](const auto &read) {
            return read.first <= flag_address && read.first + read.second > flag_address;
          }), "the flag used to specialize a branch must be tracked for draw hazards");
  Require(name, "unmapped inactive descriptor", memory.fifth_reads == 0,
          "branch analysis eagerly accessed the absent fifth buffer table entry");
  Compile(inactive, 4);
  Require(name, "unmapped descriptor survives compilation", memory.fifth_reads == 0,
          "compilation accessed an inactive fifth buffer table entry");

  // Recorded mapped garbage and an unmapped address must have identical behavior
  // when the game takes the branch around the fetch; no descriptor normalization.
  memory.fifth_mapped = true;
  const std::array<u32, 4> garbage{0xc152c280u, 0x00000002u, 0x000000a0u, 0x000000a0u};
  std::copy(garbage.begin(), garbage.end(), memory.words.begin() + fifth_descriptor / sizeof(u32));
  vertex.resources[4] = ShaderBufferResource{{garbage[0], garbage[1], garbage[2], garbage[3]}};
  const auto mapped_inactive = Analyze();
  Require(name, "mapped garbage ignored", HasDecision(mapped_inactive, true) &&
              Attributes(mapped_inactive) == std::vector<int>({0, 1, 2, 3}) && memory.fifth_reads == 0,
          "inactive fetch reachability must not depend on whether garbage points into mapped memory");
  Compile(mapped_inactive, 4);
  Require(name, "mapped garbage survives compilation", memory.fifth_reads == 0,
          "compilation read or normalized a mapped inactive descriptor");

  vertex.resources[4].fields[3] =
      BufferFormat(Prospero::BufferFormat::k32_32_32_32Float) << 12u | DstSel(4, 5, 6, 7);
  memory.words[flag_address / sizeof(u32)] = 0x3f800000u;
  const auto active = Analyze();
  Require(name, "nonzero retains fifth fetch", !HasDecision(active, true) &&
              Attributes(active) == std::vector<int>({0, 1, 2, 3, 4}),
          "a nonzero flag must keep the game's actual fifth vertex input");
  Compile(active, 5);

  auto key_input = vertex;
  std::vector<u32> inactive_key;
  std::vector<u32> active_key;
  key_input.fetch_branches = inactive.decisions;
  BuildStageStaticKey(key_input, inactive_key);
  key_input.fetch_branches = active.decisions;
  BuildStageStaticKey(key_input, active_key);
  Require(name, "branch cache identity", inactive_key != active_key,
          "changing the uniform flag at the same pointer must select a different compiled branch variant");
  memory.words[flag_address / sizeof(u32)] = 0;
  const auto inactive_again = Analyze();
  key_input.fetch_branches = inactive_again.decisions;
  std::vector<u32> repeated_key;
  BuildStageStaticKey(key_input, repeated_key);
  Require(name, "branch cache reuse", repeated_key == inactive_key,
          "returning to the previous branch decision must reuse its shader key");

  memory.words[flag_address / sizeof(u32)] = 0x80000000u;
  Require(name, "signed zero", HasDecision(Analyze(), true),
          "negative floating-point zero must skip the same fetch as positive zero");
  memory.words[flag_address / sizeof(u32)] = 0x7fc00000u;
  Require(name, "NaN remains conservative", !HasDecision(Analyze(), true),
          "an unordered comparison must not discard a potentially reachable fetch");
  memory.flag_readable = false;
  const auto unreadable = Analyze();
  Require(name, "unreadable uniform remains dynamic", unreadable.decisions.empty() &&
              Attributes(unreadable) == std::vector<int>({0, 1, 2, 3, 4}),
          "failed uniform reads must preserve the original shader branch and fetch");
  memory.flag_readable = true;
  auto divergent_code = code;
  divergent_code[compare_word + 1] = EncodeVop3Word1(InlineU32(0), Vgpr(5));
  ShaderRecompiler::Decoder::Program divergent;
  ShaderRecompiler::Decoder::DecodeProgram(divergent_code, divergent);
  const auto divergent_control = AnalyzeShaderFetchControl(divergent, user_data, 8, Read, &memory, 32);
  Require(name, "divergent condition remains dynamic", divergent_control.decisions.empty(),
          "a per-vertex value must not be substituted with a draw-wide branch decision");

  // Two predecessors and a loop-carried scalar value must not be resolved by
  // whichever instruction happened to be visited last in a linear scan.
  const auto CheckUnknownControl = [&](const std::vector<u32> &control_code,
                                       const char *stage) {
    ShaderRecompiler::Decoder::Program control_decoded;
    ShaderRecompiler::Decoder::DecodeProgram(control_code, control_decoded);
    const auto control = AnalyzeShaderFetchControl(control_decoded, user_data, 8,
                                                   Read, &memory, 32);
    Require(name, stage,
            std::ranges::none_of(control.decisions, [](const auto &decision) {
              return decision.first == 8 * sizeof(u32);
            }), "ambiguous predecessor values must not specialize the final branch");
  };
  const std::vector<u32> merged{
      EncodeSopp(0x06u, 2),
      EncodeSMovB32(45, InlineU32(0)),
      EncodeSopp(0x02u, 2),
      EncodeSMovB32(45, 255), 0x3f800000u,
      EncodeVop3Word0(0x0du, 62), EncodeVop3Word1(InlineU32(0), 45),
      EncodeSop1(0x04u, 106, 62),
      EncodeSopp(0x06u, 1), EncodeSopp(0x00u), EncodeSopp(0x01u)};
  CheckUnknownControl(merged, "merged scalar condition remains dynamic");
  const std::vector<u32> loop{
      EncodeSMovB32(45, InlineU32(0)),
      EncodeSopp(0x04u, 3),
      EncodeSMovB32(45, 255), 0x3f800000u,
      EncodeSopp(0x02u, static_cast<u32>(-4)),
      EncodeVop3Word0(0x0du, 62), EncodeVop3Word1(InlineU32(0), 45),
      EncodeSop1(0x04u, 106, 62),
      EncodeSopp(0x06u, 1), EncodeSopp(0x00u), EncodeSopp(0x01u)};
  CheckUnknownControl(loop, "loop-carried scalar condition remains dynamic");

  user_data[4] = 0x1100;
  memory.words[0x1158 / sizeof(u32)] = 0;
  const auto moved_uniform = Analyze();
  Require(name, "current uniform read address", HasDecision(moved_uniform, true) &&
              std::ranges::any_of(moved_uniform.reads, [](const auto &read) {
                return read.first <= 0x1158 && read.first + read.second > 0x1158;
              }) &&
              std::ranges::none_of(moved_uniform.reads, [&](const auto &read) {
                return read.first <= flag_address && read.first + read.second > flag_address;
              }), "cached branch variants must track the current buffer pointer's reads");

  // Skip a semantic in the middle as well: native table indices must survive
  // compaction while Vulkan locations become dense resource indices.
  user_data[4] = 0x1000;
  memory.words[flag_address / sizeof(u32)] = 0;
  code[3 + 6 * 2] = EncodeSmem1(48);
  code[3 + 6 * 3] = EncodeSmem1(64);
  code[branch_word + 2] = EncodeSmem1(8);
  code[branch_word + 4] = EncodeSmem1(32);
  ShaderRecompiler::Decoder::DecodeProgram(code, decoded);
  vertex.resources_dst[2].attr_id = 3;
  vertex.resources_dst[3].attr_id = 4;
  vertex.resources_dst[4].attr_id = 2;
  const auto compacted = Analyze();
  Require(name, "nonidentity semantic compaction", HasDecision(compacted, true) &&
              Attributes(compacted) == std::vector<int>({0, 1, 3, 4}),
          "compacted embedded inputs must retain their original attribute table semantics");
  Compile(compacted, 4);
  std::printf("[host]    %-32s ok\n", name);
}
