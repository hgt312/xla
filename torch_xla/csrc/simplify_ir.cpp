#include "torch_xla/csrc/simplify_ir.h"

#include <algorithm>
#include <list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_domain_map.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_query.h"
#include "tensorflow/compiler/xla/service/hlo_reachability.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/errors.h"

namespace torch_xla {

using namespace xla;
namespace m = match;

namespace {

// lazy: select(unselect(x, src, dim, start, end, stride), dim, start, end, stride) -> src
// hlo: slice(select(pad(ones(), 0), pad(src, 0), x)) -> src
StatusOr<bool> HandleUnselect(HloComputation* computation, HloInstruction* slice) {
  bool changed = false;

  HloInstruction* pad;
  HloInstruction* src;

  if (Match(slice, m::Slice(m::Select(m::Pad(m::Op(), m::Constant()), m::Pad(&pad, m::Op(&src), m::Constant()), m::Op())))) {
    // Is the result of the slice the pad operand.
    bool slice_undoes_pad = true;
    for (int64_t i = 0; i < slice->shape().rank(); ++i) {
      const int64_t start = slice->slice_starts(i);
      const int64_t stride = slice->slice_strides(i);
      const int64_t limit = slice->slice_limits(i);

      const int64_t size = pad->shape().dimensions(i);
      const auto& dim = pad->padding_config().dimensions(i);
      const int64_t low = dim.edge_padding_low();
      const int64_t high = dim.edge_padding_high();
      const int64_t interior = dim.interior_padding();
      const int64_t edge = size - high;

      if (start != low || stride - 1 != interior || limit != edge) {
        slice_undoes_pad = false;
        break;
      }
    }

    if (slice_undoes_pad && ShapeUtil::Equal(slice->shape(), src->shape())) {
      TF_RETURN_IF_ERROR(computation->ReplaceInstruction(slice, src));
      changed = true;
    }
  }
  return changed;
}

}  // namespace

StatusOr<bool> IRSimplifier::Run(HloModule* module) {
  bool changed = false;
  for (auto computation : module->MakeComputationPostOrder()) {
    for (auto inst : computation->MakeInstructionPostOrder()) {
      if (inst->opcode() == HloOpcode::kSlice) {
        TF_ASSIGN_OR_RETURN(bool local_changed, HandleUnselect(computation, inst));
        changed |= local_changed;
      }
    }
  }
  return changed;
}

}  // namespace torch_xla
