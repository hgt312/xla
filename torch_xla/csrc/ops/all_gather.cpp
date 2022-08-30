#include "torch_xla/csrc/ops/all_gather.h"

#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/ops/xla_ops.h"

namespace torch_xla {
namespace {

xla::Shape NodeOutputShape(const torch::lazy::Value& input,
                           const torch::lazy::Value& token, int64_t dim,
                           int64_t shard_count,
                           const std::vector<std::vector<int64_t>>& groups,
                           bool pin_layout) {
  auto shape_fn = [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    AllGatherResult result = BuildAllGather(operands[0], operands[1], dim,
                                            shard_count, groups, pin_layout);
    return xla::Tuple(operands[0].builder(), {result.result, result.token});
  };
  return InferOutputShape({GetXlaShape(input), GetXlaShape(token)}, shape_fn);
}

}  // namespace

AllGather::AllGather(const torch::lazy::Value& input,
                     const torch::lazy::Value& token, int64_t dim,
                     int64_t shard_count,
                     std::vector<std::vector<int64_t>> groups, bool pin_layout)
    : XlaNode(xla_all_gather, {input, token},
              [&]() {
                return NodeOutputShape(input, token, dim, shard_count, groups,
                                       pin_layout);
              },
              /*num_outputs=*/2,
              torch::lazy::MHash(dim, shard_count, groups, pin_layout)),
      dim_(dim),
      shard_count_(shard_count),
      groups_(std::move(groups)),
      pin_layout_(pin_layout) {}

torch::lazy::NodePtr AllGather::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<AllGather>(operands.at(0), operands.at(1), dim_,
                                          shard_count_, groups_, pin_layout_);
}

const torch::lazy::Output& SimplifyUnselect(const torch::lazy::Output& x) {
  if (!xla::sys_util::GetEnvBool("SIMPLIFY_FOR_DEEPSPEED", false)) {
    return x;
  }
  if (x.node->op() == xla_select) {
    auto& y = x.node->operand(0);
    if (y.node->op() == xla_unselect) {
      return y.node->operand(1);
    }
  }
  if (x.node->op() == xla_unselect) {
    auto& y = x.node->operand(0);
    if (y.node->op() == xla_select) {
      return x.node->operand(1);
    }
  }
  return x;
}

XlaOpVector AllGather::Lower(LoweringContext* loctx) const {
  auto& input0 = SimplifyUnselect(SimplifyUnselect(operand(0)));
  xla::XlaOp input = loctx->GetOutputOp(input0);
  xla::XlaOp token = loctx->GetOutputOp(operand(1));
  AllGatherResult result =
      BuildAllGather(input, token, dim_, shard_count_, groups_, pin_layout_);
  return ReturnOps({result.result, result.token}, loctx);
}

std::string AllGather::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", dim=" << dim_
     << ", shard_count=" << shard_count_ << ", pin_layout=" << pin_layout_
     << ", groups=(";
  for (size_t i = 0; i < groups_.size(); ++i) {
    ss << (i == 0 ? "(" : ",(");
    ss << absl::StrJoin(groups_[i], ", ") << ")";
  }
  ss << ")";
  return ss.str();
}

}  // namespace torch_xla
