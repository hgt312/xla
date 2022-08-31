#pragma once

#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace torch_xla {

using namespace xla;

class IRSimplifier : public HloModulePass {
 public:
  IRSimplifier() {};

  absl::string_view name() const override { return "ir-simplifier"; }

  StatusOr<bool> Run(HloModule* module) override;
};

}  // namespace xla
