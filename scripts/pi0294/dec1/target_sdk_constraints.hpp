// Cold client of the existing SDK, never a replacement controller/solver.
// The caller owns a DEDICATED, admitted original SDK problem. Do not attach
// this reader to a live solver: its precomputation cache is mutable. This
// samples existing native policy knots; it is not a continuous/full-body
// safety certificate and has no quota, ROS publication or physics API.
#pragma once
#include <cmath>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <ocs2_core/soft_constraint/StateSoftConstraint.h>
#include <ocs2_core/soft_constraint/StateInputSoftBoxConstraint.h>
#include <ocs2_oc/oc_problem/OptimalControlProblem.h>

namespace dec1 {
struct SdkKnot {
  double time;
  ocs2::vector_t state, input;
};
struct SdkConstraintSample {
  double time;
  double joint_limit_cost;
  ocs2::vector_t self_collision;
};

class OriginalSdkConstraints {
 public:
  explicit OriginalSdkConstraints(const ocs2::OptimalControlProblem& problem)
      : problem_(problem), thread_(std::this_thread::get_id()) {
    require(problem_.preComputationPtr && problem_.softConstraintPtr &&
            problem_.stateSoftConstraintPtr, "MISSING_ORIGINAL_COLLECTION");
    // Exact task_floating config 0b69bd58 enables these bound/collision
    // terms. endEffector/finalEndEffector are tracking equalities/costs,
    // not collision margins or a resurrected hold-RMS gate.
    size_t index;
    require(problem_.softConstraintPtr->getTermIndex("jointLimits", index) &&
            problem_.stateSoftConstraintPtr->getTermIndex("selfCollision", index),
            "MISSING_FROZEN_ENABLED_TERM");
    require(!problem_.softConstraintPtr->getTermIndex("policyFeasibility", index) &&
            !problem_.stateSoftConstraintPtr->getTermIndex("bodyRelative", index) &&
            !problem_.stateSoftConstraintPtr->getTermIndex("environmentCollision", index),
            "DIFFERENT_FROZEN_TERM_PROFILE");
    // The actual joint term is a SoftBox COST with private bounds, NOT a
    // StateInputSoftConstraint. Its scalar includes an initialized offset;
    // zero/sign is not a feasibility threshold. Never cast/copy private ABI.
    joint_ = &problem_.softConstraintPtr->get<ocs2::StateInputSoftBoxConstraint>("jointLimits");
    collision_ = &problem_.stateSoftConstraintPtr->get<ocs2::StateSoftConstraint>("selfCollision")
                      .get<ocs2::StateConstraint>();
  }
  OriginalSdkConstraints(const OriginalSdkConstraints&) = delete;
  OriginalSdkConstraints& operator=(const OriginalSdkConstraints&) = delete;

  std::vector<SdkConstraintSample> sample(const std::vector<SdkKnot>& knots) {
    require(thread_ == std::this_thread::get_id(), "SAME_OWNER_THREAD");
    require(!failed_, "LATCHED_CLIENT_FAILURE");
    try {
      require(knots.size() >= 2 && knots.size() <= 4096, "BOUNDED_NATIVE_KNOTS");
      // Validate the WHOLE batch before touching the original SDK cache.
      for (size_t i = 0; i < knots.size(); ++i) {
        const auto& k = knots[i];
        require(std::isfinite(k.time) && (i == 0 || k.time > knots[i-1].time),
                "FINITE_STRICT_NATIVE_CLOCK");
        require(k.state.size() == 12 && k.input.size() == 12 &&
                k.state.allFinite() && k.input.allFinite(), "ORIGINAL_12D_FINITE_VECTORS");
      }
      std::vector<SdkConstraintSample> result;
      for (const auto& k : knots) {
        require(joint_->isActive(k.time) && collision_->isActive(k.time),
                "FROZEN_TERM_UNEXPECTEDLY_INACTIVE");
        problem_.preComputationPtr->request(
            ocs2::Request::Constraint + ocs2::Request::SoftConstraint,
            k.time, k.state, k.input);
        const auto j = joint_->getValue(k.time, k.state, k.input,
                                      ocs2::TargetTrajectories{}, *problem_.preComputationPtr);
        auto c = collision_->getValue(k.time, k.state, *problem_.preComputationPtr);
        require(std::isfinite(j) && c.size() > 0 && c.allFinite() &&
                static_cast<size_t>(c.size()) == collision_->getNumConstraints(k.time),
                "ORIGINAL_CONSTRAINT_VECTOR_SHAPE_AND_FINITE");
        // Preserve signed original residuals and the original box COST.
        // Do not clamp, add tolerance or emit a safety PASS here.
        result.push_back({k.time, std::move(j), std::move(c)});
      }
      return result;
    } catch (...) {
      failed_ = true;  // no partial response or automatic fallback/retry
      throw;
    }
  }

 private:
  static void require(bool ok, const char* why) {
    if (!ok) throw std::runtime_error(std::string("DEC1_SDK_") + why);
  }
  const ocs2::OptimalControlProblem& problem_;
  std::thread::id thread_;
  ocs2::StateInputSoftBoxConstraint* joint_ = nullptr;
  ocs2::StateConstraint* collision_ = nullptr;
  bool failed_ = false;
};
}  // namespace dec1
