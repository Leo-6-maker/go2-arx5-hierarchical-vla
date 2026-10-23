"""Cold transport and CPU compiled SDK-API fixtures, never native ABI proof."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from . import target_sdk_constraint_wire as unit
from .target_mpc_prediction import BoundMPCPrediction


@pytest.fixture
def prediction():
    return BoundMPCPrediction('a'*64, 'b'*64, 'CPU_ONLY', 0., (0., .2, .4),
                              ((0.,)*12,)*3, ((0.,)*12,)*3)


def response(prediction):
    return dict(request=unit.request(prediction), samples=[
        dict(time=t, jointLimitsCost=-3.5, selfCollision=[.01, -.002])
        for t in prediction.time_trajectory])


def bind(prediction, reply):
    raw = json.dumps(reply).encode()
    return unit.bind_samples(prediction, raw, expected_reply_sha256=hashlib.sha256(raw).hexdigest())


def test_signed_margin_and_negative_box_cost_are_not_safety_permission(prediction):
    value = bind(prediction, response(prediction))
    assert value.minimum_self_collision_residual == -.002
    assert value.minimum_joint_limit_cost == value.maximum_joint_limit_cost == -3.5
    assert value.sampling == 'ORIGINAL_NATIVE_POLICY_KNOTS_ONLY'
    assert value.native_prefix_safety_qualified is value.physics_advance_authorized is False


@pytest.mark.parametrize('change', ['stale_policy', 'stale_install', 'state', 'bool', 'missing_knot',
                                   'extra_term', 'ragged', 'empty', 'nan_cost', 'nan_margin', 'bool_cost'])
def test_reject_stale_partial_or_changed_sdk_observation(prediction, change):
    r = response(prediction)
    if change == 'stale_policy': r['request']['policy_sha256'] = 'c'*64
    elif change == 'stale_install': r['request']['wire_sha256'] = 'd'*64
    elif change == 'state': r['request']['states'][0][0] = 1.
    elif change == 'bool': r['request']['states'][0][0] = False
    elif change == 'missing_knot': r['samples'].pop()
    elif change == 'extra_term': r['samples'][0]['endEffector'] = [0.]
    elif change == 'ragged': r['samples'][1]['selfCollision'] = [.01]
    elif change == 'empty': r['samples'][0]['selfCollision'] = []
    elif change == 'nan_cost': r['samples'][0]['jointLimitsCost'] = float('nan')
    elif change == 'nan_margin': r['samples'][0]['selfCollision'] = [float('inf')]
    elif change == 'bool_cost': r['samples'][0]['jointLimitsCost'] = True
    with pytest.raises(PermissionError): bind(prediction, r)


def test_bad_owner_bytes_and_duplicate_json_rejected(prediction):
    with pytest.raises(PermissionError):
        unit.bind_samples(prediction, b'{}', expected_reply_sha256='e'*64)
    raw = b'{"request":{},"request":{},"samples":[]}'
    with pytest.raises(PermissionError, match='DUPLICATE'):
        unit.bind_samples(prediction, raw, expected_reply_sha256=hashlib.sha256(raw).hexdigest())


@pytest.mark.parametrize('change', ['dimension', 'clock', 'nonfinite', 'bool'])
def test_request_validates_complete_batch(prediction, change):
    p = prediction
    if change == 'dimension': p = replace(p, input_trajectory=((0.,)*11,)*3)
    elif change == 'clock': p = replace(p, time_trajectory=(0., .4, .2))
    elif change == 'nonfinite': p = replace(p, state_trajectory=((float('nan'),)*12,)*3)
    elif change == 'bool': p = replace(p, time_trajectory=(False, .2, .4))
    with pytest.raises(PermissionError): unit.request(p)


def test_cache_switch_requires_exact_frozen_bytes():
    with pytest.raises(PermissionError, match='EXACT_FROZEN_TASK_BYTES'):
        unit.cache_only_client_config(b'model_settings { recompileLibraries true }')


# Deliberately small, generated C++ API stand-ins. This executes our real
# header's dispatch and failure behavior, NOT SDK models or native dynamics.
SDK_FIXTURE = r'''
#pragma once
#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <vector>
namespace ocs2 {
struct vector_t : std::vector<double> {
  using std::vector<double>::vector;
  bool allFinite() const { for(auto x:*this) if(!std::isfinite(x)) return false; return true; }
};
enum class Request { Constraint=4, SoftConstraint=8 };
inline int operator+(Request a, Request b) { return int(a)+int(b); }
struct PreComputation {
  int calls=0; void request(int flags,double,const vector_t&,const vector_t&) {
    if(flags!=12) throw std::runtime_error("WRONG_REQUEST"); ++calls;
  }
};
struct TargetTrajectories {};
struct StateCost { virtual ~StateCost()=default; };
struct StateInputCost { virtual ~StateInputCost()=default; };
struct StateConstraint {
  bool active=true; size_t width=2; vector_t values{.01,-.002}; bool second_failure=false;
  bool isActive(double) { return active; }
  size_t getNumConstraints(double) { return width; }
  vector_t getValue(double t,const vector_t&,const PreComputation&) {
    return second_failure && t>.1 ? vector_t{} : values;
  }
};
struct StateSoftConstraint : StateCost {
  StateConstraint underlying;
  template<class T> T& get() { return dynamic_cast<T&>(underlying); }
};
struct StateInputSoftBoxConstraint final : StateInputCost {
  double value=-3.5; bool active=true;
  bool isActive(double) { return active; }
  double getValue(double,const vector_t&,const vector_t&,const TargetTrajectories&,const PreComputation&) {return value;}
};
template<class T> struct Collection {
  std::map<std::string,std::unique_ptr<T>> terms;
  bool getTermIndex(const std::string& n,size_t&) {return terms.count(n);}
  template<class D> D& get(const std::string& n) {return dynamic_cast<D&>(*terms.at(n));}
};
struct OptimalControlProblem {
  std::unique_ptr<Collection<StateInputCost>> softConstraintPtr = std::make_unique<Collection<StateInputCost>>();
  std::unique_ptr<Collection<StateCost>> stateSoftConstraintPtr = std::make_unique<Collection<StateCost>>();
  std::unique_ptr<PreComputation> preComputationPtr=std::make_unique<PreComputation>();
};
}
'''
DRIVER = r'''
#include <iostream>
#include <limits>
#include "target_sdk_constraints.hpp"
int main(int argc,char** argv) {
  const std::string mode=argv[1]; ocs2::OptimalControlProblem p;
  auto box=std::make_unique<ocs2::StateInputSoftBoxConstraint>(); auto* j=box.get();
  auto term=std::make_unique<ocs2::StateSoftConstraint>(); auto* c=&term->underlying;
  p.softConstraintPtr->terms["jointLimits"]=std::move(box);
  p.stateSoftConstraintPtr->terms["selfCollision"]=std::move(term);
  // Tracking equality may be arbitrarily bad; never read/reclassify it.
  p.stateSoftConstraintPtr->terms["endEffector"]=std::make_unique<ocs2::StateCost>();
  if(mode=="missing")p.softConstraintPtr->terms.erase("jointLimits");
  if(mode=="wrong_type")p.softConstraintPtr->terms["jointLimits"]=std::make_unique<ocs2::StateInputCost>();
  if(mode=="extra")p.stateSoftConstraintPtr->terms["environmentCollision"]=std::make_unique<ocs2::StateCost>();
  std::vector<dec1::SdkKnot> knots{{0.,ocs2::vector_t(12,0.),ocs2::vector_t(12,0.)},
                                {.4,ocs2::vector_t(12,0.),ocs2::vector_t(12,0.)}};
  if(mode=="late_nan")knots[1].state[0]=std::numeric_limits<double>::quiet_NaN();
  if(mode=="dimension")knots[1].input.resize(11);
  if(mode=="clock")knots[1].time=0.;
  if(mode=="inactive")c->active=false;
  if(mode=="nan_residual")c->values[0]=std::numeric_limits<double>::quiet_NaN();
  if(mode=="nan_cost")j->value=std::numeric_limits<double>::infinity();
  if(mode=="empty")c->values.clear();
  if(mode=="mismatch")c->width=3;
  if(mode=="partial")c->second_failure=true;
  try {
    dec1::OriginalSdkConstraints client(p);
    if(mode=="thread") {
      bool rejected=false; std::thread other([&]{try{client.sample(knots);}catch(...){rejected=true;}});other.join();
      if(!rejected || p.preComputationPtr->calls) return 11; std::cout<<"REJECTED";return 0;
    }
    try {
      auto v=client.sample(knots);
      if(mode!="ok" || v.size()!=2 || p.preComputationPtr->calls!=2 ||
         v[0].joint_limit_cost!=-3.5 || v[1].self_collision[1]!=-.002) return 12;
      std::cout<<"SIGNED_COST_AND_RESIDUAL_PRESERVED";return 0;
    } catch(...) {
      int calls=p.preComputationPtr->calls;
      try{client.sample(knots);return 13;}catch(...){}
      if(p.preComputationPtr->calls!=calls) return 14;
      if((mode=="late_nan"||mode=="dimension"||mode=="clock") && calls!=0)return 15;
      if(mode=="ok")return 16;std::cout<<"REJECTED";return 0;
    }
  } catch(...) {if(mode!="missing" && mode!="wrong_type" && mode!="extra")return 17; std::cout<<"REJECTED";return 0;}
}
'''


@pytest.fixture(scope='module')
def compiled_cpu_client(tmp_path_factory):
    if not shutil.which('g++'): pytest.skip('C++ CPU fixture unavailable; no native ABI qualification')
    folder = tmp_path_factory.mktemp('original_sdk_api_cpu_fixture')
    for name in ('ocs2_core/soft_constraint/StateSoftConstraint.h',
                 'ocs2_core/soft_constraint/StateInputSoftBoxConstraint.h',
                 'ocs2_oc/oc_problem/OptimalControlProblem.h'):
        path = folder/name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('#include "cpu_sdk_fixture.hpp"\n')
    (folder/'cpu_sdk_fixture.hpp').write_text(SDK_FIXTURE)
    (folder/'client.cpp').write_text(DRIVER)
    binary = folder/'cpu_client'
    subprocess.run(['g++', '-std=c++17', '-Wall', '-Wextra', '-Werror', '-Wno-misleading-indentation',
                    '-Wno-unused-parameter', '-pthread', '-I'+str(folder), '-I'+str(Path(__file__).parent),
                    str(folder/'client.cpp'), '-o', str(binary)], check=True, capture_output=True, timeout=30)
    return binary


@pytest.mark.parametrize('mode', ['ok', 'missing', 'wrong_type', 'extra', 'late_nan', 'dimension', 'clock',
                                  'inactive', 'nan_residual', 'nan_cost', 'empty', 'mismatch', 'partial', 'thread'])
def test_compiled_reader_batch_identity_dispatch_and_failures(compiled_cpu_client, mode):
    r = subprocess.run([str(compiled_cpu_client), mode], check=True, capture_output=True, text=True, timeout=5)
    assert r.stdout == ('SIGNED_COST_AND_RESIDUAL_PRESERVED' if mode == 'ok' else 'REJECTED')
