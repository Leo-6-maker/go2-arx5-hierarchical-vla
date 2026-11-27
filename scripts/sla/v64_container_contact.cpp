#include <cmath>
#include <fstream>
#include <string>
#include <string_view>

#include <mujoco/mujoco.h>

namespace {

// Issued from the 2 ms measurements and accepted OPEN receipt at a paused atom boundary.
bool valid_release(const char* path, const char* condition, long tick, long last_foreign_contact) {
  if (!path || !*path) return false;
  std::ifstream input(path);
  std::string version;
  std::string recorded_condition;
  long pause, grasp, lift, issue, ack, separation;
  if (!(input >> version >> recorded_condition >> pause >> grasp >> lift >> issue >> ack >> separation)) {
    return false;
  }
  input >> std::ws;
  return input.eof() && version == "V64_CONTAINER_RELEASE" && recorded_condition == condition &&
         0 <= grasp && grasp <= lift && lift <= issue && issue < separation &&
         issue <= ack && ack - issue == 25 && separation > last_foreign_contact &&
         separation <= pause && ack <= pause &&
         5703 <= pause && (pause - 5703) % 25 == 0 && pause < tick && tick - pause <= 25;
}

bool fits_floor(const mjModel* model, const mjData* data, int target, int floor) {
  const auto* center = data->geom_xpos + 3 * target;
  const auto* floor_center = data->geom_xpos + 3 * floor;
  const auto* rotation = data->geom_xmat + 9 * target;
  const auto* half = model->geom_size + 3 * target;
  const auto radius = model->geom_size[3 * floor];
  for (int sx : {-1, 1}) {
    for (int sy : {-1, 1}) {
      for (int sz : {-1, 1}) {
        auto x = center[0] - floor_center[0] +
                 sx * half[0] * rotation[0] + sy * half[1] * rotation[1] + sz * half[2] * rotation[2];
        auto y = center[1] - floor_center[1] +
                 sx * half[0] * rotation[3] + sy * half[1] * rotation[4] + sz * half[2] * rotation[5];
        if (!(std::hypot(x, y) <= radius)) return false;
      }
    }
  }
  return true;
}

}  // namespace

extern "C" bool WM02ContainerHasForeignContactV64(const mjModel* model, const mjData* data,
                                                  const char* condition) {
  if (!condition) return false;
  const std::string_view role(condition);
  if (role != "EF5_Q_CONTAINER_01" && role != "EF5_Q_CONTAINER_02") return false;
  const int floor = mj_name2id(model, mjOBJ_GEOM, "ef5_static_bowl_base");
  const int target = mj_name2id(model, mjOBJ_GEOM,
      role == "EF5_Q_CONTAINER_01" ? "task_cube_geom" : "task_blue_cube_geom");
  if (floor < 0 || target < 0) return false;
  for (int i = 0; i < data->ncon; ++i) {
    const auto& contact = data->contact[i];
    if ((contact.geom1 == target && contact.geom2 != floor) ||
        (contact.geom2 == target && contact.geom1 != floor)) return true;
  }
  return false;
}

extern "C" bool WM02ContainerFloorContactV64(const mjModel* model, const mjData* data,
                                            int index, const char* selected, long tick,
                                            const char* condition, const char* receipt_path,
                                            long last_foreign_contact) {
  if (!condition || !selected || !receipt_path) return false;
  const std::string_view role(condition);
  const std::string_view color(selected);
  if (!((role == "EF5_Q_CONTAINER_01" && color == "red") ||
        (role == "EF5_Q_CONTAINER_02" && color == "blue"))) return false;
  if (index < 0 || index >= data->ncon) return false;
  const auto& contact = data->contact[index];
  if (!std::isfinite(contact.dist) || contact.dist < -.004) return false;
  const int floor = mj_name2id(model, mjOBJ_GEOM, "ef5_static_bowl_base");
  const int target = mj_name2id(model, mjOBJ_GEOM, color == "red" ? "task_cube_geom" : "task_blue_cube_geom");
  if (floor < 0 || target < 0 ||
      !((contact.geom1 == floor && contact.geom2 == target) ||
        (contact.geom2 == floor && contact.geom1 == target))) return false;
  const int bowl = mj_name2id(model, mjOBJ_BODY, "ef5_static_bowl");
  if (bowl <= 0 || model->geom_bodyid[floor] != bowl ||
      model->body_parentid[bowl] != 0 || model->body_jntnum[bowl] != 0 ||
      model->geom_type[floor] != mjGEOM_CYLINDER || model->geom_type[target] != mjGEOM_BOX ||
      !std::isfinite(data->time) || std::abs(data->time - .002 * tick) > 1e-8) return false;
  return valid_release(receipt_path, condition, tick, last_foreign_contact) &&
         fits_floor(model, data, target, floor);
}
