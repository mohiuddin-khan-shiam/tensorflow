/* Copyright 2024 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <linux/limits.h>
#include <unistd.h>

#include <string>

#include "absl/log/log.h"
#include "xla/pjrt/plugin/dynamic_registration.h"

static constexpr char kMyPluginName[] = "myplugin";

[[maybe_unused]] auto setup_test_plugin = []() -> bool {
  char build_dir[PATH_MAX];
  char* return_path = getcwd(build_dir, sizeof(build_dir));
  if (return_path == nullptr) {
    LOG(ERROR) << "Failed to get current working directory.";
    return false;
  }

  std::string library_path = std::string(build_dir) +
                             "/third_party/tensorflow/compiler/xla/pjrt/plugin/"
                             "example_plugin/pjrt_c_api_myplugin_plugin.so";

  setenv("MYPLUGIN_DYNAMIC_PATH", library_path.c_str(), 1);
  REGISTER_DYNAMIC_PJRT_PLUGIN(kMyPluginName, "MYPLUGIN_DYNAMIC_PATH")
  return true;
}();
