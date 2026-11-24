// Copyright 2019 Google LLC, Andrew Hines
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "errors.h"

#include "absl/status/status.h"
#include "gtest/gtest.h"

namespace Visqol {
namespace {

TEST(VisqolError, VisqolErrorSpace) {
  auto status = absl::InvalidArgumentError("invalid argument");
  EXPECT_FALSE(IsVisqolError(status));
  EXPECT_FALSE(IsInputAudioTooShort(status));
}

TEST(VisqolError, VisqolErrorSpaceOk) {
  auto status = absl::OkStatus();
  EXPECT_FALSE(IsVisqolError(status));
  EXPECT_FALSE(IsInputAudioTooShort(status));
}

TEST(VisqolError, InputAudioTooShort) {
  auto status = InputAudioTooShort("too short");
  EXPECT_FALSE(status.ok());
  EXPECT_TRUE(IsVisqolError(status));
  EXPECT_TRUE(IsInputAudioTooShort(status));
  EXPECT_TRUE(absl::IsInvalidArgument(status));
}

}  // namespace
}  // namespace Visqol
