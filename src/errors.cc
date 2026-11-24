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

#include <optional>
#include <string>
#include <type_traits>

#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/numbers.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"

namespace Visqol {

constexpr absl::string_view kVisqolErrorSpaceKey = "visqol";
constexpr absl::string_view kVisqolErrorSpaceValue = "1";
constexpr absl::string_view kVisqolErrorSpaceCodeKey = "error_code";
constexpr absl::string_view kVisqolErrorSpaceCodeDescriptionKey = "error_descr";

enum class VisqolErrorCode {
  kOk = 0,
  kInputAudioTooShort = 1,
};

using VisqolErrorCode_t = std::underlying_type<VisqolErrorCode>::type;

std::string VisqolErrorCodeToDescriptionString(
    VisqolErrorCode visqol_error_code) {
  switch (visqol_error_code) {
    case VisqolErrorCode::kOk:
      return "OK";
    case VisqolErrorCode::kInputAudioTooShort:
      return "Input audio is too short to compute a score";
  }
  return "";
}

absl::StatusCode VisqolErrorCodeToCanonicalErrorCode(
    VisqolErrorCode visqol_error_code) {
  switch (visqol_error_code) {
    case VisqolErrorCode::kOk:
      return absl::StatusCode::kOk;
    case VisqolErrorCode::kInputAudioTooShort:
      return absl::StatusCode::kInvalidArgument;
  }
  return absl::StatusCode::kUnknown;
}

bool IsVisqolError(const absl::Status& status) {
  auto is_visqol_error = status.GetPayload(kVisqolErrorSpaceKey);
  return is_visqol_error && is_visqol_error == kVisqolErrorSpaceValue;
}

absl::Status MakeVisqolError(VisqolErrorCode visqol_error_code,
                             absl::string_view message) {
  absl::StatusCode canonical_status_code =
      VisqolErrorCodeToCanonicalErrorCode(visqol_error_code);
  absl::Status status{canonical_status_code, message};
  status.SetPayload(kVisqolErrorSpaceKey, absl::Cord(kVisqolErrorSpaceValue));
  status.SetPayload(
      kVisqolErrorSpaceCodeKey,
      absl::Cord(absl::Substitute(
          "$0", static_cast<VisqolErrorCode_t>(visqol_error_code))));
  // Not strictly needed, but helpful message for user to understand the type of
  // error.
  status.SetPayload(
      kVisqolErrorSpaceCodeDescriptionKey,
      absl::Cord(VisqolErrorCodeToDescriptionString(visqol_error_code)));
  return status;
}

std::optional<VisqolErrorCode> GetVisqolErrorCode(const absl::Status& status) {
  std::optional<absl::Cord> visqol_error_code_cord =
      status.GetPayload(kVisqolErrorSpaceCodeKey);
  if (!visqol_error_code_cord) {
    return std::nullopt;
  }
  if (VisqolErrorCode_t visqol_error_code; !absl::SimpleAtoi(
          visqol_error_code_cord->Flatten(), &visqol_error_code)) {
    return std::nullopt;
  } else {
    return static_cast<VisqolErrorCode>(visqol_error_code);
  }
}

bool IsVisqolErrorType(const absl::Status& status, VisqolErrorCode expected) {
  if (!IsVisqolError(status)) {
    return false;
  }
  auto actual = GetVisqolErrorCode(status);
  return actual && *actual == expected;
}

absl::Status InputAudioTooShort(absl::string_view message) {
  return MakeVisqolError(VisqolErrorCode::kInputAudioTooShort, message);
}

bool IsInputAudioTooShort(const absl::Status& status) {
  return IsVisqolErrorType(status, VisqolErrorCode::kInputAudioTooShort);
}
}  // namespace Visqol
