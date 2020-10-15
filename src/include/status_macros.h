#ifndef VISQOL_INCLUDE_STATUSOR_MACROS_H
#define VISQOL_INCLUDE_STATUSOR_MACROS_H

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace Visqol {

#define VISQOL_RETURN_IF_ERROR(expr)                                         \
  do {                                                                       \
    /* Using _status below to avoid capture problems if expr is "status". */ \
    const absl::Status _status = (expr);                                     \
    if (!_status.ok()) return _status;                                       \
  } while (0)

template<typename T>
absl::Status DoAssignOrReturn(T& lhs, absl::StatusOr<T> result) {
  if (result.ok()) {
    lhs = result.value();
  }
  return result.status();
}

#define VISQOL_ASSIGN_OR_RETURN_IMPL(status, lhs, rexpr) \
  absl::Status status = ::Visqol::DoAssignOrReturn(lhs, (rexpr)); \
  if (!status.ok()) return status;

#define VISQOL_STATUS_MACROS_CONCAT_NAME_INNER(x, y) x##y
#define VISQOL_STATUS_MACROS_CONCAT_NAME(x, y) VISQOL_STATUS_MACROS_CONCAT_NAME_INNER(x, y)

// Executes an expression that returns a util::StatusOr, extracting its value
// into the variable defined by lhs (or returning on error).
//
// Example: Assigning to an existing value
//   ValueType value;
//   ASSIGN_OR_RETURN(value, MaybeGetValue(arg));
//
// WARNING: VISQOL_ASSIGN_OR_RETURN expands into multiple statements; it cannot be used
//  in a single statement (e.g. as the body of an if statement without {})!
#define VISQOL_ASSIGN_OR_RETURN(lhs, rexpr) \
  VISQOL_ASSIGN_OR_RETURN_IMPL( \
      VISQOL_STATUS_MACROS_CONCAT_NAME(_status_or_value, __COUNTER__), lhs, rexpr);

}  // namespace Visqol

#endif  // VISQOL_INCLUDE_STATUSOR_MACROS_H
