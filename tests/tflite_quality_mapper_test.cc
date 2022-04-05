#include "tflite_quality_mapper.h"

#include "file_path.h"
#include "gtest/gtest.h"

constexpr absl::string_view kSpeechModelPath =
    "/model/"
    "lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_ep2400"
    "_train1_7_raw.tflite";
constexpr int kSpeechModelFrequencyBands = 21;

namespace Visqol {
namespace {

static TFLiteQualityMapper CreateModelFromGoogle3Path(absl::string_view path) {
  return TFLiteQualityMapper(absl::StrCat(FilePath::currentWorkingDir(), path),
                             kSpeechModelFrequencyBands);
}

TEST(TFLiteQualityMapperLoading, InitFailsWithBadPath) {
  TFLiteQualityMapper model = CreateModelFromGoogle3Path("nonexistent.tflite");

  EXPECT_FALSE(model.Init().ok());
}

TEST(TFLiteQualityMapperUsage, PredictMLObservationGood) {
  TFLiteQualityMapper model = CreateModelFromGoogle3Path(kSpeechModelPath);

  ASSERT_TRUE(model.Init().ok());

  // Make a fake observation with high similarity.
  std::vector<double> good_fvnsim(kSpeechModelFrequencyBands, 1.0);
  std::vector<double> good_fvnsim10(kSpeechModelFrequencyBands, 1.0);
  std::vector<double> good_fstdnsim(kSpeechModelFrequencyBands, 1.0);
  std::vector<double> good_fvdegenergy(kSpeechModelFrequencyBands, 1.0);
  double good_mos = model.PredictQuality(good_fvnsim, good_fvnsim10,
                                         good_fstdnsim, good_fvdegenergy);

  // A perfect similarity MOS should be above 3.0 for any reasonable model.
  EXPECT_GT(good_mos, 3.0);
  // 5.0 is the maximum score.
  EXPECT_LE(good_mos, 5.0);
}

TEST(TFLiteQualityMapperUsage, PredictMLObservationBad) {
  TFLiteQualityMapper model = CreateModelFromGoogle3Path(kSpeechModelPath);

  ASSERT_TRUE(model.Init().ok());

  // Make a fake observation with high similarity.
  std::vector<double> bad_fvnsim(kSpeechModelFrequencyBands, 0.2);
  std::vector<double> bad_fvnsim10(kSpeechModelFrequencyBands, 0.1);
  std::vector<double> bad_fstdnsim(kSpeechModelFrequencyBands, 1.0);
  std::vector<double> bad_fvdegenergy(kSpeechModelFrequencyBands, 1.0);
  double bad_mos = model.PredictQuality(bad_fvnsim, bad_fvnsim10, bad_fstdnsim,
                                        bad_fvdegenergy);

  // A bad similarity MOS should be below 3.0 for any reasonable model.
  EXPECT_GE(bad_mos, 1.0);
  EXPECT_LT(bad_mos, 3.0);
}

}  // namespace
}  // namespace Visqol
