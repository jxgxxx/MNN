//
//  QuantizedAddTest.cpp
//  MNNTests
//

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

#include "MNNTestSuite.h"

#ifdef MNN_SUPPORT_DEPRECATED_OP

#include "MNN_generated.h"
#include "backend/cpu/CPUFixedPoint.hpp"
#include "backend/cpu/CPUQuantizationUtils.hpp"
#include "core/Macro.h"

using namespace MNN;
using namespace MNN::Express;

namespace {

struct QuantizedAddSpec {
    int32_t input1ZeroPoint;
    int32_t input2ZeroPoint;
    int32_t outputZeroPoint;
    float input1Scale;
    float input2Scale;
    float outputScale;
    FusedActivation activation;
};

struct QuantizedAddParams {
    int32_t input1Offset;
    int32_t input2Offset;
    int32_t outputOffset;
    int32_t leftShiftResult1;
    int32_t leftShiftResult2;
    int32_t input1Multiplier;
    int32_t input2Multiplier;
    int32_t rightShift1;
    int32_t rightShift2;
    int32_t leftShiftOut;
    int32_t outputMultiplier;
    int32_t rightShiftOut;
    int32_t outputActivationMin;
    int32_t outputActivationMax;
};

static QuantizedAddParams makeParams(const QuantizedAddSpec& spec) {
    QuantizedAddParams params;
    params.input1Offset = -spec.input1ZeroPoint;
    params.input2Offset = -spec.input2ZeroPoint;
    params.outputOffset = spec.outputZeroPoint;
    const int leftShift = 20;
    const double twiceMax = 2.0 * std::max(spec.input1Scale, spec.input2Scale);
    QuantizeMultiplierSmallerThanOne(spec.input1Scale / twiceMax, &params.input1Multiplier, &params.rightShift1);
    QuantizeMultiplierSmallerThanOne(spec.input2Scale / twiceMax, &params.input2Multiplier, &params.rightShift2);
    QuantizeMultiplierSmallerThanOne(twiceMax / ((1 << leftShift) * spec.outputScale), &params.outputMultiplier,
                                     &params.rightShiftOut);
    params.leftShiftResult1 = 1 << leftShift;
    params.leftShiftResult2 = 1 << leftShift;
    params.leftShiftOut = 0;
    CalculateActivationRangeUint8(spec.activation, spec.outputZeroPoint, spec.outputScale, &params.outputActivationMin,
                                  &params.outputActivationMax);
    return params;
}

static VARP makeQuantizedAdd(VARP input1, VARP input2, const QuantizedAddSpec& spec) {
    std::unique_ptr<OpT> op(new OpT);
    op->type = OpType_QuantizedAdd;
    op->main.type = OpParameter_QuantizedAdd;
    op->main.value = new QuantizedAddT;
    auto add = op->main.AsQuantizedAdd();
    add->activationType = spec.activation;
    add->input1QuantizedParam.reset(new QuantizedParamT);
    add->input2QuantizedParam.reset(new QuantizedParamT);
    add->outputQuantizedParam.reset(new QuantizedParamT);
    add->input1QuantizedParam->zeroPoint = spec.input1ZeroPoint;
    add->input1QuantizedParam->scale = spec.input1Scale;
    add->input2QuantizedParam->zeroPoint = spec.input2ZeroPoint;
    add->input2QuantizedParam->scale = spec.input2Scale;
    add->outputQuantizedParam->zeroPoint = spec.outputZeroPoint;
    add->outputQuantizedParam->scale = spec.outputScale;
    return Variable::create(Expr::create(op.get(), {input1, input2}, 1));
}

static void quantizedAddScalar(const uint8_t* input1, const uint8_t* input2, uint8_t* output, size_t size,
                               const QuantizedAddParams& params) {
    for (size_t i = 0; i < size; ++i) {
        const int32_t input1Value = params.input1Offset + input1[i];
        const int32_t input2Value = params.input2Offset + input2[i];
        const int32_t scaledInput1 = RoundingDivideByPOT(
            SaturatingRoundingDoublingHighMul(input1Value * params.leftShiftResult1, params.input1Multiplier),
            params.rightShift1);
        const int32_t scaledInput2 = RoundingDivideByPOT(
            SaturatingRoundingDoublingHighMul(input2Value * params.leftShiftResult2, params.input2Multiplier),
            params.rightShift2);
        const int32_t rawSum = (scaledInput1 + scaledInput2) * (1 << params.leftShiftOut);
        const int32_t rawOutput =
            RoundingDivideByPOT(SaturatingRoundingDoublingHighMul(rawSum, params.outputMultiplier),
                                params.rightShiftOut) +
            params.outputOffset;
        output[i] =
            static_cast<uint8_t>(std::min(params.outputActivationMax, std::max(params.outputActivationMin, rawOutput)));
    }
}

static bool runCase(const std::vector<uint8_t>& input1, const std::vector<uint8_t>& input2,
                    const QuantizedAddSpec& spec, int caseIndex) {
    if (input1.size() != input2.size() || input1.size() % 4 != 0) {
        MNN_ERROR("Invalid QuantizedAdd test input size\n");
        return false;
    }

    const int width = static_cast<int>(input1.size() / 4);
    auto input1Var = _Input({1, 4, 1, width}, NCHW, halide_type_of<uint8_t>());
    auto input2Var = _Input({1, 4, 1, width}, NCHW, halide_type_of<uint8_t>());
    ::memcpy(input1Var->writeMap<uint8_t>(), input1.data(), input1.size());
    ::memcpy(input2Var->writeMap<uint8_t>(), input2.data(), input2.size());
    input1Var->unMap();
    input2Var->unMap();

    std::vector<uint8_t> expected(input1.size());
    quantizedAddScalar(input1.data(), input2.data(), expected.data(), input1.size(), makeParams(spec));
    auto output = makeQuantizedAdd(input1Var, input2Var, spec);
    const uint8_t* actual = output->readMap<uint8_t>();
    if (actual == nullptr) {
        MNN_ERROR("QuantizedAdd execution failed for case %d\n", caseIndex);
        return false;
    }

    for (size_t i = 0; i < input1.size(); ++i) {
        if (expected[i] != actual[i]) {
            MNN_ERROR("QuantizedAdd mismatch: case=%d, index=%zu, input1=%d, input2=%d, scalar=%d, actual=%d\n",
                      caseIndex, i, static_cast<int>(input1[i]), static_cast<int>(input2[i]),
                      static_cast<int>(expected[i]), static_cast<int>(actual[i]));
            return false;
        }
    }
    return true;
}

static uint32_t nextRandom(uint32_t& state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

static float randomUnit(uint32_t& state) {
    return static_cast<float>(nextRandom(state) & 0xffff) / 65535.0f;
}

} // namespace

class QuantizedAddTest : public MNNTestCase {
public:
    virtual ~QuantizedAddTest() = default;

    virtual bool run(int precision) {
        (void)precision;
        const std::vector<uint8_t> boundaryInput1 = {127, 125, 123, 129, 131, 133, 0, 255, 127, 129, 1, 254};
        const std::vector<uint8_t> boundaryInput2(boundaryInput1.size(), 128);

        const QuantizedAddSpec outputTieSpec = {128, 128, 128, 1.0f, 1.0f, 2.0f, FusedActivation_kTfLiteActNone};
        if (!runCase(boundaryInput1, boundaryInput2, outputTieSpec, -1)) {
            return false;
        }

        const QuantizedAddSpec inputTieSpec = {
            128, 128, 128, 1.0f / static_cast<float>(1 << 20), 1.0f, 1.0f / 393216.0f, FusedActivation_kTfLiteActNone};
        if (!runCase(boundaryInput1, boundaryInput2, inputTieSpec, -2)) {
            return false;
        }

        constexpr int kRandomCaseCount = 32;
        constexpr int kWidth = 4037;
        constexpr size_t kElements = 4 * kWidth;
        uint32_t randomState = 0x4547u;
        std::vector<uint8_t> randomInput1(kElements);
        std::vector<uint8_t> randomInput2(kElements);
        for (int caseIndex = 0; caseIndex < kRandomCaseCount; ++caseIndex) {
            const int input1Exponent = static_cast<int>(nextRandom(randomState) % 21);
            const int input2Exponent = static_cast<int>(nextRandom(randomState) % 21);
            QuantizedAddSpec spec;
            spec.input1ZeroPoint = static_cast<int32_t>(nextRandom(randomState) & 0xff);
            spec.input2ZeroPoint = static_cast<int32_t>(nextRandom(randomState) & 0xff);
            spec.outputZeroPoint = 128;
            spec.input1Scale = std::ldexp(0.5f + 0.5f * randomUnit(randomState), -input1Exponent);
            spec.input2Scale = std::ldexp(0.5f + 0.5f * randomUnit(randomState), -input2Exponent);
            const float maxInputScale = std::max(spec.input1Scale, spec.input2Scale);
            spec.outputScale = maxInputScale * (0.5f + 3.5f * randomUnit(randomState));
            spec.activation = FusedActivation_kTfLiteActNone;
            for (size_t i = 0; i < kElements; ++i) {
                randomInput1[i] = static_cast<uint8_t>(nextRandom(randomState) >> 24);
                randomInput2[i] = static_cast<uint8_t>(nextRandom(randomState) >> 24);
            }
            if (!runCase(randomInput1, randomInput2, spec, caseIndex)) {
                return false;
            }
        }
        return true;
    }
};

MNNTestSuiteRegister(QuantizedAddTest, "op/quantized_add");

#endif
