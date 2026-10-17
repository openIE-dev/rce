import RCE.Core.F32Fmt
namespace RCE.F64
set_option maxRecDepth 200000
example : to32 4591870180066957722 = some 4591870180174331904 := by decide            -- 0.1 as f32
example : add32 4591870180174331904 4596373779801702400 = some 4599075939685498880 := by decide  -- 0.1f+0.2f
example : text32 4599075939685498880 = "0.3" := by decide
example : ofInt32 16777217 = some 4715268809856909312 := by decide                     -- ties to even
example : text32 4715268809856909312 = "16777216" := by decide
example : ofDec32 1 20 = some 4906019910326419456 := by decide
example : text32 4906019910326419456 = "100000000000000000000" := by decide
example : div32 4607182418800017408 (4613937818241073152) = some 4599676419600023552 := by decide  -- 1/3
example : text32 4599676419600023552 = "0.33333334" := by decide
example : text32 4502148214576644096 = "0.0000001" := by decide
example : ofInt32 (2 ^ 128) = none := by decide
end RCE.F64
