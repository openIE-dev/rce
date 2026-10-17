import RCE.Core.F64
/-! Kernel-checked vectors for the unary float methods (bits from Rust/IEEE). -/
namespace RCE.F64
set_option maxRecDepth 200000
example : sqrt 4611686018427387904 = some 4609047870845172685 := by decide   -- √2
example : sqrt 9094988921128908188 = some 6850974717710472879 := by decide   -- √1e300
example : sqrt 1 = some 2188749418902061056 := by decide                    -- √5e-324
example : sqrt 9223372036854775808 = some 9223372036854775808 := by decide  -- √-0 = -0
example : sqrt 13830554455654793216 = none := by decide                     -- √-1 traps
example : toIntegral .floor 13826050856027422720 = some 13830554455654793216 := by decide  -- floor -0.5 = -1
example : toIntegral .ceil 13826050856027422720 = some 9223372036854775808 := by decide    -- ceil -0.5 = -0
example : toIntegral .away 4612811918334230528 = some 4613937818241073152 := by decide     -- round 2.5 = 3
example : toIntegral .away 13824249416176474522 = some 9223372036854775808 := by decide    -- round -0.4 = -0
example : toIntegral .floor 4609434218613702656 = some 4607182418800017408 := by decide    -- floor 1.5 = 1
example : abs 13830554455654793216 = 4607182418800017408 := by decide
end RCE.F64
