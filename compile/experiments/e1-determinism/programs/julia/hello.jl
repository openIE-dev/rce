# Julia compiles a method when it is first called. The harness writes the
# LLVM IR and the native code of these methods to files and compares them.
# E1_DEBUGINFO is "default" (source lines, with the file's path) or "none".
sumsq(n::Int) = sum(i * i for i in 1:n)
saxpy(a::Float64, x::Vector{Float64}, y::Vector{Float64}) = a .* x .+ y

using InteractiveUtils
const DI = Symbol(get(ENV, "E1_DEBUGINFO", "default"))
open("sumsq.ll", "w") do io
    code_llvm(io, sumsq, (Int,); debuginfo = DI)
end
open("sumsq.s", "w") do io
    code_native(io, sumsq, (Int,); debuginfo = DI)
end
open("saxpy.ll", "w") do io
    code_llvm(io, saxpy, (Float64, Vector{Float64}, Vector{Float64}); debuginfo = DI)
end
