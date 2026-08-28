module T5
function choose(nt::Vector{Int}, qt::Vector{Float64}, t::Int)
    target = qt .+ sqrt.((2*log(t)) ./ nt)
    return argmax(target)
end
end

module T6
using Random
using Distributions
using Plots
using ..T4

function benchmark_bandits(eps)
    K = 10
    T = 1000
    L = 2000

    benchmark_rewards = zeros(Float64, T)
    fraction_optimal_choice = zeros(Float64, T)

    for _ in 1:L
        avgrewards = zeros(Float64, K)
        draws = zeros(Int, K)
        means = [rand(Normal(0, 1)) for _ in 1:K]

        rewards = zeros(Float64, T)
        choices = zeros(Int, T)

        for t in 1:T
            j = T4.choose(avgrewards, eps)
            r = T4.reward(j, means)
            T4.update!(j, r, draws, avgrewards)

            rewards[t] = r
            choices[t] = j
        end

        benchmark_rewards .+= rewards

        optimum = argmax(means)
        optimal_choices = [j == optimum for j in choices]
        fraction_optimal_choice .+= convert.(Float64, optimal_choices)
    end

    benchmark_rewards ./= L
    fraction_optimal_choice ./= L

    return [benchmark_rewards, fraction_optimal_choice]
end

function final()
    b0, f0 = benchmark_bandits(0.0)
    b1, f1 = benchmark_bandits(0.1)
    b2, f2 = benchmark_bandits(0.01)

    #plot([b0 b1 b2], label=["0.0" "0.1" "0.01"])

    plot([f0 f1 f2], label=["0.0" "0.1" "0.01"])
end

end 