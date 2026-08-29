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

function benchmark_bandits(eps::Float64, initialrewards::Vector{Float64} = zeros(Float64, 10))
    K = length(initialrewards)
    T = 1000
    L = 2000

    benchmark_rewards = zeros(Float64, T)
    fraction_optimal_choice = zeros(Float64, T)

    for _ in 1:L
        avgrewards = copy(initialrewards)
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
        optimal_choices = [(j == optimum) for j in choices]
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

    plot([b0 b1 b2], label=["0.0" "0.1" "0.01"])

    #plot([f0 f1 f2], label=["0.0" "0.1" "0.01"])
end

end

module T8
using ..T6
using Plots

function final()
    initial_rewards = fill(5.0, 10)
    b0, f0 = T6.benchmark_bandits(0.0, initial_rewards)
    b1, f1 = T6.benchmark_bandits(0.1, initial_rewards)

    #= TODO: optimality ratios are fine, 0.1 is better than 0.0 but for some reason
             0.0 has better rewards than 0.1. Something is broken as 0.1 should easily
             surpass it with noteworthily higher optimal hit ratios
    =#
    plot([b0 b1], label=["0.0" "0.1"])
    #plot([f0 f1], label=["0.0" "0.1"])    
end
end