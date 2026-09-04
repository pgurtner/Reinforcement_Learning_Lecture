module T5
function choose(nt::Vector{Int}, qt::Vector{Float64}, t::Int)
    target = qt .+ sqrt.((2*log(t)) ./ nt)
    return argmax(target)
end
end

module T6
using Random
using Distributions
using CairoMakie
using ..T4
using ..Utils

@enum RewardUpdateType SampleAverage ConstantStepSize

function benchmark_bandits(eps::Float64, initial_rewards::Vector{Float64} = zeros(Float64, 10), reward_update_type::RewardUpdateType = SampleAverage)
    K = length(initial_rewards)
    T = 1000
    L = 2000

    benchmark_rewards = zeros(Float64, T)
    fraction_optimal_choice = zeros(Float64, T)

    for _ in 1:L
        avg_rewards = copy(initial_rewards)
        draws = zeros(Int, K)
        means = rand(Normal(0, 1), K)

        rewards = zeros(Float64, T)
        choices = zeros(Int, T)

        for t in 1:T
            j = T4.choose(avg_rewards, eps)
            r = T4.reward(j, means)
            if reward_update_type == SampleAverage
                T4.update_sample_avg!(j, r, draws, avg_rewards)
            else
                T4.update_step_size!(j, r, draws, avg_rewards, 0.1)
            end

            rewards[t] = r
            choices[t] = j
        end

        benchmark_rewards .+= rewards

        optimum = argmax(means)
        fraction_optimal_choice .+= (choices .== optimum)
    end

    benchmark_rewards ./= L
    fraction_optimal_choice ./= L

    return benchmark_rewards, fraction_optimal_choice
end

function final()
    b0, f0 = benchmark_bandits(0.0)
    b1, f1 = benchmark_bandits(0.1)
    b2, f2 = benchmark_bandits(0.01)

    fig1, ax1 = Utils.make_GL_plot("Rewards")
    lines!(ax1, 1:length(b0), b0, label="0.0")
    lines!(ax1, 1:length(b1), b1, label="0.1")
    lines!(ax1, 1:length(b2), b2, label="0.01")
    axislegend(position = :rb)
    display(fig1)

    fig2, ax2 = Utils.make_GL_plot("% optimal action")
    lines!(ax2, 1:length(f0), f0, label="0.0")
    lines!(ax2, 1:length(f1), f1, label="0.1")
    lines!(ax2, 1:length(f2), f2, label="0.01")
    axislegend(position = :rb)
    display(fig2)
end

end

module T8
using ..T6
using CairoMakie
using ..Utils

function final()
    b0, f0 = T6.benchmark_bandits(0.0, fill(5.0, 10), T6.ConstantStepSize)
    b1, f1 = T6.benchmark_bandits(0.1, fill(0.0, 10), T6.ConstantStepSize)

    # fig1, ax1 = Utils.make_GL_plot("Rewards")
    # lines!(ax1, 1:length(b0), b0, label="optimistic greedy")
    # lines!(ax1, 1:length(b1), b1, label="realistic, eps-greedy")
    # axislegend(position = :rb)
    # display(fig1)

    fig2, ax2 = Utils.make_GL_plot("% optimal action")
    lines!(ax2, 1:length(f0), f0, label="optimistic greedy")
    lines!(ax2, 1:length(f1), f1, label="realistic, eps-greedy")
    axislegend(position = :rb)
    display(fig2)
end
end