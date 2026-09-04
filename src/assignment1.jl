module T1
using Random
using Distributions
using Plots

function armed_bandit_greedy(m::Int, t::Int, arms::Vector{<:Distribution})
    @assert m > 0

    K = length(arms)
    rewards = zeros(Int, K)
    draws = fill(m, K)
    avgs = zeros(Float64, t, K)

    for i in 1:m
        for k in 1:K
            rewards[k] += rand(arms[k])
        end

        localdraws = i .* ones(Int, K)
        avgs[i, :] .= rewards ./ localdraws
    end

    for i in m+1:t
        k = argmax(avgs[i - 1, :])

        rewards[k] += rand(arms[k])
        draws[k] += 1

        avgs[i, :] .= rewards ./ draws
    end

    return avgs
end

function final()
    avgs = armed_bandit_greedy(10, 1000, [Bernoulli(0.5), Bernoulli(0.6)])

    plot(avgs)
end

end

module T2
using Plots
using Distributions
using ..T1



function final()
    J = 100
    m = 1000
    t = 1000

    arms = [Bernoulli(0.5), Bernoulli(0.6)]
    K = length(arms)
    
    avgs = zeros(Float64, t, K)
    for j in 1:J
        avgs .+= T1.armed_bandit_greedy(m, t, arms)
    end
    
    avgs ./= J

    plot(avgs)
end

end

module T4
using Random
using Distributions
using Plots

function choose(Qt::Vector{Float64}, eps::Float64)
    K = length(Qt)
    @assert K > 0

    optimum = argmax(Qt)

    probabilities = [
        if i == optimum
            (1 - eps) + eps/K
        else
            eps/K
        end
        for i in 1:K]
    distribution = Categorical(probabilities)
    
    rand(distribution)
end

function reward(arm::Int, means::Vector{Float64})
    @assert arm <= length(means)

    d = Normal(means[arm], 1)
    rand(d)
end

function update_sample_avg!(arm::Int, reward::Float64, draws::Vector{Int}, avgrewards::Vector{Float64})
    @assert arm <= length(draws)
    @assert length(draws) == length(avgrewards)

    draws[arm] += 1
    avgrewards[arm] = (avgrewards[arm]*(draws[arm] - 1) + reward) / draws[arm]
end

function update_step_size!(arm::Int, reward::Float64, draws::Vector{Int}, avgrewards::Vector{Float64}, α::Float64)
    @assert arm <= length(draws)
    @assert length(draws) == length(avgrewards)

    draws[arm] += 1
    avgrewards[arm] += α * (reward - avgrewards[arm])
end


function final()
    t = 1000
    means = [1.0, 1.1, 1.2, 1.3]
    eps = 0.1
    
    K = length(means)

    avgrewards = [reward(j, means) for j in 1:K]
    draws = ones(Int, K)

    for _ in 1:t
        j = choose(avgrewards, eps)
        r = reward(j, means)
        update!(j, r, draws, avgrewards)
    end
end

end