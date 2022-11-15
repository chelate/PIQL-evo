using StatsBase
include("../Enviroment/gridworld.jl")
include("../Enviroment/agent.jl")

#Infromation reg Q learner
mutable struct MIRL_system

    q_table::Array{Float64,2}
    n_table::Array{Int64,2}
    actions::UnitRange{Int64}
    prior::Array{Float64,1}
    β::Float64


    gamma::Float64
    β_LR::Float64
    prior_LR::Float64
    omega::Float64

end


function initilize_RL_system(grid::Gridworld, β, gamma, β_LR, prior_LR, omega)
    grid_size = length(grid.grid)
    num_act = length(grid.actions)
    actions = range(1, length = num_act)
    q_tabel = zeros(num_act, grid_size)
    n_tabel = zeros(num_act, grid_size)
    prior = ones(num_act) / num_act
    return MIRL_system(q_tabel, n_tabel, actions, prior, β, gamma, β_LR, prior_LR, omega)
end


function choose_action(QL::MIRL_system, actor::agent)

    q_values = QL.q_table[:, actor.position]
    p_actions = give_wights(q_values, QL.prior, QL.β)
    return sample(QL.actions, Weights(p_actions))

end

function greedy_action(QL::MIRL_system, actor::agent)

    #choose random between all the qs that are max
    q_values = QL.q_table[:, actor.position]
    maxes = Vector{Int}()
    max_q = maximum(q_values)
    for ii = 1:length(q_values)
        if (q_values[ii] == max_q)
            append!(maxes, ii)
        end
    end
    return rand(maxes)
end

function give_wights(q_values, prior, β)
    maxq = maximum(q_values)
    unnormd = exp.(β .* (q_values .- maxq)) #calculate exponentials but keep in range
    Z = sum(unnormd)
    return unnormd ./ Z
end

function give_free_energy(q_values, prior, β)
    maxq = maximum(q_values)
    unnormd = exp.(β .* (q_values .- maxq)) #calculate exponentials but keep in range
    Z = sum(unnormd)
    return log(Z) / β + maxq
end





function update_System(QL::MIRL_system, actor::agent, action, reward)


    "adaptive learning rate for q values: alpha_q = n(s, a)^{-omega}
     soft update: q_new = q_old * alpha_q [r + gamma * Free_energy(S_t+1, a) - q_old]
    "

    QL.n_table[action, actor.old_position] += 1

    alpha_q = QL.n_table[action, actor.old_position]^(-QL.omega)

    old_q = QL.q_table[action, actor.old_position]
    next_qs = QL.q_table[:, actor.position]
    t_soft = reward + QL.gamma * give_free_energy(next_qs, QL.prior, QL.β)
    new_q = old_q + alpha_q * (t_soft - old_q)

    new_prior =
        (1 - QL.prior_LR) .* QL.prior .+
        QL.prior_LR .* give_wights(QL.q_table[:, actor.old_position], QL.prior, QL.β)

    QL.q_table[action, actor.old_position] = new_q
    QL.prior = new_prior
    QL.β += QL.β_LR

end
