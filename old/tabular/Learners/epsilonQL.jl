include("../Enviroment/gridworld.jl")
include("../Enviroment/agent.jl")

# Normal epsilon gridy QL_System

mutable struct QL_system
    epsilon::Float64
    q_table::Array{Float64,2}
    n_table::Array{Int64,2}
    actions::UnitRange{Int64}

    gamma::Float64
    omega::Float64

end


function initilize_RL_system(grid::Gridworld, epsilon, gamma, omega)
    grid_size = length(grid.grid)
    num_act = length(grid.actions)
    actions = range(1, length = num_act)
    q_tabel = zeros(num_act, grid_size)
    n_tabel = zeros(num_act, grid_size)
    return QL_system(epsilon, q_tabel, n_tabel, actions, gamma, omega)
end


function choose_action(QL::QL_system, actor::agent)

    # epsilon-greedy

    if (rand() <= QL.epsilon)
        return rand(QL.actions)
    else
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
end

function greedy_action(QL::QL_system, actor::agent)

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

function update_System(QL::QL_system, actor::agent, action, reward)
    QL.n_table[action, actor.old_position] += 1

    alpha_q = QL.n_table[action, actor.old_position]^(-QL.omega)

    old_q = QL.q_table[action, actor.old_position]
    next_qs = QL.q_table[:, actor.position]
    new_q = old_q + alpha_q * (reward + QL.gamma * maximum(next_qs) - old_q)

    QL.q_table[action, actor.old_position] = new_q
end
