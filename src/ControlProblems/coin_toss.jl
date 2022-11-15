
export coin_control_problem

function heads(state::State, u::Integer) # increment heads of coin u
    step = ntuple(i -> i == 2u-1,length(state.x)) # one hot
    return State(state.t-1,state.x .+ step)
end

function tails(state::State, u::Integer) # increment tails of coin u
    step = ntuple(i -> i == 2u,length(state.x)) # one hot
    return State(state.t-1,state.x .+ step)
end

function coin_cost(state0::S, state1::S) where S
    val = ntuple(i -> mod(i+1,2),length(state0.x)) # alternating values
    sum((state1.x .- state0.x) .* val)
end

function coin_prob(state, u::Integer; prior = (1/2,1/2) )
    (j,k) = state.x[u*2-1:u*2] .+ prior
    return j/(j+k)
end

function coin_propagator(state, u::Integer ; prior = (1/2,1/2))
    p = coin_prob(state, u; prior)
    states = [heads(state,u),tails(state,u)]
    w = weights([p,1-p])
    return StateVector(states, w)
end

function coin_prior(x; ncoins = 2) # prior preference on the control choice
    control = collect(1:ncoins) # constant prior on control
    w = weights(fill(1/ncoins, ncoins))
    return StateVector(control, w)
end

function coin_control_problem(temperature;
         ncoins = 2, prior_fun = (u -> (1/2,1/2)))
    # control_prior
    # propagator
    # cost_function
    # terminal_condition
    # terminal_cost
    # temperature
    ControlProblem(
    x->coin_prior(x; ncoins),
    (x,u) -> coin_propagator(x, u ; prior = prior_fun(u)),
    (x1,x2)->coin_cost(x1,x2),
    x -> x.t <= 0, # terminal_condition is t = 0
    x -> 0.0, #terminal_cost is zero
    Float64(temperature)
    )
end