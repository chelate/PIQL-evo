
export ControlProblem,StateAction,initial_state_action,new_state_action
using Statistics, Setfield
using StatsBase: weights
"""
This is a set of functions that completely define a KL-control problem
"""
struct ControlProblem{A, U, P, C, T, W}
    action_space::Vector{A} # something that we can iterate over
    action_prior::U # π(s,a) -> Float64 exactly like energy
    propagator::P # p(x0, a) -> x1 ("random" state (could be detemrinistic))
    cost_function::C # c(x0, a, x1) -> Cost ::Float64
    terminal_condition::T # T(x) -> bool
    initial_state::W # W() -> x0 generates inital states of interest
end


struct StateAction{S,A} # static and constructed on forward pass
    # atomic unit of data for all reinforcement learning
    state::S
    action::A
    E_actor::Float64
end


### Control problem

function initial_state_action(ctrl::ControlProblem, actor; critic_samples = 1)
    #function initial_state_action(ctrl, actor; critic_samples = 1)
        # begin a trajectory from the initial_state distributon
        # return a StateAction object
        state = ctrl.initial_state() # new_state
        (action, E_actor) = choose_action(state, ctrl, actor) # new_action
        return StateAction(state, action, E_actor)
end
    

function new_state_action(sa::StateAction{S,A}, ctrl::ControlProblem, actor) where {S,A}
    # atomic unit of state evolution
    state = ctrl.propagator(sa.state, sa.action) # new_state
    (action, E_actor) = choose_action(state, ctrl, actor) # new_action
    if isnan(E_actor)
        error("the E_actor is going bad")
        #break
    end
    return StateAction{S,A}(state, action, E_actor) # Let the compiler know that it is type invariant
end



## change of sign here becuase sampling seams wrong
function choose_action(state, ctrl, actor)
    priors = [ctrl.action_prior(state,a) for a in ctrl.action_space]
    priors = priors ./ sum(priors)
    energies = [actor(state,a) for a in ctrl.action_space]
    emin = minimum(energies)
    ii = sample(weights(priors .* exp.(- actor.β .* (energies .- emin))))
    return (ctrl.action_space[ii], energies[ii])
end

function energy_critic(state, action, ctrl, actor; critic_samples = 1)
    function energy_sample(new_state)
        ctrl.cost_function(state, action, new_state) + free_energy(new_state, ctrl, actor)
    end
    # helper function capturing input variables

    if ctrl.terminal_condition(state)
        # here we catch the terminal state
        return zero(Float64)
    else
        return mean(energy_sample(ctrl.propagator(state, action)) for ii in 1:critic_samples)
    end
end

## again sign was wrong I think
function free_energy(state, ctrl, actor)
    priors=[ctrl.action_prior(state, a) for a in ctrl.action_space]
    prior_normalization=sum(priors)
    #prior_normalization = sum( ctrl.action_prior(state, a) for a in ctrl.action_space)
    energies=[actor(state,a) for a in ctrl.action_space]
    emin=minimum(energies)
    # z = sum( ctrl.action_prior(state, a) * exp(-actor.β * actor(state,a))
    #    for a in ctrl.action_space) / prior_normalization
    z = sum(priors.* exp.(-actor.β .*(energies .-emin)))/prior_normalization
    #z = sum( ctrl.action_prior(state, a) * exp(-actor.β * actor(state,a))
    #    for a in ctrl.action_space) / prior_normalization
    if isnan(z)
        println(energies)
        println(emin)
        println(exp.(-actor.β .*(energies .-emin)))
    end

    #return - log(z) / actor.β
    return - log(z)/ actor.β + emin
end


