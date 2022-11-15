

function estimate_performance(actor, ctrl::ControlProblem; number = 10, β = actor.β)
    actor.β = β
    total_state_cost = 0.0
    var_state_cost = 0.0
    total_control_cost = 0.0
    var_control_cost = 0.0
    for _ in 1:number
        run_state_cost = 0.0
        run_control_cost = 0.0
        sa = initial_state_action(ctrl, actor)
        while !ctrl.terminal_condition(sa.state)
            (sa, state_cost, control_cost) = new_state_action_cost(sa, ctrl, actor)
            run_state_cost += state_cost
            run_control_cost += control_cost
        end
        total_state_cost += run_state_cost
        total_control_cost += run_control_cost
        var_state_cost += run_state_cost^2
        var_control_cost += run_control_cost^2
    end
    return  total_state_cost + total_control_cost
    # can return state and control cost etc.
    # will focus on control cost
    #((total_control_cost + total_control_cost)/number, 
    #    total_state_cost/number, 
    #    total_control_cost/number)
end


function new_state_action_cost(sa::StateAction{S,A}, ctrl::ControlProblem, actor) where {S,A}
    # atomic unit of state evolution
    state = ctrl.propagator(sa.state, sa.action) # new_state
    state_cost = ctrl.cost_function(sa.state, sa.action, state)
    (action, E_actor, control_cost) = choose_action_cost(state, ctrl, actor) # new_action + update control cost
    (return StateAction{S,A}(state, action, E_actor), state_cost, control_cost)  # Let the compiler know that it is type invariant
end

"""
cost of control = sum( p(a|E) β (F - E_a)   for a in actions)
since βF = -log Z 
where Z = sum π_a e^{-β E_a}
- sum( p(a|E) log(Z) + βE_a)
-log(Z) - sum( p(a|E) (β E_a) )

control measured in units of cost function = 
"""


function choose_action_cost(state, ctrl, actor)
    priors = [ctrl.action_prior(state,a) for a in ctrl.action_space]
    priors = priors ./ sum(priors)
    energies = [actor(state,a) for a in ctrl.action_space]
    emin = minimum(energies)
    ii = sample(weights(priors .* exp.(- actor.β .* (energies .- emin))))
    z = sum(priors .* exp.(- actor.β .* (energies .- emin)))
    e = sum(priors .* exp.(- actor.β .* (energies .- emin)) .* 
        (energies .- emin)) / z
    control_cost = - log(z) / actor.β - e
    return (ctrl.action_space[ii], energies[ii], control_cost)
end