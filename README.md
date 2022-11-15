Basis use case: first you build a ControlProblem struct.
This defines completely the control problem of interest.


```julia

cntrl = make_gridworld(parameters)::ControlProblem # returns ControlProblem type with fields


step_direction = Dict(1=>(0,1),2=>(1,0),3=>(-1,0),4=>(0,-1))


function make_gridworld(parmaeters)
    gridworld = gridworld_constructor(paramters)
    control_prior(x) = StateVector(1:4,zeros(4)/4)
    function propagator(x::Tuple{Int64,Int64}, a::Int64)
        new = x .+ step_directions[a]
        if gridworld.walls(new)
            return x
        else
            return new
        end
    end
    cost_function(x0, x1, a) = 1
    return ControlProblem(
        control_prior,
        propagator,
        cost_function,
        gridworld.terminal_cond,
        1,
        1
        )
end

struct ControlProblem{U, P, C, T, W}
    control_prior::U # c(x) -> StateVector{Actions}
    propagator::P # p(x0, a) -> x1 ("random" state)
    cost_function::C # c(x0, a, x1) -> Cost ::Float64
    terminal_condition::T # T(x) -> bool
    initial_state::W # W() -> x0 generates inital states of interest
    temperature::Float64 # positive number parameterizing kl-cost of prior deviation
    gamma::Float64 # positive number discount over time
end
```

The next object constructs the data for trianing. This requires something like


```julia
# actor(state,action) = energy, -qfunction or whatever future cost to go
# current estimate

data = QLearningData(cntrl::ControlProblem, actor<:Actor; other_pars... )

function QLearningDataStepper(cntrl::ControlProblem, actor<:Actor;
        sample_number = 1, start  = cntrl.inital_state())
    β,γ = (cntrl.β, contrl.γ)
    new_control = tilt(a -> actor(start, a) cntrl.prior(start))
    (action, actor_energy) = sample(new_control)
    new_state = cntrl.propagator(start, action)
    if sample_number == 0
        critic_energy = cntrl.cost_function(start, action, new_state) + free_energy(new_state, actor, cntrl)
    else
        critic_energy = mean(
            cntrl.cost_function(start, action, new_state_alt) + free_energy(new_state, actor, cntrl)
            # need some care
            for new_state_alt in [cntrl.propagator(start, action) for ii in 1:sample_number]
            ) #something like this
    end
    out = (start, action, actor_energy, critic_energy)  
    return (new_state, out)  
end
```

need one more class of functions for training actors on Q-type data streams.

```julia
# actor(state,action) = energy, -qfunction or whatever future cost to go
# current estimate

function actor_gradients(pdat::PiqlData, actor, cntrl)
    β = cntrl.β
    function grad(v)
        sa = pdat.data[v]
        ee = sa.actor_energy
        e = actor(sa.state, sa.action)
        m = pdat.multiplicity[v]
        w = pdat.multiplicity[v]
        g = m β - ((m + w) β)/(1 + exp(β (-ee + e))) # hopefully correct??
        return (sa.state, sa.action, g)
    end
    return (grad(v) for v in vertices(pdat.tree))
end

function actor_gradients(pdat::QlData, actor, cntrl)
    # some other generator of (state, action, g)
end


function train!(actor, pdat::PiqlData, cntrl; learning_rate = 1)
    for (s, a, g) in actor_gradients(pdat::PiqlData, actor, cntrl)
        update(actor, s, a, g; learning_rate) # needs to dispatch on different actor types.
    end
end
```