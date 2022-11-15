include("../src/control_problem.jl")
include("../src/piql_pop.jl")
include("../src/ControlProblems/pendulum_problem.jl")
include("../src/Actors/empty_actor.jl")
include("../src/Actors/chain_actor.jl")

let m = 1.0, g = 9.8, L = 1.0
    global H_pendulum(θ,p) = m*g*L*cos(θ) + p^2/(2 * m * L^2)
    global ctrl = pendulum_control_problem(; g, m1 = m, L, 
        δt = .01, control_force = 1.0, damping = 0.1, potential_sharpness = 1)
end

function ignoretime(u)
    v = Float32.(u)
    deleteat!(v,3)
    return v
end

sc = SimpleChain(static(4),  TurboDense(tanh, 32),
TurboDense(tanh, 16), TurboDense(identity, 1))
x,y = make_xy(pop.memory,prefun = ignoretime)
sc_loss= SimpleChains.add_loss(sc,ContrastiveCrossEntropyLoss(y))

params = SimpleChains.init_params(sc_loss)
grads = similar(params)
SimpleChains.valgrad!(grads, sc_loss, x, params)



ca = init_chain_actor( 
  TurboDense(tanh, 32),
  TurboDense(tanh, 16); prefun = ignoretime, ndims = 4 # ignoring time, state is 2D, action is 1D, + β
)



pop = initial_piql(ctrl, ca)
timestep_pop!(pop, ctrl, ca)

generate_data!(pop, ctrl, ca)
train!(ca,pop.memory, grad_steps = 1000)
ca.grads



# initialize population
function training_epoch(pop,ctrl,actor)
    while 
        generate_data!(pop, ctrl, actor)
        (x,y) = sample_xy(pop.history)
        loss = train!(actor,x,y)
        pop.λ
    end
end

# run population and train network

# ?-> What happens with terminal states? Depends what the situation is...
# 
# if most of the population is in a terminal state, population restarts.