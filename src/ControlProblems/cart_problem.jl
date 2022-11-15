export ControlProblem,StateAction,intial_state_action,new_state_action
using Statistics, Setfield

# state of cart state = (x, θ, p_x, p_θ)

## action_space::Vector{A} # something that we can iterate over
## action_prior::U # π(s,a) -> Float64 exactly like energy
## propagator::P # p(x0, a) -> x1 ("random" state)
## cost_function::C # c(x0, a, x1) -> Cost ::Float64
## terminal_condition::T # T(x) -> bool
# initial_state::W # W() -> x0 generates inital states of interest
# β::Float64 # positive number parameterizing kl-cost of prior deviation
# γ::Float64 # positive number discount over time

function cart_control_problem(; g = 9.8, m1 = 2.0 , m2 = 1.0, L = 0.5, 
    δt = .05, control_force = 1.0, β = 1.0, γ = 1.0, damping = 0.1 .* ones(2), potential_sharpness = 1)
    action_space = [[-1.],[0.],[1.]] .* control_force .* sqrt(δt)
    action_prior = (s,a) -> 1
    ControlProblem(
        action_space, action_prior,
        (state,action) -> cart_euler_step(state, action; g , m1, m2, L, δt, damping),
        (s0,a,s1) -> cart_cost(s0; potential_sharpness), 
        pendulum_stop, pendulum_start, β, γ
    )
end

function cart_cost(state0; potential_sharpness = 1)
    θ = state0[2]
    x = state0[1]
    return (1 - cos(θ))^(1/potential_sharpness) + x^2 / 50
end

function cart_start()
    [0.0, 0.0, 0.0, 0.0, 0.0]
end


# action space = [-1,0,1]

function cart_euler_step(state, action; g = 9.8, m1 = 2.0 , m2 = 1.0, L = 0.5, δt =0.01, damping = 0.1 .* ones(2)) 
    (x, θ, px, pθ, t) = state
    x_dot = (L * px - pθ * cos(θ)) / (L*m1 + L*m2 * sin(θ)^2) # h_px
    θ_dot = ((m1 + m2) * pθ - L*m2*px*cos(θ)) / (L^2*m2*(m1 + m2*sin(θ)^2)) # h_pθ
    px_dot = action[1] # h_x
    pθ_dot = g*L*m2*sin(θ) + (4(-(m1 + m2)*pθ + L*m2*px*cos(θ))*(L*px - pθ*cos(θ))*sin(θ))/(L^2*(2*m1 + m2 - m2*cos(2θ))^2) # h_θ
    newstate = [
        mod(x + x_dot * δt + 5.0 ,10.0) - 5.0, 
        θ + θ_dot * δt, 
        px + px_dot * δt - x_dot * damping[1] * δt, 
        pθ + pθ_dot * δt - θ_dot * damping[2] * δt,
        t + δt]
    return newstate
end

function cart_stop(state; θ_tol = 2*10^-2, pθ_tol = 2*10^-2, px_tol = 2*10^-2, end_time = 2*10^2)
    (x, θ, px, pθ, t) = state
    # return reduce(&, [abs(mod(θ - 0, 2*pi)) < θ_tol, abs(pθ - 0) < pθ_tol, abs(px - 0) < px_tol]) |
    return (t > end_time)
end

# function cart_hamiltonian(p, q, control; g = 9.8, m1 = 2.0 , m2 = 1.0, L = 0.5) 
#     (x, θ) = q
#     (p_x, p_θ) = p
#     h =  g * L * m2 * cos(θ) + 
#         (L^2 * m2 * p_x^2 + (m1 + m2)* p_θ^2 - 
#         2 *L * m2 * p_x * p_θ * cos(θ) ) / 
#         ( 2 * L^2*m2*(m1 + m2 * sin(θ)^2)) 
#         - control * x # this is the control, an added linear potential that kicks the system
#     return h
# end

# function initialize(p,q,c)
#     a = HamiltonianProblem(cart_hamiltonian,p,q,[0.0,1.0],c)
#     return init(a, Tsit5())
# end

# function update(s,c; dt = .2)
#    (t,p,q) = s
#    integrator = initialize(p,q,c)
#    step!(integrator, dt, true)
#    (p1,q1) = integrator.u
#    return (t+dt,p1,q1)
# end

# @time update((1.0,[0.1,0.2],[.3,.4]),.2)