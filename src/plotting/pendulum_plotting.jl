using GLMakie
using Makie.Colors
using Makie

include("../control_problem.jl")
include("../piql_pop.jl")
include("../ControlProblems/pendulum_problem.jl")
include("../Actors/empty_actor.jl")
include("../Actors/chain_actor.jl")


#
#
# Define the control problem
#
#

#begin
let m = 1.0, g = 9.8, δt = .01, L = 1.0, size = 5000
    global H_pendulum(θ,p) = m*g*L*cos(θ) + p^2/(2 * m * L^2)
    global ctrl = pendulum_control_problem(; g, m1 = m, L, 
    δt, control_force = 1.0, damping = 0.003, potential_sharpness = 1)
    function ignoretime(u)
        v = Float32.(u)
        θ = v[1]
        #psign = 2*(v[2] > 0) - 1
        v[1] = cos(θ)
        v[3] = sin(θ) #* psign
        #v[4] *= psign 
        #v[2] *= psign
        return v
    end
    global actor = init_chain_actor( 
      TurboDense(tanh, 32),
      TurboDense(tanh, 8); prefun = ignoretime, ndims = 5 # ignoring time, state is 2D, action is 1D, + β
    )
    global pop = initial_piql(ctrl, actor; size, memory_size = 2*10^3)
    println("time horizon = Ne = $(size*sqrt(size)*δt)")
    println("period = T = $(2*π*sqrt(L/g))")
end



#actor  = empty_actor()

function getphasepoint(pop)
    [Point2f(getfield(sa,:state)[[1,2]]) for sa in pop.living]
end

function getgradient(chain_actor::ChainActor; β = chain_actor.β, ε =10^-6,  
    x_c = LinRange(-pi,pi,101),
    y_c = LinRange(-10,10,101))
    βold = chain_actor.β
    chain_actor.β = β
    out = [(chain_actor([x,y,0],[ε]) - chain_actor([x,y,0],[-ε]))/ε for x in x_c, y in y_c]
    chain_actor.β = βold
    return out
end
x_c = LinRange(-pi,pi,101)
y_c = LinRange(-10,10,101)
#end;


#
# create an energy contour plot
#
pts = Observable(getphasepoint(pop))
ctr_grad = Observable(getgradient(actor))
c_mat = H_pendulum.(x_c,reshape(y_c,1,:))
Makie.inline!(false)
fig = Figure()
ax1 = Axis(fig[1, 1])
ax2 = Axis(fig[1, 2])
contourf!(ax1, x_c, y_c, c_mat, levels = 10)
contourf!(ax2, x_c, y_c, ctr_grad, levels = 10)
scatter!(ax1, pts; color = :white)
ax3 = Axis(fig[2, 1])
ax4 = Axis(fig[2, 2])
ylims!(ax3, 0, 5)
ylims!(ax4, 0, 5)
betavec = Observable([actor.β])
performance = Observable([estimate_performance(actor, ctrl; number = 3, β = 1.0)])
performance10 = Observable([estimate_performance(actor, ctrl; number = 3, β = 10.0)])
lines!(ax3, 1:length(betavec[]) , betavec)
lines!(ax4,1:length(performance[]) , performance)
lines!(ax4,1:length(performance10[]),performance10)


# for ii in 1:20000
#     timestep_pop!(pop, ctrl, actor)
#     pts[] = getphasepoint(pop)
#     sleep(.001)
#     if mod(ii,200) == 0
#         train!(actor,pop.memory, grad_steps = 3000)
#         ctr_grad[] = getgradient(actor)
#     end
#     push!(βvec,pop.β)
# end
for ii in 1:1*10^2
    timestep_pop!(pop, ctrl, actor)
    if mod(ii,5) == 0
        pts[] = getphasepoint(pop)
        sleep(.001)
    end
end


#record(fig,"src/plotting/movies/control_learning2.mp4", 1:2*10^4; framerate = 500, sleep = false) do ii
for ii in 1:10*10^2
    timestep_pop!(pop, ctrl, actor)
    if mod(ii,5) == 0
        pts[] = getphasepoint(pop)
        sleep(.0001)
        train!(actor,pop.memory, grad_steps = 50)
    end
    if mod(ii,200) == 0
        ctr_grad[] = asinh.(getgradient(actor))
        push!(betavec[], actor.β) #appending data 
        push!(performance[], estimate_performance(actor, ctrl; number = 3, β = 1.0))
        push!(performance10[], estimate_performance(actor, ctrl; number = 3, β = 10.0))
        autolimits!(ax3)
        autolimits!(ax4)
        betavec[] = betavec[]
        performance[] = performance[]
        performance10[] = performance10[]
        sleep(.0001)
    end
end



fig = Figure()
Axis(fig[1, 1])
contourf!(x_c, y_c, c_mat, levels = 10)
scatter!(pts; color = :white)
record(fig,"src/plotting/movies/stabilized.mp4", 1:10^3; framerate = 500) do ii
    timestep_pop!(pop, ctrl, actor)
    pts[] = getphasepoint(pop)
end
#
# evolve a population and plot the points over time
# 

points = Observable(Point2f[randn(2)])

fig, ax = scatter(points)
limits!(ax, -4, 4, -4, 4)

fps = 60
nframes = 120

for i = 1:nframes
    new_point = Point2f(randn(2))
    points[] = push!(points[], new_point)
    sleep(1/fps) # refreshes the display!
end

####


q_observable = Observable([0.0,1.1]) 

@lift θ_ob = $q_observable[1]
@lift p_ob = $q_observable[2]

f = Figure()
ax = Axis(f[1, 1], aspect = 2)
GLMakie.xlims!(ax, -6, 6)
GLMakie.ylims!(ax, -3, 3)
@lift begin 
    empty!(ax)
    (x,θ) = $q_observable
    c = Point2f(x,0.0)
    d = Point2f(1.0, 0.5)
    e = Point2f(2*sin(θ), 2*cos(θ))
    poly!(Rect2{Float64}(c-d, 2d), color = :gray)
    lines!([c,c+e]; linewidth = 10, color = :black)
    poly!(Circle(c+e, .5), color = :brown)
    f
end


framerate = 30
timestamps = range(0, 10, step=1/framerate)



cart_animate(name, traj) = record(f, "$name.mp4", traj;
        framerate = framerate) do x
    q_observable[] = x
end


