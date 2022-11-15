using GLMakie
using Makie.Colors

q_observable = Observable([0.0,1.1]) 

@lift x_ob = $q_observable[1]
@lift θ_ob = $q_observable[2]

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
# shape decomposition

# function cart_euler_step(state, action; g = 9.8, m1 = 2.0 , m2 = 1.0, L = 0.5, δt =0.01, damping = 0.1 .* ones(2)) 
#     (x, θ, px, pθ) = state
#     x_dot = (L * px - pθ * cos(θ)) / (L*m1 + L*m2 * sin(θ)^2) # h_px
#     θ_dot = ((m1 + m2) * pθ - L*m2*px*cos(θ)) / (L^2*m2*(m1 + m2*sin(θ)^2)) # h_pθ
#     px_dot = action[1] # h_x
#     pθ_dot = g*L*m2*sin(θ) + (4(-(m1 + m2)*pθ + L*m2*px*cos(θ))*(L*px - pθ*cos(θ))*sin(θ))/(L^2*(2*m1 + m2 - m2*cos(2θ))^2) # h_θ
#     newstate = [
#         mod(x + x_dot * δt + 5.0 ,10.0) - 5.0, 
#         mod(θ + θ_dot * δt, 2.0 * pi), 
#         px + px_dot * δt - x_dot * damping[1] * δt, 
#         pθ + pθ_dot * δt - θ_dot * damping[2] * δt]
#     return newstate
# end



# traj=Vector{Float64}[]
# for ii in 1:1000
#    state = cart_euler_step(state,0)
#    if mod(ii,4) == 1
#         push!(traj,state[[1,2]])
#    end
# end
# cart_animate("damped", traj)