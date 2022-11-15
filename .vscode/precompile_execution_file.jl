using GLMakie
using Makie.Colors
using SimpleChains
using ElasticArrays


let m = 1.0, g = 9.8, L = 1.0
    global H_pendulum(θ,p) = m*g*L*cos(θ) + p^2/(2 * m * L^2)
end


x_c = LinRange(-pi,pi,201)
y_c = LinRange(-10,10,201)
c_mat = H_pendulum.(x_c,reshape(y_c,1,:))
fig = Figure()
Axis(fig[1, 1])
contourf!(x_c, y_c, c_mat, levels = 10)

sc = SimpleChain(static(ndims), TurboDense(tanh, 12), TurboDense(identity, 1))