include("control_problem.jl")
using Distributions
using DataStructures: CircularBuffer

"""
struct constituting a single unit of learning: 
two balanced samples taken from the same timepoint
What is stored as inseperable data from the evolving population
pop.memory = CircularBuffer{SelectionSamples}
"""


struct SelectionSample{S,A}
    sa1::StateAction{S,A}
    sa2::StateAction{S,A}
    ecrit1::Float64 #energy critic 1
    ecrit2::Float64 # energy critic 2
    β::Float64 
    # if possible always call β from pop
    # Why because β is generated by population parameters
    # β determines the evolutionary process
    # we link them in the dynamics
    λ::Float64 # λ at that time. measured in terms of the step time τ^-1.
end


"""
This is the set of y vales in the list of outputs.
"""

struct SampleContrast
    ΔEcrit::Float64
    ΔEact::Float64
    βeff::Float64
end

function sample_contrast(samp::SelectionSample)
    SampleContrast(
        samp.ecrit1 - samp.ecrit2,
        samp.sa1.E_actor - samp.sa2.E_actor,
        samp.β/samp.λ
        )
end

function contrast_loss(y::SampleContrast, Δε)
    # Δε = ε1-ε2
    (;ΔEcrit, ΔEact, βeff) = y # property destructuring 
    y_i = (1 + tanh(βeff * (ΔEcrit - ΔEact)))/2
    p_i = (1 + tanh(βeff * (Δε - ΔEact)))/2
    return  - y_i * log(p_i) - (1 - y_i) * log(1 - p_i)
end

function contrast_grad(y::SampleContrast, Δε)
    # Δε = ε1-ε2
    (;ΔEcrit, ΔEact, βeff) = y # property destructuring 
    y_i = (1 + tanh(βeff * (ΔEcrit - ΔEact)))/2
    p_i = (1 + tanh(βeff * (Δε - ΔEact)))/2
    return  βeff * (y_i - p_i)
end


function flatten(sample::SelectionSample; prefun = identity) 
    (;sa1,sa2,β) = sample
    return hcat(prefun(vcat(sa1.state, sa1.action, β)),
        prefun(vcat(sa2.state, sa2.action, β)))
end

function make_xy(memory; prefun = identity)
    x = reduce(hcat, flatten(sample; prefun) for sample in memory)
    y = [sample_contrast(sample) for sample in memory]
    return (x,y)
end


mutable struct PiqlPop{S,A}
    size::Int
    living::Vector{StateAction{S,A}}  # list of states
    memory::CircularBuffer{SelectionSample{S,A}} # stored selection events for training
    var_fit::Float64  # runnning average of the variance of the fitness
    λ::Float64 # lambda is fixed by the decay properties.
    α::Float64 # alpha sets the memory of the fitness variance, similar to the adam hyperparameters
    β::Float64 # the current temperature
end

"""
Construct a circular buffer of the correct type
"""

function similar_circular_buffer(::StateAction{S,A}, memory_size) where {S,A}
    CircularBuffer{SelectionSample{S,A}}(memory_size)
end

"""
λ/N = γ
γ * N = λ in units of δt
Λ = λ * N = γ * N^2 - rate of events
sparsity (= .05 events per particle, fraction of the population hit)
.01 / N = γ = 1000 frames, larger populations have longer memory
"""

function initial_piql(ctrl::ControlProblem, actor; 
    size = 10^2, memory_size = 2*10^3, var_fit = 10.0, λ = 1 / sqrt(size), β = 10^(-6), α = λ / 10)
    living = [initial_state_action(ctrl, actor) for _ in 1:size] # just screwed up the indices
    return PiqlPop(size, living, similar_circular_buffer(first(living), memory_size), var_fit, λ, α, β)
end

"""
Dynamics
"""


function timestep_pop!(pop::PiqlPop, ctrl, actor)
    terminal::Int = 0
    sel_events::Int = min(rand(Poisson(pop.size*pop.λ / 2)), ceil(Int,pop.size/2)) # the number of selection events in a time step
    for _ in 1:sel_events
        selection_event!(pop, ctrl, actor)
    end
    pop.β = pop.λ / (sqrt(pop.var_fit)) # rule of thumb ?
    actor.β = pop.β # just make the actor mutable
    # we expect that selection has become weak
    # our selected population should then be roughly the optimally controlled one
    for ii in eachindex(pop.living)
        if ctrl.terminal_condition(pop.living[ii].state) 
            # if you've reached the terminal condition
            # the dynamics end
            terminal += 1
        else
            pop.living[ii] = new_state_action(pop.living[ii], ctrl, actor)
        end
    end
    if terminal > pop.size / 2 # general restart condition
        pop.living = [initial_state_action(ctrl, actor) for _ in 1:pop.size]
        pop.var_fit = 1.0 # just screwed up the indices
    end
    return sel_events
end

"""
Selection 
"""


function selection_event!(pop::PiqlPop{S,A}, ctrl::ControlProblem, actor) where {S,A}
    i1 = rand(1:pop.size)
    i2 = rand(1:pop.size)
    if i1 != i2 
        sa1 = pop.living[i1]
        sa2 = pop.living[i2]
        if ctrl.terminal_condition(sa1.state) & ctrl.terminal_condition(sa1.state)
            return nothing
        end
        ec1 = energy_critic(sa1.state, sa1.action, ctrl, actor; critic_samples = 1)
        ec2 = energy_critic(sa2.state, sa2.action, ctrl, actor; critic_samples = 1)
        fit1 = (sa1.E_actor - ec1)  # when actor energy is higher than observed, 
        fit2 = (sa2.E_actor - ec2) # fitness is higher, the state did better than expected
        if one_beats(fit1, fit2, pop.β/pop.λ)
            pop.living[i2] = sa1
        else
            pop.living[i1] = sa2
        end
        sel = SelectionSample{S,A}(sa1, sa2, ec1, ec2, pop.β, pop.λ)
        push!(pop.memory, sel)
        pop.var_fit = (1-pop.α)*pop.var_fit + pop.α * (fit1 - fit2)^2
    end
    return nothing
end


function one_beats(fit1, fit2, β) # returns true if v1 beats v2
    rand(Bernoulli((1 + tanh((fit1-fit2) * β))/2))
end

function generate_data!(pop, ctrl, actor)
    # evolve population
    # get new data
    total_sels = 0
    while total_sels < length(pop.memory)
        (term, sels) = timestep_pop!(pop::PiqlPop, ctrl, actor)
        total_sels += sels
        if term > pop.size / 2 # general restart condition
            pop.living = [initial_state_action(ctrl, actor) for _ in 1:pop.size] # just screwed up the indices
        end
    end
end