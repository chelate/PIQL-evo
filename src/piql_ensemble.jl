"""
Expected structure
# multiple populations at different times and temperatures and states
actor 1     pop1 pop2 pop3  -learn from your mistakes
actor 2     pop4 pop5 pop6  -learn from your friend's knowledge, ignore their mistakes
control problem
"""


struct PiqlEnsemble{P,A,C}
    populations::Array{P}
    actors::Vector{A}
    ctrl::C
end

function get_memory(ens::PiqlEnsemble)
    reduce(vcat, p.memory for p in ens.populations)
end


function train_ensemble!(ens::PiqlEnsemble)
    memory = reduce(vcat, p.memory for p in ens.populations)
    for actor in ens.actors
        train!(actor,memory)
    end
end

function evolve_ensemble!(ens::PiqlEnsemble)
    for (actor, pops) in zip(ens.actors, eachcol(ens.populations))
        timestep_pop!(pop::PiqlPop, ctrl, actor)
    end
end

