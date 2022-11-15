include("../tabular/Enviroment/gridworld.jl")

mutable struct Tabular_actor
    energy::Dict{Tuple{Vector{},Vector{}},Float64}
    visits::Dict{Tuple{Vector{},Vector{}},Int}
    update::Function  ## updates energies and visits according the the learning rule
    hyperparameters::Dict{String,Vector{Float64}}
end

function (a::Tabular_actor)(state,action)
    #println(a.energy)
    if haskey(a.hyperparameters,"round_max")
        state=state.+a.hyperparameters["round_min"]
        state=mod.(state./(a.hyperparameters["round_max"].-a.hyperparameters["round_min"]), 1.0)
        state=(floor.(state.*a.hyperparameters["round_bins"]))
    end
    if !haskey(a.energy,(state,action))
        push!(a.energy, (state,action)=> 0.0)
    end
    #println(a.energy)
    return a.energy[(state,action)]
end


#


function initilize_tabular_dict(states,actions,value)
    a= Dict{Tuple{Int,Int},Float64}
    merge!(a, Dict((1,1)=> 1))
    for ii in states
        for jj in actions
            a=merge!(a, Dict((ii,jj)=> value))
        end
    end
    return a
end


function make_tabular_actor(learning_rate = visits -> 1/(1+visits), hyperparameters)
    function tabular_update_lr(actor::Tabular_actor,s,a,g; learning_rate)
        if haskey(actor_hyper,"round_max")
            s=s.+actor_hyper["round_min"]
            s=mod.(s./(actor_hyper["round_max"].-actor_hyper["round_min"]), 1.0)
            s=(floor.(s.*actor_hyper["round_bins"]))
        end
        if haskey(actor.visits,(s,a))
            actor.visits[ (s,a)] += 1
        else
            push!(actor.visits,(s,a)=>1)
        end
        visits = actor.visits[ (s,a)]
        if haskey(actor.energy,(s,a))
            actor.energy[ (s,a)] -= learning_rate(visits)* g
        else
            push!(actor.energy, (s,a)=>  -learning_rate(visits)* g)
        end
    end
    return Tabular_actor(
        Dict{Tuple{ Vector{}, Vector{} },Float64}(),
        Dict{Tuple{ Vector{}, Vector{}},Int64}(),
        tabular_update_lr, # f(actor, state, action, gradient)
        hyperparameters
        )
end

# update rules

# learning_rate_decay=actor.hyperparameters["learning_rate_decay"][1]
# alpha_q = actor.hyperparameters["learning_rate"][1] * (learning_rate_decay /(visits +learning_rate_decay))



function leraning_update_rule(learning_function)
    if haskey(actor_hyper,"round_max")
        s=s.+actor_hyper["round_min"]
        s=mod.(s./(actor_hyper["round_max"].-actor_hyper["round_min"]), 1.0) # add circularity to really enforce n_bins
        s=(floor.(s.*actor_hyper["round_bins"]))
    end


function make_softQL_actor(gridworld::Gridworld,actor_hyper::Dict{String,Vector{Float64}})
    function softQL_update(actor::Tabular_actor,s,a,g)

        actor.visits[ (s,a)] += 1
        alpha_q = actor.visits[ (s,a)] ^(-actor.hyperparameters["omega"][1])
        actor.energy[ (s,a)]  -= alpha_q* g

    end


     return Tabular_actor(
        initilize_tabular_dict(1:length(gridworld.grid),(gridworld.actions),0.0),
        initilize_tabular_dict(1:length(gridworld.grid),(gridworld.actions),0),
        softQL_update,
        actor_hyper
        )
end



function make_dynamic_actor()
    println("using a def dynamic actor")
    actor_hyper=Dict("learning_rate"=>[1.0],"learning_rate_decay"=>[1000.0])
    return make_dynamic_actor(actor_hyper)
end



function make_dynamic_actor(actor_hyper::Dict{String,Vector{Float64}})
    if haskey(actor_hyper,"learning_rate")
        if !haskey(actor_hyper,"learning_rate_decay")
            push!(actor_hyper,"learning_rate_decay"=>[1000.0])
            println("set default decay of the learning rate to 1000/(1000+ visits)")
            println("set the key learning_rate_decay to change this")
        end
        println("You are using a actor with learning rate $(actor_hyper["learning_rate"]) and decay s/(s+visits) with s=$(actor_hyper["learning_rate_decay"])")
        return make_dynamic_actor_learning_rate(actor_hyper)
    elseif askey(actor_hyper,"omega")
        println("You are using a actor with learning_rate 1/ visits^(omega) with omega = $(actor_hyper["omega"]) ")
        return make_dynamic_actor_stand_learning_decay(actor_hyper)
    else
        println("please either def the learing rate with key learning_rate or the decay omega")
        return -1
    end

end

function reset_visits(actor::Tabular_actor)
        actor.visits=typeof(actor.visits)()()
end



function make_dynamic_actor_stand_learning_decay(actor_hyper::Dict{String,Vector{Float64}})
    function softQL_update(actor::Tabular_actor,s,a,g)
        if haskey(actor_hyper,"round_max")
            s=s.+actor_hyper["round_min"]
            s=mod.(s./(actor_hyper["round_max"].-actor_hyper["round_min"]), 1.0) # add circularity to really enforce n_bins
            s=(floor.(s.*actor_hyper["round_bins"]))
        end
        if haskey(actor.visits,(s,a)) # if a site has been visisted already
            actor.visits[ (s,a)] += 1
        else
            push!(actor.visits,(s,a)=>1)
        end
        alpha_q = actor.visits[ (s,a)] ^(-actor.hyperparameters["omega"][1])
        if haskey(actor.energy,(s,a))
            actor.energy[ (s,a)] -= alpha_q* g
        else
            push!(actor.energy, (s,a)=>  - alpha_q* g)
        end
    end
     return Tabular_actor(
        Dict{Tuple{ Vector{}, Vector{} },Float64}(),
        Dict{Tuple{ Vector{}, Vector{}},Int64}(),
        softQL_update,
        actor_hyper
        )
end



function make_dynamic_actor_learning_rate(actor_hyper::Dict{String,Vector{Float64}})
    function tabular_update_lr(actor::Tabular_actor,s,a,g)
        if haskey(actor_hyper,"round_max")
            s=s.+actor_hyper["round_min"]
            s=mod.(s./(actor_hyper["round_max"].-actor_hyper["round_min"]), 1.0)
            s=(floor.(s.*actor_hyper["round_bins"]))
        end
        if haskey(actor.visits,(s,a))
            actor.visits[ (s,a)] += 1
        else
            push!(actor.visits,(s,a)=>1)
        end
        visits=actor.visits[ (s,a)]
        learning_rate_decay=actor.hyperparameters["learning_rate_decay"][1]
        alpha_q = actor.hyperparameters["learning_rate"][1] * (learning_rate_decay /(visits +learning_rate_decay))
        if haskey(actor.energy,(s,a))
            actor.energy[ (s,a)] -= alpha_q* g
        else
            push!(actor.energy, (s,a)=>  - alpha_q* g)
        end
    end
    return Tabular_actor(
        Dict{Tuple{ Vector{}, Vector{}},Float64}(),
        Dict{Tuple{ Vector{}, Vector{}},Int64}(),
        tabular_update_lr,
        actor_hyper
        )
end
