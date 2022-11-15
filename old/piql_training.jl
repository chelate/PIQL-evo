using Flux
using Zygote
using Flux.Data: DataLoader
using Graphs: vertices
using StatsFuns: logsumexp, log1pexp


function exponential_beta_schedule(ii; beta_start = 1.0, beta_end = beta_start, beta_steps = 1)
    β = exp(log(beta_start) + ii*(log(beta_end) - log(beta_start))/beta_steps)
    return β
end

function train_piql(
    cntr::ControlProblem,
    actor,
    sim_steps::Int;
    n_samples::Int = 10^3,
    n_steps::Int = 10,
    resample = true,
    piqldata = brine_piql(cntr, actor; n_samples, n_steps, resample),
    beta_start = cntr.β,
    beta_end = beta_start,
    beta_schedule = ii -> exponential_beta_schedule(ii; beta_start, beta_end, beta_steps = sim_steps)
)
    for ii = 1:sim_steps
        piqldata =
            rerun_piql(piqldata, cntr, actor; n_samples=n_samples, n_steps=n_steps, resample=resample)
        actor_training(piqldata, actor, cntr)
        β = beta_schedule(ii)
        cntr = set_β(cntr, β)
        println(cntr.β)
    end
    return actor, piqldata
end

function bounded_exp(x; bound = log(10.0) )
    exp(bound - log1pexp(bound - x))
end



function actor_training(piqldat::PiqlData, actor::Tabular_actor, ctrl::ControlProblem)
    β = ctrl.β
    for v in vertices(piqldat.tree)
        #println(v)
        #println(haskey(piqldat.multiplicity, v))
        #println(piqldat.multiplicity)
        if !ctrl.terminal_condition(piqldat.data[v].state)
            event = piqldat.data[v]
            log_m = piqldat.log_multiplicity[v] + β * (event.E_critic - event.E_actor)
            log_w = piqldat.log_weight[v]
            m = bounded_exp(log_m - log_w)
            e = actor(event.state, event.action)
            E_c = event.E_critic
            g = (m - ((1+m) / (1 + exp(β * (e - E_c))))) / β
            if isnan(g)
                println("m = $(log_m)")
                println("w = $(log_w)")
                println("E_A = $(E_A)")
                println("e = $(e)")
            end


            actor.update(actor, event.state, event.action, g)
        end
    end

end

function actor_training(piqldat::PiqlData, actor::Deep_actor, ctrl::ControlProblem)
    #loss(x, y) = Flux.Losses.mse(actor.model(x), y)
    loss(x, y) = actor.lossfunction(actor.model(x), y)
    β = ctrl.β

    #@time loss(x, y) = piql_loss(actor.model(x), y)
    ps = actor.trainparameters
    X = [append!(deepcopy(piqldat.data[v].state), piqldat.data[v].action) for v in vertices(piqldat.tree)
                if !ctrl.terminal_condition(piqldat.data[v].state)]
    
    Y = [
        [piqldat.log_multiplicity[v], piqldat.log_weight[v], piqldat.data[v].E_actor, β] for v in vertices(piqldat.tree)
            if !ctrl.terminal_condition(piqldat.data[v].state)]
    X = reduce(hcat, X)
    Y = reduce(hcat, Y)

    #data=DataLoader((X, Y), batchsize=length(Y[1,:]),shuffle=true)  ## need to figure out how to make this a batch_size thing
    data = DataLoader((X, Y), batchsize = 32, shuffle = true)  ## need to figure out how to make this a batch_size thing

    for d in data
        #print(d)
        train_loss, back = Zygote.pullback(() -> loss(d...), ps)

        gs = back(one(train_loss))
        actor.update(actor, gs)
    end


end
