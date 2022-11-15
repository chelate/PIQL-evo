using Flux
using Zygote
using Flux.Data: DataLoader




function train_ql(cntr::ControlProblem, actor, particles, steps; critic_samples = 1)
    for ii = 1:steps
        data = Vector{typeof(particles[1])}()
        for ii = 1:length(particles)
            particle = particles[ii]
            results_data, new_state =KL_free_energy_DataStepper(cntr, actor, particle)
            particles[ii] = new_state

            push!(data, results_data)
        end
        qldat = QLData(data)
        #println(qldat)
        actor_training(qldat, actor)
    end
    return actor
end



function actor_training(qldat::QLData, actor::Tabular_actor)
    for event in qldat.data
        e=actor(event.state,event.action)
        g=e -event.E_critic
        actor.update(actor,event.state,event.action,g)
    end

end


function actor_training(qldat::QLData, actor::Deep_actor)
    #loss(x, y) = Flux.Losses.mse(actor.model(x), y)
    loss(x, y) = actor.lossfunction(actor.model(x), y)
    ps=actor.trainparameters
    X=[append!(deepcopy(event.state),event.action) for event in qldat.data]
    Y=[[event.E_critic] for event in qldat.data]
    X=reduce(hcat,X)
    Y=reduce(hcat,Y)[:,:]

    data=DataLoader((X, Y), batchsize=32,shuffle=true)  ## need to figure out how to make this a batch_size thing

    for d in data
        train_loss, back = Zygote.pullback(() -> loss(d...), ps)
        gs = back(one(train_loss))
        actor.update(actor,gs)
    end


end
