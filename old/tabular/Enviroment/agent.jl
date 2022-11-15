include("gridworld.jl")

mutable struct agent
    position::Int64
    old_position::Int64
    totalreward::Int64
end

function start_agent(grid::Gridworld)
    pos = start_random_actor(grid.grid, minimum(grid.grid))
    return agent(pos, pos, 0)
end

function move_actor(grid::Gridworld, actor::agent, action)
    actor.old_position = copy(actor.position)
    actor.position = actor.position + grid.actions[action]
    #finding the goal
    # hitting a wall
    if (grid.grid[actor.position] == 0)
        # println("reflection")
        # getting reflected
        actor.position = actor.position - grid.actions[action]#-grid.actions[action]
        # getting stuch between wall
        # if (grid.grid[actor]==0)
        #    println(" double reflection")
        #   actor.position=actor.position+grid.actions[action]
        #end
    end
    actor.totalreward += grid.grid[actor.position]
    return (grid.grid[actor.position] == maximum(grid.grid)), grid.grid[actor.position]
    # true says you are done   # true you found goal flase says you have to keep searching

end

function start_random_actor(grid, allowed_reward)
    actor = rand(2:length(grid))
    while (grid[actor] != allowed_reward)
        actor = rand(2:length(grid))
    end
    return actor
end
