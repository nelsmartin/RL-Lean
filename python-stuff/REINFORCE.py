






def get_moves(oracle, state):
    pass

















"""
initialize policy_network θ
initialize optimizer
initialize baseline = 0

for epoch in range(num_epochs):

    batch_trajectories = []

    # --- Collect batch ---
    for _ in range(batch_size):

        theorem = sample(dataset)

        state = env.reset(theorem)

        log_probs = []
        rewards = []

        done = False

        while not done:
            moves = get_moves(state)

            probs = policy_network(state, moves)
            dist = Categorical(probs)

            action_idx = dist.sample()
            action = moves[action_idx]

            log_prob = dist.log_prob(action_idx)

            next_state, reward, done = env.step(action)

            log_probs.append(log_prob)
            rewards.append(reward)

            state = next_state

        # compute returns
        returns = compute_returns(rewards)

        batch_trajectories.append((log_probs, returns))


    # --- Update baseline (across batch) ---
    all_returns = [traj_returns[0] for (_, traj_returns) in batch_trajectories]
    batch_mean_return = mean(all_returns)

    baseline = 0.9 * baseline + 0.1 * batch_mean_return


    # --- Compute loss over batch ---
    loss = 0

    for (log_probs, returns) in batch_trajectories:
        for t in range(len(log_probs)):
            advantage = returns[t] - baseline
            loss += -log_probs[t] * advantage


    loss = loss / batch_size   # normalize


    # --- Backprop ---
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



"""