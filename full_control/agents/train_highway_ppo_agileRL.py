# ─────────────────────────────────────────────────────────────────────────────
# agents/train_highway_ppo_agileRL.py
# AgileRL evolutionary HPO for PPO on HighwayEnv (continuous actions)
# AgileRL runs a POPULATION of PPO agents simultaneously, evolves them
# using tournament selection and mutation to find optimal hyperparameters
# automatically — no manual tuning needed
# ─────────────────────────────────────────────────────────────────────────────

import sys
import os

# Required to avoid multiprocessing errors with vectorized envs on Windows
if __name__ == "__main__":

    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

    import numpy as np
    import torch
    import matplotlib.pyplot as plt

    from agilerl.algorithms import PPO
    from agilerl.algorithms.core.registry import HyperparameterConfig, RLParameter
    from agilerl.hpo.mutation import Mutations
    from agilerl.hpo.tournament import TournamentSelection
    from agilerl.training.train_on_policy import train_on_policy
    from agilerl.utils.utils import create_population, make_vect_envs

    from full_control.envs.highway_config_v1 import HIGHWAY_CONFIG

    # ── Device ────────────────────────────────────────────────────────────────
    # Uses your GTX 1650 Ti GPU — multiple agents benefit from GPU parallelism
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ── Environment setup ─────────────────────────────────────────────────────
    # make_vect_envs creates multiple parallel environment instances
    # num_envs=4 means 4 environments run simultaneously to collect experience faster
    num_envs = 4
    env = make_vect_envs(
        "highway-v0",
        num_envs=num_envs,
        env_kwargs={"config": HIGHWAY_CONFIG}
    )

    # Extract observation and action spaces from the vectorized environment
    observation_space = env.single_observation_space
    action_space      = env.single_action_space
    print(f"Observation space: {observation_space}")
    print(f"Action space:      {action_space}")

    # ── Initial hyperparameters ───────────────────────────────────────────────
    # These are the STARTING values for the population
    # AgileRL will evolve these during training to find better combinations
    INIT_HP = {
        "POP_SIZE"      : 4,        # 4 agents in population — evolve simultaneously
        "BATCH_SIZE"    : 128,      # minibatch size for gradient computation
        "LR"            : 3e-4,     # α — starting learning rate
        "LEARN_STEP"    : 1024,     # steps collected before each policy update
        "GAMMA"         : 0.99,     # γ — discount factor
        "GAE_LAMBDA"    : 0.95,     # λ — GAE smoothing factor
        "ACTION_STD_INIT": 0.6,     # initial std of action distribution
        "CLIP_COEF"     : 0.2,      # ε — PPO clipping parameter
        "ENT_COEF"      : 0.01,     # entropy bonus coefficient
        "VF_COEF"       : 0.5,      # value function loss coefficient
        "MAX_GRAD_NORM" : 0.5,      # gradient clipping threshold
        "TARGET_KL"     : None,     # KL divergence threshold (None = disabled)
        "UPDATE_EPOCHS" : 4,        # gradient passes per collected batch
        "TARGET_SCORE"  : 160.0,    # target mean reward to consider solved
        "MAX_STEPS"     : 200_000,  # total training steps per agent
        "EVO_STEPS"     : 10_000,   # steps between each evolution cycle
        "EVAL_STEPS"    : None,     # steps per evaluation episode (None = full)
        "EVAL_LOOP"     : 3,        # evaluation episodes per agent per cycle
        "TOURN_SIZE"    : 2,        # agents compared per tournament
        "ELITISM"       : True,     # always preserve best agent
    }

    # ── Mutation parameters ───────────────────────────────────────────────────
    # Controls what types of mutations happen and how often
    MUT_P = {
        "NO_MUT"    : 0.4,   # 40% chance of no mutation (stability)
        "ARCH_MUT"  : 0.2,   # 20% chance of network architecture change
        "NEW_LAYER" : 0.2,   # 20% chance of adding a new layer
        "PARAMS_MUT": 0.2,   # 20% chance of weight perturbation
        "ACT_MUT"   : 0.2,   # 20% chance of activation function change
        "RL_HP_MUT" : 0.2,   # 20% chance of RL hyperparameter mutation
        "MUT_SD"    : 0.1,   # mutation strength (standard deviation)
        "RAND_SEED" : 42,    # random seed for reproducibility
    }

    # ── Hyperparameter search space ───────────────────────────────────────────
    # Defines the bounds AgileRL searches within during RL_HP_MUT mutations
    # These are the hyperparameters we want AgileRL to optimize automatically
    hp_config = HyperparameterConfig(
        lr         = RLParameter(min=1e-4, max=1e-2),   # learning rate range
        batch_size = RLParameter(min=32,   max=512),     # batch size range
    )

    # ── Neural network architecture ───────────────────────────────────────────
    # Two hidden layers of 64 neurons — same as our SB3 MlpPolicy
    # AgileRL can also mutate this architecture during training (ARCH_MUT)
    net_config = {
        "head_config": {
            "hidden_size": [64, 64]
        }
    }

    # ── Create population of PPO agents ──────────────────────────────────────
    # Each agent starts with INIT_HP values but gets slightly varied
    # hyperparameters — this is what makes it a population, not just one agent
    pop = create_population(
        algo             = "PPO",
        observation_space= observation_space,
        action_space     = action_space,
        net_config       = net_config,
        INIT_HP          = INIT_HP,
        hp_config        = hp_config,
        population_size  = INIT_HP["POP_SIZE"],
        num_envs         = num_envs,
        device           = device,
    )
    print(f"\nPopulation of {len(pop)} PPO agents created")
    print("Starting evolutionary HPO training...\n")

    # ── Tournament selection ──────────────────────────────────────────────────
    # Selects which agents survive to the next generation
    # TOURN_SIZE=2: randomly pick 2 agents, the better one survives
    # ELITISM=True: best agent always survives regardless of tournament
    tournament = TournamentSelection(
        INIT_HP["TOURN_SIZE"],
        INIT_HP["ELITISM"],
        INIT_HP["POP_SIZE"],
        INIT_HP["EVAL_LOOP"],
    )

    # ── Mutations ─────────────────────────────────────────────────────────────
    # Applies random mutations to the population after each evolution cycle
    # Explores new hyperparameter combinations and network architectures
    mutations = Mutations(
        no_mutation  = MUT_P["NO_MUT"],
        architecture = MUT_P["ARCH_MUT"],
        new_layer_prob= MUT_P["NEW_LAYER"],
        parameters   = MUT_P["PARAMS_MUT"],
        activation   = MUT_P["ACT_MUT"],
        rl_hp        = MUT_P["RL_HP_MUT"],
        mutation_sd  = MUT_P["MUT_SD"],
        rand_seed    = MUT_P["RAND_SEED"],
        device       = device,
    )

    # ── Save path ─────────────────────────────────────────────────────────────
    os.makedirs("experiments", exist_ok=True)
    save_path = "experiments/ppo_highway_agilerl_best.pt"

    # ── Train with evolutionary HPO ───────────────────────────────────────────
    # train_on_policy orchestrates the entire process:
    # 1. Each agent collects LEARN_STEP steps of experience
    # 2. Each agent updates its policy (PPO gradient steps)
    # 3. Every EVO_STEPS: evaluate all agents, run tournament, mutate
    # 4. Repeat until MAX_STEPS reached
    # Returns the trained population and their fitness scores
    trained_pop, pop_fitnesses = train_on_policy(
        env        = env,
        env_name   = "highway-v0",
        algo       = "PPO",
        pop        = pop,
        INIT_HP    = INIT_HP,
        MUT_P      = MUT_P,
        max_steps  = INIT_HP["MAX_STEPS"],
        evo_steps  = INIT_HP["EVO_STEPS"],
        eval_steps = INIT_HP["EVAL_STEPS"],
        eval_loop  = INIT_HP["EVAL_LOOP"],
        tournament = tournament,
        mutation   = mutations,
        wb         = False,       # set True to log to Weights & Biases
        save_elite = True,        # save the best agent automatically
        elite_path = save_path,
    )

    env.close()

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("AGILERL HPO TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best agent saved to: {save_path}")
    print(f"\nFinal population fitnesses:")
    for i, (agent, fitness) in enumerate(zip(trained_pop, pop_fitnesses)):
        print(f"  Agent {i+1}: fitness = {fitness:.3f} | "
              f"lr = {agent.lr:.2e} | "
              f"batch_size = {agent.batch_size}")

    best_fitness = max(pop_fitnesses)
    print(f"\nBest fitness achieved: {best_fitness:.3f}")
    print(f"Baseline PPO fitness:  162.72")
    print(f"Improvement:           {best_fitness - 162.72:+.3f}")

    # ── Plot population fitness over generations ───────────────────────────────
    plt.figure(figsize=(10, 5))
    for i, agent in enumerate(trained_pop):
        if len(agent.fitness) > 0:
            plt.plot(agent.fitness, alpha=0.6, label=f"Agent {i+1}")
    plt.xlabel("Evolution cycle")
    plt.ylabel("Fitness (mean evaluation reward)")
    plt.title("AgileRL PPO — Population fitness over evolution cycles")
    plt.axhline(y=162.72, color='gray', linestyle='--',
                linewidth=1, label='Baseline PPO (162.72)')
    plt.legend()
    plt.tight_layout()
    plt.savefig("experiments/agilerl_fitness_curve.png", dpi=150)
    plt.show()
    print("Fitness curve saved to experiments/agilerl_fitness_curve.png")