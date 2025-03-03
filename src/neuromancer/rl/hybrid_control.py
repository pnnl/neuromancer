import time
from neuromancer.psl.gym import BuildingEnv
from neuromancer.rl.gym_nssm import NSSMTrainer
from neuromancer.rl.gym_dpc import DPCTrainer  # Import the DPCTrainer class
from neuromancer.rl.ppo import Agent, Args

# Define the physical system model using ODEs
env = BuildingEnv(simulator='SimpleSingleZone', seed=1, backend='torch')

# Train the Neural State Space Model (NSSM)
# nssm_trainer = NSSMTrainer(env, batch_size=100, epochs=10)
# nssm_trainer.train(nsim=2000, nsteps=2)

# Pre-train the policy network using DPC
dpc_trainer = DPCTrainer(env, batch_size=100, epochs=10)
dpc_trainer.train(nsim=100, nsteps=100, nsamples=100)

DPC_PRETRAINING = False

# Train the policy network using DRL
args = Args(
    env_id='SimpleSingleZone',
    seed=1,
    total_timesteps=1000000,
    learning_rate=3e-4,
    num_envs=1,
    num_steps=2048,
    anneal_lr=True,
    gamma=0.99,
    gae_lambda=0.95,
    num_minibatches=32,
    update_epochs=10,
    norm_adv=True,
    clip_coef=0.2,
    clip_vloss=True,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    target_kl=None
)

args.batch_size = int(args.num_envs * args.num_steps)
args.minibatch_size = int(args.batch_size // args.num_minibatches)
args.num_iterations = args.total_timesteps // args.batch_size
run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

if args.track:
    import wandb

    wandb.init(
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        sync_tensorboard=True,
        config=vars(args),
        name=run_name,
        monitor_gym=True,
        save_code=True,
    )

if DPC_PRETRAINING:
    # load the policy model pre-trained by DPC
    agent = Agent(args, actor=dpc_trainer.policy)
else:
    agent = Agent(args)
    
agent.train()
agent.evaluate_and_save()