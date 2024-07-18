# Train diffusion model on D4RL transitions.
import argparse
import pathlib
import re

# import d4rl
import gin
import gymnasium as gym
import numpy as np
import torch
import wandb

from synther.diffusion.elucidated_diffusion import Trainer
from synther.diffusion.norm import MinMaxNormalizer
from synther.diffusion.utils import make_inputs, split_diffusion_samples, construct_diffusion_model

from dmc2gymnasium import DMCGym

@gin.configurable
class SimpleDiffusionGenerator:
    def __init__(
            self,
            env: gym.Env,
            ema_model,
            num_sample_steps: int = 128,
            sample_batch_size: int = 100000,
    ):
        self.env = env
        self.diffusion = ema_model
        self.diffusion.eval()
        # Clamp samples if normalizer is MinMaxNormalizer
        self.clamp_samples = isinstance(self.diffusion.normalizer, MinMaxNormalizer)
        self.num_sample_steps = num_sample_steps
        self.sample_batch_size = sample_batch_size
        print(f'Sampling using: {self.num_sample_steps} steps, {self.sample_batch_size} batch size.')

    def sample(
            self,
            num_samples: int,
            cond: torch.Tensor = None,
    ) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        assert num_samples % self.sample_batch_size == 0, 'num_samples must be a multiple of sample_batch_size'
        num_batches = num_samples // self.sample_batch_size
        observations = []
        actions = []
        rewards = []
        next_observations = []
        terminals = []
        for i in range(num_batches):
            print(f'Generating split {i + 1} of {num_batches}')
            sampled_outputs = self.diffusion.sample(
                batch_size=self.sample_batch_size,
                num_sample_steps=self.num_sample_steps,
                clamp=self.clamp_samples,
                cond=cond,
            )
            sampled_outputs = sampled_outputs.cpu().numpy()

            # Split samples into (s, a, r, s') format
            transitions = split_diffusion_samples(sampled_outputs, self.env)
            if len(transitions) == 4:
                obs, act, rew, next_obs = transitions
                terminal = np.zeros_like(next_obs[:, 0])
            else:
                obs, act, rew, next_obs, terminal = transitions
            observations.append(obs)
            actions.append(act)
            rewards.append(rew)
            next_observations.append(next_obs)
            terminals.append(terminal)
        observations = np.concatenate(observations, axis=0)
        actions = np.concatenate(actions, axis=0)
        rewards = np.concatenate(rewards, axis=0)
        next_observations = np.concatenate(next_observations, axis=0)
        terminals = np.concatenate(terminals, axis=0)

        return observations, actions, rewards, next_observations, terminals


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="dm-cartpole-test-length0.025-0.35-v0")
    parser.add_argument('--gin_config_files', nargs='*', type=str, default=['../config/resmlp_denoiser.gin'])
    parser.add_argument('--gin_params', nargs='*', type=str, default=[])
    # wandb config
    parser.add_argument('--wandb-project', type=str, default="Diffusion")
    parser.add_argument('--wandb-entity', type=str, default="")
    parser.add_argument('--wandb-group', type=str, default="Dataset")
    #
    parser.add_argument('--results_folder', type=str, default='./results')
    parser.add_argument('--use_gpu', action='store_true', default=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_samples', type=int, default=int(0))
    parser.add_argument('--save_num_samples', type=int, default=int(5e6))
    parser.add_argument('--save_file_name', type=str, default='5m_samples.npz')
    parser.add_argument('--load_checkpoint', type=int, default=int(0))
    parser.add_argument('--minari', type=int, default=int(1))
    parser.add_argument('--cond', type=float, nargs='+', default=None)
    args = parser.parse_args()
    gin.parse_config_files_and_bindings(args.gin_config_files, args.gin_params)

    match = re.search(r'length(.+?)-v0', args.dataset)
    if match:
        result = match.group(1)
        args.wandb_group = f'Dataset{result}'
    # Set seed.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.use_gpu:
        torch.cuda.manual_seed(args.seed)

    # Create the environment and dataset.
    if args.minari:
        env = DMCGym("cartpole", "swingup", task_kwargs={'random':args.seed})
        env = gym.wrappers.RecordEpisodeStatistics(env)
        inputs = make_inputs(args.dataset, context=True)
        inputs = torch.from_numpy(inputs[0]).float(), torch.from_numpy(inputs[1]).float()
        # print(inputs[0].shape, inputs[1].shape)
        dataset = torch.utils.data.TensorDataset(*inputs)
        print("save_samples: ", args.save_samples)
        print("load_checkpoint: ", args.load_checkpoint)
    else:
        env = gym.make(args.dataset)
        inputs = make_inputs(env)
        inputs = torch.from_numpy(inputs).float()
        dataset = torch.utils.data.TensorDataset(inputs)

    results_folder = pathlib.Path(args.results_folder)
    results_folder.mkdir(parents=True, exist_ok=True)
    with open(results_folder / 'config.gin', 'w') as f:
        f.write(gin.config_str())

    # Create the diffusion model and trainer.
    diffusion = construct_diffusion_model(inputs=inputs[0] if args.minari else inputs, cond_dim=len(args.cond) if args.cond is not None else None)
    trainer = Trainer(
        diffusion,
        dataset,
        results_folder=args.results_folder,
        # train_num_steps=10,
    )

    if not args.load_checkpoint:
        # Initialize logging.
        wandb.init(
            project=args.wandb_project,
            # entity=args.wandb_entity,
            config=args,
            group=args.wandb_group,
            name=args.results_folder.split('/')[-1],
        )
        # Train model.
        trainer.train()
    else:
        trainer.ema.to(trainer.accelerator.device)
        # Load the last checkpoint.
        trainer.load(milestone=trainer.train_num_steps)

    # Generate samples and save them.
    if args.save_samples:
        generator = SimpleDiffusionGenerator(
            env=env,
            ema_model=trainer.ema.ema_model,
        )
        observations, actions, rewards, next_observations, terminals = generator.sample(
            num_samples=args.save_num_samples,
            cond=torch.tensor(args.cond, dtype=torch.float32)[:, None] if args.cond is not None else None,
        )
        np.savez_compressed(
            results_folder / (args.save_file_name + "_" + "_".join(map(str, args.cond)) if args.cond else ""),
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            terminals=terminals,
        )
