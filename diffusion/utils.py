# Utilities for diffusion.
from typing import Optional, List, Union

# import d4rl
import gin
import gym
import numpy as np
import torch
from torch import nn

# GIN-required Imports.
from synther.diffusion.denoiser_network import ResidualMLPDenoiser
from synther.diffusion.elucidated_diffusion import ElucidatedDiffusion
from synther.diffusion.norm import normalizer_factory

import minari

# Make transition dataset from data.
@gin.configurable
def make_inputs(
        env: gym.Env,
        modelled_terminals: bool = False,
) -> np.ndarray:
    dataset = d4rl.qlearning_dataset(env)
    obs = dataset['observations']
    actions = dataset['actions']
    next_obs = dataset['next_observations']
    rewards = dataset['rewards']
    inputs = np.concatenate([obs, actions, rewards[:, None], next_obs], axis=1)
    if modelled_terminals:
        terminals = dataset['terminals'].astype(np.float32)
        inputs = np.concatenate([inputs, terminals[:, None]], axis=1)
    return inputs

def make_inputs(
        dataset_name: str,
        modelled_terminals: bool = False,
        original: bool = False,
        context: bool = False,
) -> np.ndarray:
    dataset = minari.load_dataset(dataset_name, download=False)
    obs, actions, rewards, next_obs, contexts, episode_lens, dones = [], [], [], [], [], [], []

    # ********************************************************************************************************************
    # top_episodes = []
    # num_ep_per_split = 500
    # n = dataset.total_episodes // num_ep_per_split
    # for i in range(n):
    #     indices = np.arange(0, num_ep_per_split) + i * num_ep_per_split
    #     episodes_generator = dataset.iterate_episodes(episode_indices=indices)
    #     episodes_with_rewards = []
    #     for episode in episodes_generator:
    #         total_reward = sum(episode.rewards)
    #         episodes_with_rewards.append((episode, total_reward))
    #     episodes_with_rewards.sort(key=lambda x: x[1], reverse=True)
    #     top_episodes_with_rewards = episodes_with_rewards[:150]
    #     top_episodes += [episode for episode, _ in top_episodes_with_rewards]
    # for episode in top_episodes:
    #     obs.append(episode.observations[:-1])
    #     actions.append(episode.actions)
    #     rewards.append(episode.rewards)
    #     next_obs.append(episode.observations[1:])
    #     dones.append(episode.terminations)
    #     contexts.append(episode.infos['length'][:-1])
    # ********************************************************************************************************************
    episodes_generator = dataset.iterate_episodes(episode_indices=dataset.episode_indices)
    for i, episode in enumerate(episodes_generator):
        if i % 500 >= 250:
            obs.append(episode.observations[:-1])
            actions.append(episode.actions)
            rewards.append(episode.rewards)
            next_obs.append(episode.observations[1:])
            dones.append(episode.terminations)
            contexts.append(episode.infos['length'][:-1])
    # ********************************************************************************************************************
    
    obs = np.concatenate(obs).astype(np.float32)
    actions = np.concatenate(actions).astype(np.float32)
    rewards = np.concatenate(rewards).astype(np.float32)
    next_obs = np.concatenate(next_obs).astype(np.float32)
    terminals = np.concatenate(dones).astype(np.float32)
    contexts = np.concatenate(contexts).astype(np.float32)[:, None]
    # print(obs.shape, actions.shape, rewards.shape, next_obs.shape)
    if original:
        return {'observations': obs, 'actions': actions, 'rewards': rewards, 'next_observations': next_obs, 'terminals': terminals, 'contexts': contexts}   
    else:
        inputs = np.concatenate([obs, actions, rewards[:, None], next_obs], axis=1)
        if modelled_terminals:
            inputs = np.concatenate([inputs, terminals[:, None]], axis=1)
        if context:
            return inputs, contexts
        else:
            return inputs


# Convert diffusion samples back to (s, a, r, s') format.
@gin.configurable
def split_diffusion_samples(
        samples: Union[np.ndarray, torch.Tensor],
        env: gym.Env,
        modelled_terminals: bool = False,
        terminal_threshold: Optional[float] = None,
):
    # Compute dimensions from env
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    # Split samples into (s, a, r, s') format
    obs = samples[:, :obs_dim]
    actions = samples[:, obs_dim:obs_dim + action_dim]
    rewards = samples[:, obs_dim + action_dim]
    next_obs = samples[:, obs_dim + action_dim + 1: obs_dim + action_dim + 1 + obs_dim]
    if modelled_terminals:
        terminals = samples[:, -1]
        if terminal_threshold is not None:
            if isinstance(terminals, torch.Tensor):
                terminals = (terminals > terminal_threshold).float()
            else:
                terminals = (terminals > terminal_threshold).astype(np.float32)
        return obs, actions, rewards, next_obs, terminals
    else:
        return obs, actions, rewards, next_obs


@gin.configurable
def construct_diffusion_model(
        inputs: Union[torch.Tensor, list],
        normalizer_type: str,
        denoising_network: nn.Module,
        disable_terminal_norm: bool = False,
        skip_dims: List[int] = [],
        cond_dim: Optional[int] = None,
) -> ElucidatedDiffusion:
    if type(inputs) == list:
        inputs = inputs[0]
    event_dim = inputs.shape[1]
    model = denoising_network(d_in=event_dim, cond_dim=cond_dim)

    if disable_terminal_norm:
        terminal_dim = event_dim - 1
        if terminal_dim not in skip_dims:
            skip_dims.append(terminal_dim)

    if skip_dims:
        print(f"Skipping normalization for dimensions {skip_dims}.")

    normalizer = normalizer_factory(normalizer_type, inputs, skip_dims=skip_dims)

    return ElucidatedDiffusion(
        net=model,
        normalizer=normalizer,
        event_shape=[event_dim],
    )
