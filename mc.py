#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 16:11:22 2019

@author: huiminren
# Modified By Yanhua Li on 08/19/2023 for gymnasium==0.29.0
"""
import numpy as np
import random
from collections import defaultdict
#-------------------------------------------------------------------------
'''
    Monte-Carlo
    In this problem, you will implememnt an AI player for Blackjack.
    The main goal of this problem is to get familar with Monte-Carlo algorithm.

    You could test the correctness of your code
    by typing 'nosetests -v mc_test.py' in the terminal.
'''
#-------------------------------------------------------------------------


def initial_policy(observation):
    """A policy that sticks if the player score is >= 20 and hit otherwise

    Parameters:
    -----------
    observation:
    Returns:
    --------
    action: 0 or 1
        0: STICK
        1: HIT
    """
    if observation[0] >= 20:
        action = 0
    else:
        action = 1
    return action


def mc_prediction(policy, env, n_episodes, gamma=1.0):
    """Given policy using sampling to calculate the value function
        by using Monte Carlo first visit algorithm.

    Parameters:
    -----------
    policy: function
        A function that maps an obversation to action probabilities
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor
    Returns:
    --------
    V: defaultdict(float)
        A dictionary that maps from state to value
    """
    # initialize empty dictionaries
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    # a nested dictionary that maps state -> value
    V = defaultdict(float)

    for _ in range(n_episodes):
        state = env.reset()[0]
        # print(f"State: {state}")
        episode = []

        # print("Starting the loop...")
        while True:
            action = policy(state)
            # print("State in loop: ", state)
            # print("Action: ", action)
            next_state, reward, done, info, _ = env.step(action)
            # print(f"Next state: {next_state}, reward: {reward}, done: {done}")
            episode.append((state, action, reward))
            state = next_state
            if done == True:
                break
        # print("Episode: ", episode)

        G = 0
        visited_states = set()
        for i in range(len(episode)-1, -1, -1):
            # if not episode:
            #     break
            state, action, reward = episode[i]
            if state not in visited_states:
                returns_count[state] += 1
                G = reward + gamma * G 
                returns_sum[state] += G
                visited_states.add(state)
            
                V[state] = returns_sum[state] / returns_count[state]
    
    print("V: ", V)
    return V
            
def epsilon_greedy(Q, state, nA, epsilon=0.1):
    """Selects epsilon-greedy action for supplied state.

    Parameters:
    -----------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a.
    state: 
        current state
    nA: int
        Number of actions in the environment
    epsilon: float
        The probability to select a random action, range between 0 and 1

    Returns:
    --------
    action: int
        action based current state
    Hints:
    ------
    With probability (1 - epsilon) choose the greedy action.
    With probability epsilon choose an action at random.
    """

    probs = np.ones(nA) * epsilon / nA
    best_action = np.argmax(Q[state])
    probs[best_action] += (1 - epsilon) 
    
    action = np.random.choice(nA, p = probs)

    return action

def mc_control_epsilon_greedy(env, n_episodes, gamma=1.0, epsilon=0.1):
    """Monte Carlo control with exploring starts.
        Find an optimal epsilon-greedy policy.

    Parameters:
    -----------
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor
    epsilon: float
        The probability to select a random action, range between 0 and 1
    Returns:
    --------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a.
    Hint:
    -----
    You could consider decaying epsilon, i.e. epsilon = epsilon-0.1/n_episode during each episode
    and episode must > 0.
    """
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    # a nested dictionary that maps state -> (action -> action-value)
    # e.g. Q[state] = np.darrary(nA)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    k = 0 # count episodes
    c = 1
    for _ in range(n_episodes):
        print("Episode: \n", k+1, flush=True)
        episode = []
        state = env.reset()[0]
        while True:
            # print("actions: ", env.action_space.n)
            action = epsilon_greedy(Q, state, env.action_space.n, epsilon)
            # print("action: ", action)
            next_state, reward, done, info, _ = env.step(action)

            episode.append((state, action, reward))
            print(f"State: {state}, Action: {action}, Next State: {next_state}, Reward: {reward}", flush=True)
            state = next_state 
            if done == True:
                break
            if reward > 0:
                status = "Win"
            else:
                status = "Loss"
            print(f"Status: {status}", flush=True)

        G = 0
        visited_states = set()
        for i in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[i]
            # print(f"State {state}, Action {action}, Reward {reward}")

            if (state, action) not in visited_states:
                returns_count[(state, action)] += 1
                # print("Returns count: ", returns_count)
                # print("G before: ", G)
                G = reward + gamma * G 
                # print("G after: ", G)
                # print("Returns sum before: ", returns_sum)
                returns_sum[(state, action)] += G
                # print("Returns sum after: ", returns_sum)

                visited_states.add((state, action))
                # print("Visited states: ", visited_states)
            
                Q[state][action] = returns_sum[(state, action)] / returns_count[(state, action)]
                # print(f"Q {state}, {action}: ", Q[state][action])
        k += 1
        epsilon = max(0.1, epsilon-0.1/n_episodes)
    #     if k == 100000 * c:
    #         print(f"Q: {Q} for episode: {k}")
    #         c += 1
    # print("Q: ", Q)
    return Q


