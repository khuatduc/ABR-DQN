# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 18:50:39 2021

@author: nguyen tuan
"""
import gym
env_dict = gym.envs.registration.registry.env_specs.copy()
for env in env_dict:
    if 'ABR-v0' in env:
        print("Remove {} from registry".format(env))
        del gym.envs.registration.registry.env_specs[env]