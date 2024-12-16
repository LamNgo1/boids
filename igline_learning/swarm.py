from copy import deepcopy

import numpy as np


class Swarm:
    def __init__(self, n_particles, **kwargs):
        self.n_particles = n_particles  # number of particles
        self.c1 = kwargs['c1']      # pso cognitive parameter
        self.c2 = kwargs['c2']      # pso social parameter
        self.w = kwargs['w']        # pso inertia parameter
        self.dimensions = kwargs['dimensions']
        self.positions = None       # x values of all particles, shape (n_particles, dimensions)
        self.cost = None            # fx values of all particles, shape (n_particles,)
        self.velocities = None      # displacement vectors of particles, shape (n_particles, dimensions)
        self.pso_vel = [[]]*self.n_particles # incumbent-guided direction, shape (n_particles, dimensions)
        self.gbest_pos = None       # position of the global incumbent, shape (dimensions,)
        self.pbest_pos = None       # position of the personal incumbent, shape (n_particles, dimensions)
        self.gbest = None           # value of the global incumbent, shape (1,)
        self.pbest = None           # value of the personal incumbent, shape (n_particles,)
        self.current_particle: int = 0   # current particle index
        self.history = {
            'position': [],
            'velocity': [],
            'pso_vel': [],
            'pbest': [],
            'gbest': [],
        }

    @property
    def curr_particle(self):
        return self.current_particle
    
    def get_next_particle(self):
        self.current_particle += 1
        if self.current_particle >= self.n_particles:
            self.current_particle = 0
            self.update_history(self.positions, self.velocities, self.pso_vel)
            self.pso_vel = self.pso_vel*0
        return self.current_particle
    
    @property
    def idx_best(self):
        if self.pbest is None:
            return None
        return int(np.argmin(self.pbest))
    
    def update_history(self, pos, vel, pso_vel=None):
        self.history['position'].append(deepcopy(pos))
        self.history['velocity'].append(deepcopy(vel))
        self.history['pso_vel'].append(np.array(pso_vel))
        self.history['pbest'].append((deepcopy(self.pbest_pos), deepcopy(self.pbest)))
        self.history['gbest'].append((deepcopy(self.gbest_pos), deepcopy(self.gbest)))


    
    def init_pos_vel(self, init_pos, init_cost, **kwargs):
        assert init_pos.ndim == 2
        assert init_cost.ndim == 1
        assert init_pos.shape[0] == len(init_cost) == self.n_particles
        assert init_pos.shape[1] == self.dimensions
        self.positions = deepcopy(init_pos)
        self.cost = deepcopy(init_cost)
        self.pbest_pos = deepcopy(init_pos)
        self.pbest = deepcopy(init_cost)
        for i in range(self.n_particles):   
            if self.gbest is None or init_cost[i] < self.gbest:
                self.gbest_pos = init_pos[i]
                self.gbest = init_cost[i]

        clamp = kwargs.get('clamp', None)
        min_velocity, max_velocity = (0, 1) if clamp is None else clamp
        self.velocities = np.zeros((self.n_particles, self.dimensions))
        self.velocities = (max_velocity - min_velocity) * np.random.random_sample(
            size=(self.n_particles, self.dimensions)
        ) + min_velocity
        self.pso_vel = np.zeros((self.n_particles, self.dimensions))
        self.update_history(self.positions, self.velocities, self.pso_vel)
        ...
    
    def update_pos(self, idx, new_pos, new_cost, new_vel=None):
        assert idx is not None
        if new_pos.ndim == 2:
            assert new_pos.shape[0] == 1
            new_pos = new_pos[0]
        if new_cost.ndim == 2:
            assert new_cost.shape[0] == 1
            new_cost = new_cost[0]
        assert len(new_pos) == self.dimensions
        if new_vel is None:
            self.velocities[idx] = new_pos - self.positions[idx]
        else:
            self.velocities[idx] = new_vel
        self.positions[idx] = new_pos
        self.cost[idx] = new_cost
        new_pbest = False
        if new_cost < self.pbest[idx] - 1e-3 * np.abs(self.pbest[idx]):
            new_pbest = True
        if new_cost < self.pbest[idx]:
            self.pbest_pos[idx] = new_pos
            self.pbest[idx] = new_cost
            if new_cost < self.gbest:
                self.gbest_pos = new_pos
                self.gbest = new_cost
                ...
        return new_pbest
    
    def set_velocities_to_zeros(self):
        self.velocities = np.zeros((self.n_particles, self.dimensions))
    
    def compute_pso_vel_batch(self):
        r1 = np.random.rand(self.n_particles, self.dimensions)
        r2 = np.random.rand(self.n_particles, self.dimensions)
        p = self.pbest_pos - self.positions
        g = self.gbest_pos - self.positions
        pso_vel = self.w * self.velocities + self.c1 * r1 * p + self.c2 * r2 * g
        norms = np.linalg.norm(pso_vel, axis=1)
        min_velocity, max_velocity = (0, 1)
        pso_vel[norms == 0] = (max_velocity - min_velocity) * np.random.random_sample(size=pso_vel[norms == 0].shape) + min_velocity
        self.pso_vel = pso_vel
        return pso_vel
