from parco.envs.hcvrp.generator import HCVRPGenerator


class CVRPGenerator(HCVRPGenerator):
    """Data generator for the (homogeneous-fleet) Capacitated Vehicle Routing
    Problem (CVRP) -- SRP Idea 4.

    Identical to HCVRPGenerator except the fleet is homogeneous: every
    vehicle in a given instance gets the SAME capacity (cap_m = cap for all
    m) and the same speed (fixed at 1.0, i.e. distance == travel time, as in
    the classical CVRP -- no per-vehicle speed heterogeneity at all). This is
    the only thing that changes; HCVRPEnv's state/mask/reward logic already
    treats capacity/speed as generic per-agent tensors and doesn't assume
    they differ across agents, so the env, decoder, and conflict handler are
    reused completely unchanged (see parco.envs.cvrp.env.CVRPEnv).
    """

    def _generate(self, batch_size):
        td = super()._generate(batch_size)

        # Homogeneous fleet: sample ONE capacity per instance and broadcast
        # it to every agent, instead of HCVRPGenerator's per-agent capacity.
        homogeneous_capacity = td["capacity"][..., :1].expand(*batch_size, self.num_agents)
        td.set("capacity", homogeneous_capacity.contiguous())

        # No speed heterogeneity either: every agent moves at the same
        # speed, so total time is purely a function of distance traveled.
        td.set("speed", td["speed"].new_ones(*batch_size, self.num_agents))

        return td
