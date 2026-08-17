from parco.envs.hcvrp.env import HCVRPEnv

from .generator import CVRPGenerator


class CVRPEnv(HCVRPEnv):
    """(Homogeneous-fleet) Capacitated Vehicle Routing Problem (CVRP)
    environment -- SRP Idea 4.

    A thin subclass of HCVRPEnv: state transitions, action masking, and the
    min-max reward are all identical to HCVRP and are reused verbatim (they
    already operate on generic per-agent capacity/speed tensors without
    assuming heterogeneity). The only difference is the default generator,
    which samples a homogeneous fleet (cap_m = cap for all m, unit speed)
    instead of HCVRPGenerator's heterogeneous one. No changes are needed
    anywhere else in the model (decoder, agent conflict handler,
    communication layers) -- this env plugs into PARCOPolicy exactly like
    HCVRPEnv does, just registered under a different env_name ("cvrp") so
    its own init/context embeddings and configs can be selected.

    Args:
        generator: An instance of CVRPGenerator used as the data generator.
        generator_params: Parameters configuring the generator (num_loc,
            num_agents, capacity range, etc.) -- same as HCVRPGenerator's,
            minus the heterogeneity-specific ones (min/max_speed no longer
            matter since speed is fixed at 1.0).
    """

    name = "cvrp"

    def __init__(
        self,
        generator: CVRPGenerator = None,
        generator_params: dict = {},
        check_solution: bool = False,
        **kwargs,
    ):
        if generator is None:
            generator = CVRPGenerator(**generator_params)
        super().__init__(
            generator=generator, check_solution=check_solution, **kwargs
        )
