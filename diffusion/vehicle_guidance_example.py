
# ------------------------------------------------------------------------------
# Example: How to use sample_physics for future vehicle types
# ------------------------------------------------------------------------------
# def bus_physics_constraint(x0):
#     """Example custom constraint for a BUS (Heavy Vehicle)."""
#     # 1. Power Limit: Max accel decreases as speed increases
#     # P = m * a * v => a_max = P_max / (m * v)
#     speed = (x0[:, 0, :] + 1) / 2 * 40.0
#     accel = x0[:, 1, :] * 5.0
#     
#     power_limit_accel = 200000 / (15000 * (speed + 1.0)) # Dummy values
#     accel_excess = F.relu(accel - power_limit_accel)
#     
#     # 2. Comfort: Strict jerk limit for passengers
#     jerk = torch.diff(accel, dim=1)
#     jerk_excess = F.relu(torch.abs(jerk) - 1.0)
#     
#     return torch.mean(accel_excess**2) + torch.mean(jerk_excess**2)
#
# To generate bus trajectories:
# samples = diffusion.sample_physics(model, 10, cond, physics_guide_fn=bus_physics_constraint)
# ------------------------------------------------------------------------------
