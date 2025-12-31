import matplotlib.pyplot as plt
# Latex rendering
plt.rc('text', usetex=True)

from checkpoint_schedules import Revolve, EndForward, Forward, MixedCheckpointSchedule, StorageType

total_steps = 1000
storage = [5, 10, 20, 40, 60, 80, 100]
fwd = []
# for s in storage:
#     schedule = MixedCheckpointSchedule(total_steps, s, storage=StorageType.RAM)
#     schedule = Revolve(total_steps, s)
#     fwd_steps = 0
#     reverse = False
#     for action in schedule:
#         if isinstance(action, EndForward):
#             reverse = True
#         if isinstance(action, Forward) and reverse:
#             fwd_steps += action.n1 - action.n0
#         # print(f"Action: {action}")

#     print(fwd_steps + total_steps)


revolve = [7284, 4636, 3747, 3097, 2938, 2918, 2898]
mixed = [6828, 3921, 2823, 2142, 1958, 1932, 1909]
storage = [5, 10, 20, 40, 60, 80, 100]
plt.figure(figsize=(8, 6))
plt.plot(storage, [fwd / total_steps for fwd in mixed], 's-', label=r"Mixed Schedule", lw=2)
plt.plot(storage, [fwd / total_steps for fwd in revolve],'o-', label=r"Revolve", lw=2)
# Plot a horizontal line for no checkpointing
# plt.plot(storage, [fwd / 4000 for fwd in no_checkpoint], label=r"No Checkpointing", linestyle="--", color="black", lw=2)
plt.xlabel("Number of time-steps checkpointed", fontsize=16)
plt.ylabel("Shedules time-steps / No Checkpointing time-steps", fontsize=16)
plt.legend(fontsize=16, loc="center right")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
# plt.ylim(1.8, 3.2)
# Increase the font size
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

