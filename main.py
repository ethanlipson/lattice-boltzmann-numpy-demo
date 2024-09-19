import numpy as np
from matplotlib import pyplot as plt

N = (400, 100)
tau = 0.7
cyl_pos = (N[0] // 4, N[1] // 2)
cyl_r = 20

f = np.empty((N[1], N[0], 9))
idxs = np.array(range(9))
cxs = np.array([0, 1, 1, 0, -1, -1, -1, 0, 1])
cys = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1])
ws = np.array([4 / 9, 1 / 9, 1 / 36, 1 / 9, 1 / 36, 1 / 9, 1 / 36, 1 / 9, 1 / 36])

X, Y = np.meshgrid(range(N[0]), range(N[1]))
perm = (X - cyl_pos[0]) ** 2 + (Y - cyl_pos[1]) ** 2 > cyl_r**2

f.fill(1)
f[:, :, 1].fill(4)
rho = np.sum(f, axis=2)
f /= rho[:, :, np.newaxis]

t = 0
while True:
    # Streaming
    for i, cx, cy in zip(idxs, cxs, cys):
        f[:, :, i] = np.roll(f[:, :, i], cx, axis=1)
        f[:, :, i] = np.roll(f[:, :, i], cy, axis=0)

    # Collision
    rho = np.sum(f, axis=2)
    ux = np.sum(f * cxs, axis=2) / rho * perm
    uy = np.sum(f * cys, axis=2) / rho * perm
    ux[:, [0, -1]] = 0.2
    ux[[0, -1], :] = 0.2
    uy[:, [0, -1]] = 0
    uy[[0, -1], :] = 0

    feq = np.empty((N[1], N[0], 9))
    for i, cx, cy, w in zip(idxs, cxs, cys, ws):
        feq[:, :, i] = (
            rho
            * w
            * (
                1
                + 3 * (cx * ux + cy * uy)
                + 9 / 2 * (cx * ux + cy * uy) ** 2
                - 3 / 2 * (ux**2 + uy**2)
            )
        )

    f += (feq - f) / tau
    f[:, [0, -1], :] = feq[:, [0, -1], :]
    f[[0, -1], :, :] = feq[[0, -1], :, :]

    t += 1
    if t % 10 == 0:
        plt.cla()
        vorticity = (np.roll(uy, 1, axis=1) - np.roll(uy, -1, axis=1)) / 2 - (
            np.roll(ux, 1, axis=0) - np.roll(ux, -1, axis=0)
        ) / 2
        plt.imshow(vorticity, cmap="bwr")
        plt.clim(-0.05, 0.05)
        plt.pause(0.001)
