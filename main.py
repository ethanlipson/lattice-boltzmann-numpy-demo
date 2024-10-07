import numpy as np
from matplotlib import pyplot as plt
import datetime
import os

N = (400, 400)
tau = 0.65

f = np.empty((N[1], N[0], 9))
idxs = np.array(range(9))
cxs = np.array([0, 1, 1, 0, -1, -1, -1, 0, 1])
cys = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1])
ws = np.array([4 / 9, 1 / 9, 1 / 36, 1 / 9, 1 / 36, 1 / 9, 1 / 36, 1 / 9, 1 / 36])


def get_circle_perm(pos: tuple[float, float], r: float, eps: float):
    X, Y = np.meshgrid(range(N[0]), range(N[1]))
    dist = np.sqrt((X - pos[0]) ** 2 + (Y - pos[1]) ** 2)
    return np.where(
        dist <= r - eps / 2,
        0,
        np.where(dist <= r + eps / 2, (dist - (r - eps / 2)) / eps, 1),
    )


def get_joukowsky_perm(
    pos: tuple[float, float],
    scale: float,
    chi: float,
    eta: float,
    theta: float,
    smoothing: float,
):
    c = chi + eta * 1j
    X, Y = np.meshgrid(
        np.linspace(-N[0] / 2 / scale - pos[0], N[0] / 2 / scale - pos[0], N[0]),
        np.linspace(-N[1] / 2 / scale - pos[1], N[1] / 2 / scale - pos[1], N[1]),
    )
    Z = X + Y * 1j
    Z *= np.exp(theta * 1j)

    rad = np.sqrt(Z**2 - np.ones(Z.shape) * 4)
    inv1 = 0.5 * (Z + rad)
    inv2 = 0.5 * (Z - rad)
    sdf = np.maximum(np.abs(inv1 - c), np.abs(inv2 - c))

    r = np.abs(c - 1)
    perm = np.where(
        sdf <= r / smoothing,
        0,
        np.where(
            sdf <= r * smoothing,
            (sdf - r / smoothing) / (r * smoothing - r / smoothing),
            1,
        ),
    )

    return perm


cyl_pos = (N[0] // 4, N[1] // 2)
cyl_r = 20
cyl_eps = 3

joukowsky_chi = -0.1
joukowsky_eta = 0.2
joukowsky_theta = 0.5
joukowsky_pos = (0, 0)
joukowsky_scale = 30
joukowsky_smoothing = 1.03

# perm = get_circle_perm(cyl_pos, cyl_r, cyl_eps)
perm = get_joukowsky_perm(
    joukowsky_pos,
    joukowsky_scale,
    joukowsky_chi,
    joukowsky_eta,
    joukowsky_theta,
    joukowsky_smoothing,
)

f.fill(1)
f[:, :, 1].fill(4)
rho = np.sum(f, axis=2)
f /= rho[:, :, np.newaxis]

t = 0
frame = 0
datestr = datetime.datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
os.makedirs(f"frames/{datestr}", exist_ok=True)
print(datestr)

while True:
    # Streaming
    for i, cx, cy in zip(idxs, cxs, cys):
        f[:, :, i] = np.roll(f[:, :, i], cx, axis=1)
        f[:, :, i] = np.roll(f[:, :, i], cy, axis=0)

    # Collision
    rho = np.sum(f, axis=2)
    mx = np.sum(f * cxs, axis=2)
    my = np.sum(f * cys, axis=2)

    Fx = np.sum(mx - mx * perm, axis=(0, 1))
    Fy = np.sum(my - my * perm, axis=(0, 1))
    if t % 100 == 0:
        print(Fx, Fy)

    ux = mx / rho * perm
    uy = my / rho * perm
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
    if t % 5 == 0:
        plt.cla()
        vorticity = (np.roll(uy, 1, axis=1) - np.roll(uy, -1, axis=1)) / 2 - (
            np.roll(ux, 1, axis=0) - np.roll(ux, -1, axis=0)
        ) / 2
        plt.imshow(vorticity, cmap="bwr", origin="lower")
        plt.clim(-0.05, 0.05)
        plt.savefig(f"frames/{datestr}/{frame:04d}.png")
        frame += 1
