import matplotlib.pyplot as plt
from firedrake.pyplot import triplot, tripcolor, tricontour
import os
import numpy as np

# 对比优化前后 mesh 的形状变化（即形状优化的结果）
# def plot_mesh(mesh_current, mesh_initial, title=None):
#   fig, ax = plt.subplots(ncols=1, figsize=(6, 3))
#   triplot(mesh_initial, axes=ax, interior_kw={"alpha":0}, boundary_kw={"color": "black", "linewidth": .5, "linestyle": "--"})
#   triplot(mesh_current, axes=ax, interior_kw={"alpha":0}, boundary_kw={"color": "red", "linewidth": 1})
#   ax.set_xlim(-.1, .1)
#   ax.set_ylim(-.1, .1)
#   ax.axis("off")
#   ax.set_aspect("equal")
#   if title is not None:
#     ax.set_title(str(title))
#   plt.savefig("mesh.png", dpi=300, bbox_inches="tight")

# # 绘制张量场（如最大主应力分布 tensile_limit）的空间分布图
# def plot_scalar_field(scalar_field, title=None):
#   fig, ax = plt.subplots(ncols=1, figsize=(6, 3))
#   contours = tripcolor(scalar_field, axes=ax)
#   limits = (scalar_field.dat.data.min(), scalar_field.dat.data.max())
#   contours.set_clim(limits[0], limits[1])
#   cbar = plt.colorbar(contours, orientation="vertical", label=scalar_field.name())
#   contours.set_clim(limits[0], limits[1])
#   ax.set_xlim(-.1, .1)
#   ax.set_ylim(-.1, .1)
#   ax.axis("off")
#   ax.set_aspect("equal")
#   if title is not None:
#     ax.set_title(str(title))
#   plt.savefig("scalar_field.png", dpi=300, bbox_inches="tight")

def plot_mesh(mesh_current, mesh_initial, title=None, outdir=None, suffix=None):
    fig, ax = plt.subplots(ncols=1, figsize=(6, 3))
    triplot(mesh_initial, axes=ax, interior_kw={"alpha": 0}, boundary_kw={"color": "black", "linewidth": 0.5, "linestyle": "--"})
    triplot(mesh_current, axes=ax, interior_kw={"alpha": 0}, boundary_kw={"color": "red", "linewidth": 1})
    ax.set_xlim(-0.1, 0.1)
    ax.set_ylim(-0.1, 0.1)
    ax.axis("off")
    ax.set_aspect("equal")

    if title:
        ax.set_title(str(title))

    # add the output part
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        suffix_str = f"_{suffix}" if suffix else ""
        filename = os.path.join(outdir, f"mesh{suffix_str}.png")
    else:
        filename = "mesh.png"

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_scalar_field(scalar_field, title=None, outdir=None, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(4, 4))
    contours = tripcolor(scalar_field, axes=ax, cmap="plasma", shading="gouraud")

    if vmin is not None and vmax is not None:
        contours.set_clim(vmin, vmax)
    plt.colorbar(contours, ax=ax, orientation="vertical", label=scalar_field.name())

    levels = np.linspace(vmin, vmax, 10)
    tricontour(scalar_field, axes=ax, levels=levels, colors="black", linewidths=0.3)

    # ✅ 自动设置可视范围（根据 mesh）
    # coords = scalar_field.function_space().mesh().coordinates.dat.data_ro
    # x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    # y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    # x_pad = (x_max - x_min) * 0.1
    # y_pad = (y_max - y_min) * 0.1
    # ax.set_xlim(x_min - x_pad, x_max + x_pad)
    # ax.set_ylim(y_min - y_pad, y_max + y_pad)

    ax.set_aspect("equal")
    ax.axis("off")

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        basename = scalar_field.name()
        filename = os.path.join(outdir, f"{basename}_{int(title)}.png")
    else:
        filename = f"{scalar_field.name()}_{int(title)}.png"

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()