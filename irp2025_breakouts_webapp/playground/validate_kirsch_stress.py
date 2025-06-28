from firedrake import *
import matplotlib.pyplot as plt
import numpy as np
import os

# set the material parameters
domainpath = "meshes/domain.msh"
sigma_H = 10e6  # 10 MPa
youngs_modulus = 4.9e9
poisson_ratio = 0.44
a = 0.01  # 井筒半径

# 读取网格与函数空间
mesh = Mesh(domainpath)
V = VectorFunctionSpace(mesh, "CG", 2)

# 定义应力表达式 
def epsilon(u): return sym(grad(u))
def sigma(u):
    lam = (poisson_ratio * youngs_modulus) / ((1 + poisson_ratio)*(1 - 2 * poisson_ratio))
    mu = youngs_modulus / (2 * (1 + poisson_ratio))
    return lam * tr(epsilon(u)) * Identity(2) + 2 * mu * epsilon(u)

#  变分问题设置 + 边界条件
u = Function(V, name="Displacement")
v = TestFunction(V)

# ⚠️ 加载项：σ_H 向上（编号 14 = top）
x, y = SpatialCoordinate(mesh)
T = as_vector([0.0, sigma_H])  # 张力方向 y
F = inner(sigma(u), epsilon(v)) * dx - dot(T, v) * ds(14)

# ✅ 刚体自由度锚定
bcs = [
    DirichletBC(V.sub(0), Constant(0.0), 11),  # 左边界固定 x
    DirichletBC(V.sub(1), Constant(0.0), 13),  # 底边界固定 y
]

solve(F == 0, u, bcs=bcs)

# 构造应力张量
S = TensorFunctionSpace(mesh, "DG", 0)
sigma_tensor = Function(S, name="Stress")
sigma_tensor.interpolate(sigma(u))

# 提取数值 σ_r 和 σ_θ 沿 x 轴方向
n_points = 100
coords = mesh.coordinates.dat.data_ro
r_max = np.max(np.sqrt(coords[:, 0]**2 + coords[:, 1]**2))
r_vals_all = np.linspace(a + 1e-4, r_max - 1e-3, n_points)

# 沿 x 轴从井筒边缘往外等距取样。
cosθ, sinθ = 1.0, 0.0  # 沿 x 轴

def unit_vectors():
    return np.array([cosθ, sinθ]), np.array([-sinθ, cosθ])

r_used, σr_num, σθ_num = [], [], []

# θ = 0，对应方向单位向量 e_r 和 e_θ
for r in r_vals_all:
    pt = np.array([r, 0.0])
    try:
        σ = sigma_tensor.at(pt, tolerance=1e-10)
    except:
        continue
    er, et = unit_vectors()
    σ_r = er @ σ @ er
    σ_θ = et @ σ @ et
    r_used.append(r)
    σr_num.append(σ_r) # 用张量与方向向量内积提取 σr 和 σθ 分量
    σθ_num.append(σ_θ) 

r_used = np.array(r_used)
print(f"Collected {len(r_used)} points")

# 使用Kirsch 理论解（θ=0）
σr_exact = sigma_H * (1 - (a / r_used)**2)
σθ_exact = sigma_H * (1 + (a / r_used)**2)

# 可视化
plt.figure(figsize=(8, 5))
plt.plot(r_used, σr_num, "b-", label="σ_r (Numerical)")
plt.plot(r_used, σr_exact, "b--", label="σ_r (Analytical)")
plt.plot(r_used, σθ_num, "r-", label="σ_θ (Numerical)")
plt.plot(r_used, σθ_exact, "r--", label="σ_θ (Analytical)")
plt.xlabel("Radius r (m)")
plt.ylabel("Stress (Pa)")
plt.title("Kirsch Stress Validation: σ_r and σ_θ")
plt.legend()
plt.grid(True)
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/stress_validation_kirsch.png", dpi=300, bbox_inches="tight")
plt.show()