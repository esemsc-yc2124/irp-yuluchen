from firedrake import *
from fireshape import *
import numpy as np
from utilis.visualisation import plot_scalar_field
from mode_I_case import StrengthConstraint, OutwardPenalization
import ROL

def compute_plastic_shear_strain(u):
    eps = sym(grad(u))
    dev = eps - (1/3)*tr(eps)*Identity(2)
    shear_strain = sqrt(2*inner(dev, dev))
    return shear_strain

if __name__ == "__main__":
    domainpath = "meshes/domain.msh"

    model_parameters = {
        "youngs_modulus": 4.9e9,
        "poisson_ratio": 0.44,
        "tensile_strength": 0.9e6,
        "internal_friction_angle": 38.9,
        "internal_cohesion": 4.54e6,
        "insitu_stress_x": 1.0,
        "insitu_stress_y": 1.0,
    }

    params_dict = {
        'Step': {'Type': 'Trust Region'},
        'General': {'Secant': {'Type': 'Limited-Memory BFGS', 'Maximum Storage': 25}},
        'Status Test': {
            'Gradient Tolerance': 1e-2,
            'Step Tolerance': 1e-8,
            'Iteration Limit': 40
        }
    }

    for sigma in np.arange(10, 35 + 1, 2):  # MPa
        sv = sigma * 1e6
        # sh = sigma * 1e6
        sh = 0
        model_parameters["insitu_stress_x"] = sh
        model_parameters["insitu_stress_y"] = sv

        mesh = Mesh(domainpath)
        Q = FeControlSpace(mesh)
        S = FunctionSpace(mesh, "DG", 1)
        IP = H1InnerProduct(Q, fixed_bids=[11, 12, 13, 14])
        q = ControlVector(Q, IP)

        # ramping_factor 调大，看效果是否更显著
        J_strength = StrengthConstraint(Q, model_parameters=model_parameters, ramping_factor=5.0)
        J = J_strength + OutwardPenalization(Q)

        # ✅ 打印初始目标函数值
        print(f"[{sigma} MPa] Initial objective:", J.value(q, 1e-6))

        problem = ROL.OptimizationProblem(J, q)
        solver = ROL.OptimizationSolver(problem, ROL.ParameterList(params_dict, "Parameters"))
        solver.solve()

        u = J_strength.u
        shear_field = compute_plastic_shear_strain(u)
        shear_fn = Function(S, name="shear_strain")
        shear_fn.interpolate(shear_field)
        plot_scalar_field(shear_fn, sigma, outdir="outputs/figures", vmin=0, vmax=0.12)

        # ✅ 可视化破坏区域张应力限制
        tensile_fn = Function(S, name="tensile_limit")
        tensile_fn.interpolate(J_strength.tensile_limit)
        plot_scalar_field(tensile_fn, sigma, outdir="outputs/figures_tensile", vmin=0, vmax=1.0)