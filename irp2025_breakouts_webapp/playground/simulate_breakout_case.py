from firedrake import *
from fireshape import *
import numpy as np
import os
from utilis.visualisation import plot_scalar_field, plot_mesh
from mode_I_case import StrengthConstraint, OutwardPenalization
import ROL

if __name__ == "__main__":

    # domainpath = "meshes/domain.msh"
    domainpath = "meshes/domain1.msh"  # 使用新的网格文件

    # 固定参数（Tenino sandstone）
    model_parameters = {
        "youngs_modulus": 6.66e9,
        "poisson_ratio": 0.22,
        "tensile_strength": 2.47e6,
        # cohesion and friction angle are unused => do not have Mohr-Coulomb 项
        "internal_friction_angle": 0.0,
        "internal_cohesion": 0.0,
        # σ_H will be set in the loop, σ_h is fixed
        "insitu_stress_x": 0.0, 
        "insitu_stress_y": 20e6  
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

    mesh_initial = Mesh(domainpath)

    os.makedirs("outputs/breakout_fig4", exist_ok=True)

    for sigma_H in [40e6, 50e6, 60e6, 70e6]:  # MPa -> Pa
        model_parameters["insitu_stress_x"] = sigma_H

        mesh = Mesh(domainpath)
        Q = FeControlSpace(mesh)
        IP = H1InnerProduct(Q, fixed_bids=[11, 12, 13, 14])
        q = ControlVector(Q, IP)

        # 固定加载，不使用 ramping
        J = StrengthConstraint(Q, model_parameters=model_parameters, ramping_factor=1.0) + OutwardPenalization(Q)

        problem = ROL.OptimizationProblem(J, q)
        solver = ROL.OptimizationSolver(problem, ROL.ParameterList(params_dict, "Parameters"))
        solver.solve()

        # 保存图像
        suffix = int(sigma_H * 1e-6)  # 单位 MPa 作为文件名后缀
        plot_mesh(mesh, mesh_initial, suffix, outdir="outputs/breakout_fig4")
        plot_scalar_field(J.a.tensile_limit, suffix, outdir="outputs/breakout_fig4", vmin=0, vmax=1.0)