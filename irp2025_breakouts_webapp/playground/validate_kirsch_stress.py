from firedrake import *
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

class KirschStressValidator:
    def __init__(self, mesh_path, sigma_H, E, nu, borehole_radius):
        self.mesh_path = mesh_path
        self.sigma_H = sigma_H
        self.E = E
        self.nu = nu
        self.a = borehole_radius

        # Firedrake setup
        self.mesh = Mesh(mesh_path)
        self.V = VectorFunctionSpace(self.mesh, "CG", 2)
        self.S = TensorFunctionSpace(self.mesh, "DG", 0)

        self._solve_elasticity()
        self._extract_stress_profile()

    def _epsilon(self, u):
        return sym(grad(u))

    def _sigma(self, u):
        lam = (self.nu * self.E) / ((1 + self.nu) * (1 - 2 * self.nu))
        mu = self.E / (2 * (1 + self.nu))
        return lam * tr(self._epsilon(u)) * Identity(2) + 2 * mu * self._epsilon(u)

    def _solve_elasticity(self):
        u = Function(self.V, name="Displacement")
        v = TestFunction(self.V)

        T = as_vector([0.0, self.sigma_H])  # apply on top
        F = inner(self._sigma(u), self._epsilon(v)) * dx - dot(T, v) * ds(14)
        bcs = [
            DirichletBC(self.V.sub(0), Constant(0.0), 11),
            DirichletBC(self.V.sub(1), Constant(0.0), 13),
        ]
        solve(F == 0, u, bcs=bcs)

        self.sigma_tensor = Function(self.S, name="Stress")
        self.sigma_tensor.interpolate(self._sigma(u))

    def _extract_stress_profile(self):
        coords = self.mesh.coordinates.dat.data_ro
        r_max = np.max(np.sqrt(coords[:, 0]**2 + coords[:, 1]**2))
        r_vals_all = np.linspace(self.a + 1e-4, r_max - 1e-3, 100)

        er = np.array([1.0, 0.0])
        et = np.array([0.0, 1.0])

        self.results = []

        for r in r_vals_all:
            pt = np.array([r, 0.0])
            try:
                σ = self.sigma_tensor.at(pt, tolerance=1e-10)
            except:
                continue
            σ_r_num = er @ σ @ er
            σ_θ_num = et @ σ @ et
            σ_r_kirsch = self.sigma_H * (1 - (self.a / r)**2)
            σ_θ_kirsch = self.sigma_H * (1 + (self.a / r)**2)
            self.results.append({
                "r": r,
                "sigma_r_numerical": σ_r_num,
                "sigma_r_analytical": σ_r_kirsch,
                "sigma_theta_numerical": σ_θ_num,
                "sigma_theta_analytical": σ_θ_kirsch
            })

    def plot(self, outdir="outputs/kirsch_validation"):
        os.makedirs(outdir, exist_ok=True)
        r = [d["r"] for d in self.results]
        σr_num = [d["sigma_r_numerical"] for d in self.results]
        σr_exact = [d["sigma_r_analytical"] for d in self.results]
        σθ_num = [d["sigma_theta_numerical"] for d in self.results]
        σθ_exact = [d["sigma_theta_analytical"] for d in self.results]

        # σ_r
        plt.figure()
        plt.plot(r, σr_num, "b-", label="σ_r (Numerical)")
        plt.plot(r, σr_exact, "b--", label="σ_r (Kirsch)")
        plt.xlabel("Radius r (m)")
        plt.ylabel("σ_r (Pa)")
        plt.title("Radial Stress σ_r")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "sigma_r_validation.png"), dpi=300)
        plt.close()

        # σ_θ
        plt.figure()
        plt.plot(r, σθ_num, "r-", label="σ_θ (Numerical)")
        plt.plot(r, σθ_exact, "r--", label="σ_θ (Kirsch)")
        plt.xlabel("Radius r (m)")
        plt.ylabel("σ_θ (Pa)")
        plt.title("Circumferential Stress σ_θ")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "sigma_theta_validation.png"), dpi=300)
        plt.close()

    def export(self, path="outputs/kirsch_validation/kirsch_data.csv"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df = pd.DataFrame(self.results)
        df.to_csv(path, index=False)


if __name__ == "__main__":
    # validator = KirschStressValidator(
    #     mesh_path="meshes/domain.msh",
    #     sigma_H=10e6,
    #     E=4.9e9,
    #     nu=0.44,
    #     borehole_radius=0.011
    # )

    validator = KirschStressValidator(
        mesh_path="meshes/domain1.msh",   
        sigma_H=10e6,
        E=6.66e9,
        nu=0.22,
        borehole_radius=0.011             
    )

    validator.plot()
    validator.export()