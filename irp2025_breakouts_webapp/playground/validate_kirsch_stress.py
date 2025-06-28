from firedrake import *
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

class KirschStressValidator:
    def __init__(self, mesh_path, sigma_H, sigma_h, E, nu, a):
        self.mesh_path = mesh_path
        self.sigma_H = sigma_H
        self.sigma_h = sigma_h
        self.E = E
        self.nu = nu
        self.a = a

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

        # Apply σ_H 向上 (top), σ_h 向右 (right)
        F = inner(self._sigma(u), self._epsilon(v)) * dx \
            - dot(as_vector([self.sigma_h, 0]), v) * ds(12) \
            - dot(as_vector([0, self.sigma_H]), v) * ds(14)

        bcs = [
            DirichletBC(self.V.sub(0), Constant(0.0), 11),
            DirichletBC(self.V.sub(1), Constant(0.0), 13),
        ]
        solve(F == 0, u, bcs=bcs)

        self.sigma_tensor = Function(self.S, name="Stress")
        self.sigma_tensor.interpolate(self._sigma(u))

    def _kirsch_sigma_r(self, r, θ=0):
        a = self.a
        σ1 = self.sigma_H
        σ2 = self.sigma_h
        return (σ1 + σ2)/2 * (1 - (a/r)**2) + \
               (σ1 - σ2)/2 * (1 - 4*(a/r)**2 + 3*(a/r)**4) * np.cos(2*θ)

    def _kirsch_sigma_theta(self, r, θ=0):
        a = self.a
        σ1 = self.sigma_H
        σ2 = self.sigma_h
        return (σ1 + σ2)/2 * (1 + (a/r)**2) - \
               (σ1 - σ2)/2 * (1 + 3*(a/r)**4) * np.cos(2*θ)

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
            σ_r_kirsch = self._kirsch_sigma_r(r, θ=0)
            σ_θ_kirsch = self._kirsch_sigma_theta(r, θ=0)
            self.results.append({
                "r": r,
                "sigma_r_numerical": σ_r_num,
                "sigma_r_analytical": σ_r_kirsch,
                "sigma_theta_numerical": σ_θ_num,
                "sigma_theta_analytical": σ_θ_kirsch
            })

    def plot(self, outdir="outputs/kirsch_validation"):
        os.makedirs(outdir, exist_ok=True)
        df = pd.DataFrame(self.results)

        plt.figure()
        plt.plot(df.r, df.sigma_r_numerical, "b-", label="σ_r (Numerical)")
        plt.plot(df.r, df.sigma_r_analytical, "b--", label="σ_r (Kirsch)")
        plt.xlabel("Radius r (m)")
        plt.ylabel("σ_r (Pa)")
        plt.title("Radial Stress σ_r")
        plt.legend(); plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "sigma_r_validation.png"), dpi=300)

        plt.figure()
        plt.plot(df.r, df.sigma_theta_numerical, "r-", label="σ_θ (Numerical)")
        plt.plot(df.r, df.sigma_theta_analytical, "r--", label="σ_θ (Kirsch)")
        plt.xlabel("Radius r (m)")
        plt.ylabel("σ_θ (Pa)")
        plt.title("Circumferential Stress σ_θ")
        plt.legend(); plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "sigma_theta_validation.png"), dpi=300)

    def export(self, path="outputs/kirsch_validation/kirsch_data.csv"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df = pd.DataFrame(self.results)
        df.to_csv(path, index=False)


if __name__ == "__main__":
    validator = KirschStressValidator(
        mesh_path="meshes/domain1.msh",
        sigma_H=30e6,     # vertical
        sigma_h=20e6,     # horizontal
        E=6.66e9,
        nu=0.22,
        a=0.011           # 22 mm borehole
    )
    validator.plot()
    validator.export()