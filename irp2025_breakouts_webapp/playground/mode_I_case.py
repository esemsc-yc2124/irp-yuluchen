from firedrake import *
from fireshape import *
import ROL
from numpy import tan, deg2rad
from utilis.visualisation import plot_mesh, plot_scalar_field

from fireshape.objective import PDEconstrainedObjective, DeformationObjective

class StrengthConstraint(PDEconstrainedObjective):
    """
    核心目标函数类，基于岩石破坏理论（主应力）
    计算的是 tensile failure（最大拉应力超过极限）发生的严重程度。
    该类继承自PDEconstrainedObjective，使用Firedrake进行有限元求解。
    """
    def __init__(self, *args, model_parameters, ramping_factor,  **kwargs):
        super().__init__(*args, **kwargs)
        self.model_parameters = model_parameters
        self.mesh = self.Q.mesh_m
        self.set_model_parameters(ramping_factor)
        
        degree = 2
        V = VectorFunctionSpace(self.mesh, "Lagrange", degree)
        # create container for variable of interest
        self.S = FunctionSpace(self.mesh, "DG", degree-1)
        self.tensile_limit = Function(self.S, name="Tensile limit")
        self.shear_limit = Function(self.S, name="Shear limit")
        self.S_r = FunctionSpace(self.Q.mesh_r, "DG", degree-1)
        # define VTKFile to store tensile_limit after each objective evaluation
        self.file = VTKFile("convergence_history/limits_"+str(self.ramping_factor)+".pvd")
        # The Lame parameters
        self.mu = self.E / (2.0 * (1.0 + self.nu))
        self.lmda = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))

        self.epsilon = lambda u: 0.5*(grad(u) + grad(u).T)
        # self.Id = Identity(mesh.geometric_dimension())
        self.Id = Identity(self.mesh.geometric_dimension())
        self.sigma = lambda u: (self.lmda * tr(self.epsilon(u)) * self.Id +
                                self.mu * 2 * self.epsilon(u))

        self.u = fd.Function(V, name="Displacement")
        self.v = TestFunction(V)

        self.n = FacetNormal(self.mesh)

        # Define the weak form of the PDE
        F = (inner(self.sigma(self.u), self.epsilon(self.v))*dx -
             (-self.sh)*dot(self.n,self.v)*ds(11) -
             (-self.sv)*dot(self.n,self.v)*ds(14)) # 状态变量 u 是弹性位移，使用 Lame 参数构造应力张量 σ
        bc = [DirichletBC(V.sub(0), 0, 12), DirichletBC(V.sub(1), 0, 13)]
        prb = NonlinearVariationalProblem(F, self.u, bcs=bc)
        self.state_solver = NonlinearVariationalSolver(prb)
        self.compute_strength_constraint() # 计算主应力 eigenvalues（p1, p2）最大拉应力为 sig2 = -p1, 判定是否超过极限 ts，并归一化为 tensile_limit

    def set_model_parameters(self, ramping_factor):
        # Ramping multiplier
        self.ramping_factor = Constant(ramping_factor) # Stress ramping multiplier
        # Elasticity parameters
        self.E = Constant(self.model_parameters["youngs_modulus"]) # Young's modulus
        self.nu = Constant(self.model_parameters["poisson_ratio"]) # Poisson's ratio
        # Strength parameters
        self.ts = Constant(self.model_parameters["tensile_strength"]) # Tensile strength
        self.ifc = Constant(tan(deg2rad(self.model_parameters["internal_friction_angle"]))) # Internal friction coefficient
        self.coh = Constant(self.model_parameters["internal_cohesion"]) # Internal cohesion
        # In-situ stress
        self.sv = self.ramping_factor * Constant(self.model_parameters["insitu_stress_y"]) # Vertical in-situ stress (+): compression
        self.sh = self.ramping_factor * Constant(self.model_parameters["insitu_stress_x"]) # Horizontal in-situ stress (+): compression

    def eigvals2d(self, sym_tensor):
        # Compute the first invariant (I1 = trace of the tensor)
        I1 = tr(sym_tensor)
        # Compute the second invariant (I2 = det(sym_tensor))
        I2 = det(sym_tensor)
        # Compute the discriminant of the quadratic equation
        discriminant = sqrt(I1**2 - 4 * I2)
        # Compute the two eigenvalues using the quadratic formula
        P1 = (I1 + discriminant) / 2
        P2 = (I1 - discriminant) / 2
        return P1, P2

    def compute_strength_constraint(self):
        '''
        In rock mechanics, the maximum principal stress is denoted as σ1,
        and a positive sign means compressive stresses (opposite to engineering sign).
        p2 is the min principal stress i.e. max compression, so sig1 = -p2.
        p1 is the max principal stress i.e. max tension, so sig2 = -p1.
        '''
        p1, p2 = self.eigvals2d(self.sigma(self.u))
        sig1, sig2 = -p2, -p1
        '''
        The tensile strength is a positive constant, but max tension is reached when
        sig2 is negative and is smaller then -1 * tensile strength.
        Therfore when (sig2+ts) is negative the tensile limit is reached.
        The tensile_limit is rescaled with ts, meaning if it's 1 the max tensile stress is
        twice the value of the tensile strength.
        '''
        return (abs(sig2+self.ts) - (sig2+self.ts)) / (2*self.ts)

    # 会先求解 PDE，然后积分 tensile_limit 作为目标函数值
    def objective_value(self):
        # evaluate objective function J
        self.state_solver.solve()
        tensile_limit = self.compute_strength_constraint()
        self.tensile_limit.interpolate(tensile_limit)
        strenght_penalty = Constant(1e0)
        strength_constraint = tensile_limit * strenght_penalty
        return assemble(strength_constraint * dx(degree=3))

# 抑制孔洞边界向内变形 避免 ControlVector(Q) 把孔向内推（使 mesh 折叠或物理无意义）
class OutwardPenalization(DeformationObjective):
    '''
    Penalisation of the mesh deformation towards the inside of the hole: n is
    the outward normal to the boundary, dot(dT, n) is the mesh deformation in
    that direction, and ds(15) is the integral over the hole boundary.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mesh_deformation_penalty = Constant(1e3)
        self.n = FacetNormal(self.Q.mesh_r)

    def value_form(self):
        # Penalize inward deformation
        dT = self.Q.T - self.Q.id
        dT_dot_n = dot(dT, self.n)
        inward_mesh_deformation = ((abs(dT_dot_n) + dT_dot_n) * .5) ** 2 # 这个表达式只对“向内”的 dT⋅n 有贡献，且平方加重惩罚
        return inward_mesh_deformation * self.mesh_deformation_penalty * ds(15, degree=3)

    def derivative_form(self, test):
        return fd.derivative(self.value_form(), self.Q.T, test)


if __name__ == "__main__":

    domainpath = "meshes/domain.msh"

    params_dict = {'Step': {'Type': 'Trust Region'},
                   'General': {'Secant': {'Type': 'Limited-Memory BFGS',
                                          'Maximum Storage': 25}},
                   'Status Test': {'Gradient Tolerance': 1e-2,
                                   'Step Tolerance': 1e-8,
                                   'Iteration Limit': 40}}

    model_parameters = {
        "youngs_modulus": 1e9,
        "poisson_ratio": 0.3,
        "tensile_strength": 1.5e6,
        "internal_friction_angle": 20,
        "internal_cohesion": 40.55e6,
        "insitu_stress_x": 1e6,
        "insitu_stress_y": 7.7e6}

    # model_parameters = {
    #     "youngs_modulus": 4.9e9,              # 4.9 GPa（confining = 2 MPa 对应的 E）
    #     "poisson_ratio": 0.44,                # ν = 0.44（同一行）
    #     "tensile_strength": 0.9e6,            # 1.0 MPa，估计值，弱砂岩典型
    #     "internal_friction_angle": 38.9,      # φ，论文给出
    #     "internal_cohesion": 4.54e6,          # c，论文给出
    #     "insitu_stress_x": 27e6,               # 初始设为 0，后续加载
    #     "insitu_stress_y": 27e6
    # }

    # setup problem
    mesh = Mesh(domainpath)
    mesh_initial = Mesh(domainpath)
    Q = FeControlSpace(mesh)

    File = VTKFile("tensile_limit.pvd")
    
    for rampf in np.linspace(0.1, 1, num=30, endpoint=True):
        mesh = Mesh(Q.T)
         # 1. 重建 mesh
        Q = FeControlSpace(mesh)
        # 2. 定义内积和控制向量
        IP = H1InnerProduct(Q, fixed_bids=[11, 12, 13, 14])
        q = ControlVector(Q, IP)
        # 3. 定义目标函数
        J = StrengthConstraint(Q, model_parameters=model_parameters, ramping_factor=rampf) + OutwardPenalization(Q)
        # assemble and solve ROL optimization problem
        params = ROL.ParameterList(params_dict, "Parameters")
        # 4. 定义 ROL 优化问题 启动优化器求解
        problem = ROL.OptimizationProblem(J, q) 
        ROLsolver = ROL.OptimizationSolver(problem, params)
        ROLsolver.solve()
        File.write(J.a.tensile_limit, time=rampf)
        plot_mesh(mesh, mesh_initial, rampf)
        plot_scalar_field(J.a.tensile_limit, rampf)
