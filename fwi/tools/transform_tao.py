from enum import Enum, auto
from firedrake import *
from firedrake.petsc import PETSc
from slepc4py import SLEPc
from pyadjoint import Block, annotate_tape, get_working_tape, stop_annotating
from pyadjoint.optimization.tao_solver import OptionsManager, PETScVecInterface

class TransformType(Enum):
    PRIMAL = auto()
    DUAL = auto()


def transform(v, transform_type, *args, mfn_parameters=None, **kwargs):
    with stop_annotating():
        if mfn_parameters is None:
            mfn_parameters = {}
        mfn_parameters = dict(mfn_parameters)

        space = v.function_space()
        if not ufl.duals.is_primal(space):
            space = space.dual()
        if not ufl.duals.is_primal(space):
            raise NotImplementedError("Mixed primal/dual space case not implemented")
        comm = v.comm

        class M:
            def mult(self, A, x, y):
                if transform_type == TransformType.PRIMAL:
                    v = Cofunction(space.dual())
                elif transform_type == TransformType.DUAL:
                    v = Function(space)
                else:
                    raise ValueError(f"Unrecognized transform_type: {transform_type}")
                with v.dat.vec_wo as v_v:
                    x.copy(result=v_v)
                if isinstance(v, Cofunction):
                    solver_options = {}
                    u = v.riesz_representation(*args, **solver_options,
                                               measure_options=kwargs.get("measure_options", {}))
                else:
                    u = v.riesz_representation(*args, **kwargs)
                with u.dat.vec_ro as u_v:
                    u_v.copy(result=y)

        with v.dat.vec_ro as v_v:
            n, N = v_v.getSizes()
        M_mat = PETSc.Mat().createPython(((n, N), (n, N)),
                                         M(), comm=comm)
        M_mat.setOption(PETSc.Mat.Option.SYMMETRIC, True)
        M_mat.setUp()

        mfn = SLEPc.MFN().create(comm=comm)
        options = OptionsManager(mfn_parameters, None)
        options.set_default_parameter("fn_type", "sqrt")
        mfn.setOperator(M_mat)

        options.set_from_options(mfn)
        mfn.setUp()

        with v.dat.vec_ro as v_v:
            x = v_v.copy()
            y = v_v.copy()

        if y.norm(PETSc.NormType.NORM_INFINITY) == 0:
            x.zeroEntries()
        else:
            mfn.solve(y, x)
            if mfn.getConvergedReason() <= 0:
                raise RuntimeError("Convergence failure")

        if ufl.duals.is_primal(v):
            u = Function(space)
        else:
            u = Cofunction(space.dual())
        with u.dat.vec_wo as u_v:
            x.copy(result=u_v)

    if annotate_tape():
        block = TransformBlock(v, transform_type, *args, mfn_parameters=mfn_parameters, **kwargs)
        block.add_output(u.block_variable)
        get_working_tape().add_block(block)

    return u


class TransformBlock(Block):
    def __init__(self, v, *args, **kwargs):
        super().__init__()
        self.add_dependency(v)
        self._args = args
        self._kwargs = kwargs

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        lam, = adj_inputs
        return transform(lam, *self._args, **self._kwargs)

    def recompute_component(self, inputs, block_variable, idx, prepared):
        v, = inputs
        return transform(v, *self._args, **self._kwargs)

