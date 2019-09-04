import os
import pytest
import tempfile
import pickle
import numpy as np
from ogindia.utils import comp_array, comp_scalar, dict_compare
from ogindia.get_micro_data import get_calculator
from ogindia import SS, TPI, utils
from ogindia.parameters import Specifications
from taxcalc import GrowFactors

TOL = 1e-5

CUR_PATH = os.path.abspath(os.path.dirname(__file__))


@pytest.yield_fixture
def picklefile1():
    x = {'a': 1}
    pfile = tempfile.NamedTemporaryFile(mode='a', delete=False)
    pickle.dump(x, open(pfile.name, 'wb'))
    pfile.close()
    # must close and then yield for Windows platform
    yield pfile
    os.remove(pfile.name)


@pytest.yield_fixture
def picklefile2():
    y = {'a': 1, 'b': 2}

    pfile = tempfile.NamedTemporaryFile(mode='a', delete=False)
    pickle.dump(y, open(pfile.name, 'wb'))
    pfile.close()
    # must close and then yield for Windows platform
    yield pfile
    os.remove(pfile.name)


@pytest.yield_fixture
def picklefile3():
    x = {'a': np.array([100., 200., 300.]), 'b': 2}
    pfile = tempfile.NamedTemporaryFile(mode='a', delete=False)
    pickle.dump(x, open(pfile.name, 'wb'))
    pfile.close()
    # must close and then yield for Windows platform
    yield pfile
    os.remove(pfile.name)


@pytest.yield_fixture
def picklefile4():
    x = {'a': np.array([100., 200., 300.1]), 'b': 2}
    pfile = tempfile.NamedTemporaryFile(mode='a', delete=False)
    pickle.dump(x, open(pfile.name, 'wb'))
    pfile.close()
    # must close and then yield for Windows platform
    yield pfile
    os.remove(pfile.name)


def test_import_ok():
    import ogindia


@pytest.mark.full_run
@pytest.mark.parametrize('time_path', [False, True], ids=['SS', 'TPI'])
def test_run_small(time_path):
    from ogindia.execute import runner
    # Monkey patch enforcement flag since small data won't pass checks
    SS.ENFORCE_SOLUTION_CHECKS = False
    TPI.ENFORCE_SOLUTION_CHECKS = False
    SS.MINIMIZER_TOL = 1e-6
    TPI.MINIMIZER_TOL = 1e-6
    output_base = './OUTPUT'
    input_dir = './OUTPUT'
    user_params = {'frisch': 0.41, 'debt_ratio_ss': 0.4}
    runner(output_base=output_base, baseline_dir=input_dir, test=True,
           time_path=time_path, baseline=True, user_params=user_params,
           run_micro=False)


@pytest.mark.full_run
def test_constant_demographics_TPI():
    '''
    This tests solves the model under the assumption of constant
    demographics, a balanced budget, and tax functions that do not vary
    over time.
    In this case, given how initial guesss for the time
    path are made, the time path should be solved for on the first
    iteration and the values all along the time path should equal their
    steady-state values.
    '''
    output_base = './OUTPUT'
    baseline_dir = './OUTPUT'
    # Create output directory structure
    ss_dir = os.path.join(output_base, 'SS')
    tpi_dir = os.path.join(output_base, 'TPI')
    dirs = [ss_dir, tpi_dir]
    for _dir in dirs:
        try:
            print('making dir: ', _dir)
            os.makedirs(_dir)
        except OSError:
            pass
    spec = Specifications(run_micro=False, output_base=output_base,
                          baseline_dir=baseline_dir, test=False,
                          time_path=True, baseline=True, reform={},
                          guid='')
    user_params = {'constant_demographics': True,
                   'budget_balance': True,
                   'zero_taxes': True,
                   'maxiter': 2,
                   'eta': (spec.omega_SS.reshape(spec.S, 1) *
                           spec.lambdas.reshape(1, spec.J))}
    spec.update_specifications(user_params)
    spec.get_tax_function_parameters(None, False)
    # Run SS
    ss_outputs = SS.run_SS(spec, None)
    # save SS results
    utils.mkdirs(os.path.join(baseline_dir, 'SS'))
    ss_dir = os.path.join(baseline_dir, 'SS', 'SS_vars.pkl')
    pickle.dump(ss_outputs, open(ss_dir, 'wb'))
    # Run TPI
    tpi_output = TPI.run_TPI(spec, None)
    assert(np.allclose(tpi_output['bmat_splus1'][:spec.T, :, :],
                       ss_outputs['bssmat_splus1']))


def test_compare_pickle_file_bad(picklefile1, picklefile2):
    from ogindia.utils import pickle_file_compare
    assert not pickle_file_compare(picklefile1.name, picklefile2.name)


def test_compare_pickle_file_bad2(picklefile3, picklefile4):
    from ogindia.utils import pickle_file_compare
    assert not pickle_file_compare(picklefile3.name, picklefile4.name)


def test_compare_pickle_file_relative(picklefile3, picklefile4):
    from ogindia.utils import pickle_file_compare
    assert pickle_file_compare(
        picklefile3.name, picklefile4.name, relative=True)


def test_compare_pickle_file_basic(picklefile1):
    from ogindia.utils import pickle_file_compare
    assert pickle_file_compare(picklefile1.name, picklefile1.name)


def test_compare_dict_basic():
    from ogindia.utils import dict_compare
    lhs = {'a': 1, 'b': 2}
    rhs = {'c': 4, 'b': 2}
    assert not dict_compare('lhs.pkle', lhs, 'rhs.pkle', rhs, tol=TOL)


def test_compare_dict_more_lhs():
    from ogindia.utils import dict_compare
    lhs = {'a': 1, 'b': 2, 'c': 3}
    rhs = {'c': 4, 'b': 2}
    assert not dict_compare('lhs.pkle', lhs, 'rhs.pkle', rhs, tol=TOL)


def test_compare_dict_diff_ndarrays():
    from ogindia.utils import dict_compare
    lhs = {'a': np.array([1, 2, 3]), 'b': 2}
    rhs = {'a': np.array([1, 3]), 'b': 2}
    assert not dict_compare('lhs.pkle', lhs, 'rhs.pkle', rhs, tol=TOL)


def test_compare_dict_diff_ndarrays2():
    from ogindia.utils import dict_compare
    lhs = {'a': np.array([1., 2., 3.]), 'b': 2}
    rhs = {'a': np.array([1., 2., 3.1]), 'b': 2}
    assert not dict_compare('lhs.pkle', lhs, 'rhs.pkle', rhs, tol=TOL)


def test_comp_array_relative():
    x = np.array([100., 200., 300.])
    y = np.array([100.01, 200.02, 300.03])
    unequal = []
    assert not comp_array('test', y, x, 1e-3, unequal)
    assert comp_array('test', y, x, 1e-3, unequal, relative=True)


def test_comp_array_relative_exception():
    x = np.array([100., 200., 300.])
    y = np.array([100.01, 200.02, 300.03])
    unequal = []
    exc = {'var': 1e-3}
    assert comp_array('var', y, x, 1e-5, unequal,
                      exceptions=exc, relative=True)


def test_comp_scalar_relative():
    x = 100
    y = 100.01
    unequal = []
    assert not comp_scalar('test', y, x, 1e-3, unequal)
    assert comp_scalar('test', y, x, 1e-3, unequal, relative=True)


def test_comp_scalar_relative_exception():
    x = 100
    y = 100.01
    unequal = []
    exc = {'var': 1e-3}
    assert comp_scalar('var', y, x, 1e-5, unequal,
                       exceptions=exc, relative=True)


def test_compare_dict_diff_ndarrays_relative():
    lhs = {'a': np.array([100., 200., 300.]), 'b': 2}
    rhs = {'a': np.array([100., 200., 300.1]), 'b': 2}
    assert dict_compare('lhs.pkle', lhs, 'rhs.pkle',
                        rhs, tol=1e-3, relative=True)


def test_get_micro_data_get_calculator():
    reform = {2017: {
        '_rate1': [0.09],
        '_rate2': [0.135],
        '_rate3': [0.225],
        '_rate4': [0.252]
        }}

    calc = get_calculator(baseline=False, calculator_start_year=2017,
                          reform=reform, data='pitSmallData.csv',
                          gfactors=GrowFactors(),
                          records_start_year=2017)
    assert calc.current_year == 2017
