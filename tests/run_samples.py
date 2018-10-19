# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
"""
test sample jupyter notebooks in nimbusml
"""
import os

import nbconvert
import nbformat

# Using nbconvert to execute ipython notebooks
# Sample example here :
# https://github.com/elgehelge/nbrun/blob/master/nbrun.py
def run_notebook(path, timeout=400, retry=3):
    print("Running notebook : {0}".format(path))
    ep = nbconvert.preprocessors.ExecutePreprocessor(
        extra_arguments=["--log-level=40"],
        timeout=timeout,
    )
    path = os.path.abspath(path)
    assert path.endswith('.ipynb')
    nb = nbformat.read(path, as_version=4)
    try:
        ep.preprocess(nb, {'metadata': {'path': os.path.dirname(path)}})
        print("Success\n")
    except Exception as e:
        if isinstance(e, RuntimeError) and retry != 0:
            return run_notebook(
                path, timeout=timeout, retry=retry - 1)

        raise Exception('Error running notebook {0}:  {1}'.format(path, e))

def run_notebooks(names=None):
    # get list of notebooks to run
    this = os.path.abspath(os.path.dirname(__file__))
    nb_dir = os.path.normpath(os.path.join(this, '..', 'samples'))
    if not os.path.exists(nb_dir):
        raise FileNotFoundError("Unable to find '{0}'.".format(nb_dir))
    notebooks = [_ for _ in os.listdir(nb_dir)
            if os.path.splitext(_)[-1] == '.ipynb']

    # Use this list to ommit specific notebooks from the test
    skip_list = []
    # Use this list if you quickly want to test one or two notebooks by partial
    # name; e.g. ['1.2', '3.1']
    run_only_list = []

    for nb in skip_list:
        # Requires verbatim names
        notebooks.remove(nb)
    
    if run_only_list:
        original_list = os.listdir(nb_dir)
        notebooks = []
        for nb in run_only_list:
            # Allows for partial name matching to quickly test single notebooks
            notebooks += [_ for _ in original_list if nb in _]
    
    if len(notebooks) == 0:
        raise FileNotFoundError(
            "Unable to find notebooks in '{0}'".format(nb_dir))

    # run the notebooks
    for nb in notebooks:
        nbfullpath = os.path.join(nb_dir, nb)
        run_notebook(nbfullpath)

if __name__ == '__main__':
    run_notebooks()