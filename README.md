# Intro
Commands starting with **#** must be run as root

Commands starting with **$** must be run as simple user

Commands starting **(venv)** must be run from inside a virtualenv

# Install apt packages (Debian 8)
    $ sudo apt-get update
    $ sudo apt-get install -y build-essential python3-dev python3-virtualenv \
        libopenblas-dev gfortran virtualenv pkg-config libfreetype6-dev \
        libpng12-dev git

# Create and activate the virtualenv
    $ virtualenv -p python3 venv
    $ . ./venv/bin/activate

# Install python packages
    (venv) pip install numpy scipy
    (venv) pip install scikit-learn
    (venv) pip install jupyter
    (venv) pip install pandas
    (venv) pip install matplotlib
    (venv) pip install theano
    (venv) pip install lasagne
    (venv) pip install grip

# Launch jupyter
    (venv) jupyter-notebook

# What next?
* jupyter examples
* numpy examples
