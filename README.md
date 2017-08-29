# MissingLinkAI SDK example for TensorFlow

## Requirements

You need Python 2.7 or 3.5 on your system to run this example.

To install the dependency:
- You are strongly recommended to use [`vertualenv`](https://virtualenv.pypa.io/en/stable/) to create a sandboxed environment for individual Python projects
```bash
pip install virtualenv
```

- Create and activate the virtual environment
```bash
virtualenv .venv
source .venv/bin/activate
```

- Install dependency libraries
```bash
pip install -r requirements.txt
```

## Run

In order to run an experiment with MissingLinkAI, you would need to first create a
project and obtain the credentials on the MissingLinkAI's web dashboard.

With the `owner_id` and `project_token`, you can run this example from terminal.
```bash
python mnist.py --owner-id 'owner_id' --project-token 'project_token'
python mnist_with_epoch_loop.py --owner-id 'owner_id' --project-token 'project_token'
```

Alternatively, you can copy these credentials and set them in source files.

## Examples

These examples train classification models for MNIST dataset.

- [mnist.py](https://github.com/missinglinkai/missinglink-tensorflow-example/blob/master/mnist.py): training with iterations/steps using single loop.
- [mnist_with_epoch_loop.py](https://github.com/missinglinkai/missinglink-tensorflow-example/blob/master/mnist_with_epoch_loop.py): training with epochs and batches using nested loops.
