# MissingLink SDK example for TensorFlow

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

In order to run an experiment with MissingLink, you would need to first create a 
project and obtain the credentials on the MissingLink's web dashbash.

With the `owner_id` and `project_token`, you can run this example from terminal.
```bash
python mnist --owner_id 'owner_id' --project_token 'project_token'
```

Alternatively, you can copy these credentials and set them in `mnist.py`.
