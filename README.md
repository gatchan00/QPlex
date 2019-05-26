# QPlex
Hackathon Qiskit IBM 2019

# License
MIT License

# Pre-requisites

Clone or download repo

Go to the repo and use conda to create a virtual environment:

```
conda env create -f environment.yml
```

then, activate your environment

```
conda activate Qiskitenv
```

# How does this work

This software is prepared to minimize functions like &theta;<sub>o</sub> x + &theta;<sub>1</sub>y, applying the inequality x + y &le; &beta;, where you can choose your &theta;<sub>0</sub>, &theta;<sub>1</sub> and &beta;

# API

A function called *wrapper_optimiza_f* will launch the optimization process, you need to provide:

 1. precision --> the number of qbits used for each variable
 2. coefs_param --> A list with the coeficients of each variable
 3. beta --> &beta; for the inequality constraint
 
You will get a dictionary with the value of each variable

```
from qplex_core import *

precision = 6
coefs_param = [2, -3]
beta = 7
a = wrapper_optimiza_f(precision, coefs_param, beta)
print(a)
```

You will get the desired value this way

```
{0: 0, 1: 7}
```

# API - REST

* **URL**

  `http://127.0.0.1:3333/api/optimize`

* **Method:**

  `POST`

* **Data:**

  `application/json`

```
{
  "variables": [2, -3],
  "restriction": 7,
  "precision": 6
}
```
  
* **Success Response:**

  `Code: 200`

```
{
  "results": {
      "x": 0,
      "y": 7
  },
  "status": "OK"
}
```

