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

# API

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

