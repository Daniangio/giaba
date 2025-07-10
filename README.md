# giaba


subject to constraint1 {i in J,j in J: i < j}:
    sum{a in J} y[i,a] - sum{a in J} y[j,a] - 1 <= M * (1 - p[i,j]);

subject to constraint2 {i in J,j in J: i < j}:
    sum{a in J} y[j,a] + 1 - sum{a in J} y[i,a] <= M * p[i,j];