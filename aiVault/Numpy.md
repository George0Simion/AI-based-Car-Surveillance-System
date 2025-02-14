
import numpy as np

-> folosita pt operatii de algebra liniara

-> can create arrays of 1D, 2D, 3D: np.array($[1, 2, 3], [4,  5, 6], [7, 8, 9]$)
		*Properties of arrays:*
		- .shape
		- .ndim -> nr de dimensiuni
		- .size -> nr elementelor
		- .dtype -> tipul elementelor
		- .zeros()
		- .ones()
		- .full() -> np.full((2,3), 7)
		- .eye() -> matricea I
		- .arange() 
		- .linspace()
		- arr = np.array($[1, 2, 3], [4, 5, 6]$) -> arr +/*- 10
		- poate sa faca suma, mean, variatia standard, valoarea min max, mediana etc
		- .dot() / .matmul() -> inmultire matrici
		- np.linalg.inv(A) / np.linalg.det(A)
		- Broadcasting -> adauga un vector la fiecare linie a unei matrici
		- .reshape()
		- .T -> transpusa
		- poate sa concateneze vertical sau orizontal -> np.concatenate((A, B), axis=0) si np.hstack((A, B.T))

-> e implementata in C
-> tine vectorii in memorie continua pt performanta 