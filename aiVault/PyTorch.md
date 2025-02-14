
-> foloseste Tensori 
-> Tensor: structura de data, ca un array multi-dimensional dar cu GPU acceleration
-> practic putem sa selectam data rulam modelul pe GPU, adica prin cuda sau pe cpu
-> atribute:
	- .shape()
	- .dtype()
	- .device()
-> conversie: .to()
-> operatii cu tensori:
	a = torch.tensor([1.0, 2.0, 3.0])
	b = torch.tensor([4.0, 5.0, 6.0]) 
	print(a + b) # Element-wise addition 
	print(a - b) # Element-wise subtraction 
	print(a * b) # Element-wise multiplication 
	print(a / b) # Element-wise division
	print(a @ b) # Dot product
	print(torch.matmul(a, b)) # Same as @

	x = torch.tensor([[1, 2], [3, 4]])

	print(torch.sum(x))     # Sum of all elements
	print(torch.mean(x.float()))  # Mean
	print(torch.std(x.float()))   # Standard deviation
	print(torch.min(x))     # Minimum value
	print(torch.max(x))     # Maximum value

-> Autograd: face gradientul 
		- .backward() -> derivata? -> da, derivata, dar la o functie cu mai multe variabile e gradientul pt ca face derivata dupa fiecare variabila din F ig
		- .detach() -> elimina trackingul, adica nu mai salveaza in spate o functie obiect pentru fiecare input a unui tensor

-> Neural Networks: nn -> are layere predefinite
-> Optimizers:
		- optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
		- .zero_grad() -> reseta gradientii
		- .backward() -> computeaza gradientii
		- .step() -> updateaza paramterii
		- edge case uri pt optimize-ri:
			- lr e prea mare -> explodeaza gradientul?
			  lr e prea mic -> converge prea incet

-> Datasets si DataLoader
				