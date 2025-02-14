
-> updateaza weight-urile in functie de gradienti

-> Gradient Decent: 
$$
	\huge W = W - \eta \cdot \frac{\partial L}{\partial W}
$$
unde:
- $W$ = weight 
- $\eta$ = learning rate 
- $\frac{\partial L}{\partial W}$ = gradientul functii de loss

-> tipuri de optimezere:
- Stochastic Gradient Descent(SGD): *optimizer = torch.optim.SGD(model.parameters(), lr=0.01) -> simplu dar incet pentru modele mai mari
* Momentum: *optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)* -> accelereaza update urile pt modele complexe
* Adam Optimizers: combina momentum cu adaptive learning
