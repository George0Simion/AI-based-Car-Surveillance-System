
-> arata cat de mult trebuie weight-ul sa se schimbe ca sa minimizeze loss-ul
-> derivata partiala de gradul 1 a functiei

-> practic, se face pentru loss function, si ne spune cum sa ajustam weight-urile ca sa reducem eroarea
-> exemplu:
			$\huge  \frac{\partial L}{\partial W}$
			
		unde: - L este loss function(ex: Cross Entropy, Mean Squared Error)
			   - W este matricea de weight-uri

-> edge caseuri pt gradienti:
* vanishing gradients: gradienti mici -> nu invata modelul
* Exploding gradients: gradienti foarte mari -> in invatari nestabile