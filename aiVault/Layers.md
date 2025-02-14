
-> unitatea pt neaural network
-> transforma un input intr-o reprezentare ceva specifica
-> are: weights(parametrii): valori pe care le accepta si adapteaza cat isi da train, biases: inca ceva parametrii pe care ii modifica, functia de activare: adauga neliniaritate
-> exemplu de un layer full conecatat: layer = torch.nn.Linear(in_features=3, out_features=2)
care matematic este: $Y = W  X + b$, unde:
X = input(shape[batch_size, in_features])
W = weights(shape: [out_features, in_features])
b = biases(shape: [out_features])

