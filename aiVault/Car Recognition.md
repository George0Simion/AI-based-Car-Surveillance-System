
-> Codul contine mai multe parti de baza pt implementarea unui AI care face car recognition
1. Dictionarul cu tipurile de masini + labels pt fiecare tip
2. Dataframe (panda) pentru brand + imagini:
	* un dataframe care retine path-ul unui brand + path-urile de la imagini
3. Apoi impart dataset-ul in 3 part: train, validation, test
	* ~70%: training set
	* ~20%: validation set
	* ~10% test set
4. Transformatorul unei imagini:
	* prictic o normalizare a imaginii
5. Un dataset pt masini:
	* clasa care practic citeste fiecare imagine de la path si o preproceseaza
	* Ce se intampla: DataLoader-ul apeleaza getitem pt a obtine un sample, adica imagine si label
	* Citim imaginea folosindu-ne de opencv, si o "normalizam"
6. Cream Dataset SI Dataloaders pentru fiecare set
	* Acele dataset pe care l-am impartit creat deja, impartit in 3, care contine doar path-urile -> incepem sa l creem practic.
	* Adica pentru fiecare element din acel dataset, care e un path, creem dataset-ul cu imaginile reale, procesandu-le folosind clasa CarDataset
	* Instantiem si cele 3 loadere pt cele 3 noi dataseturi
7. Model setup:
	* Incepem sa creem modelul
	* II dam load( am ales REsNet-101, deja antrenat pe ImageNet )
	* II schimbam ultimul layer din 1000 care e default in numarul de marci de masina
	* Alegem CrossEntropy pentru loss function
	* Adam pentru optimizer
	* si un scheduler, care pratic zice ca la fiecare n epoci(step size in cazul nostru), sa mareasca rata de invatare(cu gamma)
	* ceva metadata(hyperparameters)
	* daca avem gpu bine daca nu bine
8. Functia de Validare:
	* Punem modelul pe modul de evaluare -> adica sa nu faca update-uri -> nici nu calculam gradientii deci nu updatam weight-urile -> deci evaluam starea curenta, fara nimic nou -> practic sa nu intre in ceva recursg ig, gen vrea sa vada cam cum e si apoi in vata ceva, face bakpropagate si toate alea
	* itelram prin fiecare batch dat de dataloader
	* vedem modelul nostru ce output are pe acel batch
	* facem ca o predictie sa vedem pe acel batch cam ce trebuia sa ne dea
	* comparam si calculam acuratetea
9. Training loop:
	* Iteram prin fiecare epoca
	* schimbam modelul pe modul de training -> adica inavata pe scurt
	* iteram prin fiecare imagina si label din dataloader
	* de fiecare data resetam gradientii -> sa nu i amestece de la batch la batch
	* vedem outputul pe care il are modelul pe batch -> practic sunt predictii
	* folosim functia de loss (care lucreaza cu gradientii) ca sa veem cat de pe langa suntem
	* updatam parametrii -> bakpropagation
	* step -> updatam parametrii -> optimizam -> Adam
	* vedem cat de mult pierdem pe toate batchurile intr=o epoca
	* masuram acuratetea intregului training
	* masuram cat de bine identifica masinile pe validation set