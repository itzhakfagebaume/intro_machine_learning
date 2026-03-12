import pickle
import numpy as np
import matplotlib.pyplot as plt

nom_fichier = "result.txt"
def simulate_prediction(array, error_rate):
    random_booleans = np.random.rand(len(array)) < error_rate
    # Comparer les deux tableaux
    predictions = np.where(random_booleans, ~array, array)

    return predictions


def calculate_erm_multiple(train_set, true_risks, size):
    """
    Calculates the Empirical Risk Minimization (ERM) for multiple prophets.

    :param train_set: Training set, an array or list of values (booleans).
    :param true_risks: Array containing the true risk (error rate) of each prophet.
    :return: The true risk of the selected prophet (the one with the lowest empirical risk).
    """
    # Initialize empirical errors array
    n_prophets = len(true_risks)
    empirical_errors = np.zeros(n_prophets)
    random_indices = np.random.choice(len(train_set[0]), size=size, replace=False)

    # Calculate empirical errors for each prophet
    for i, true_risk in enumerate(true_risks):
        # Generate predictions for the prophet using simulate_prediction
        one_train_set = train_set[i][random_indices]

        predictions = simulate_prediction(one_train_set, true_risk)

        # Calculate the empirical error as the proportion of incorrect predictions
        empirical_errors[i] = np.mean(predictions != one_train_set)

        # print("predictions ")
        # print(predictions)
        # print(train_set)
        # print("prophete " + str(i) + " true risk " + str(true_risk) + " empirical " + str(empirical_errors[i]))

    # Find the index of the prophet with the lowest empirical error
    # print(empirical_errors)
    min_error_indices = np.where(empirical_errors == empirical_errors.min())[0]

    # Choose randomly among the indices with the minimum error
    chosen_index = np.random.choice(min_error_indices)

    # Return the true risk of the chosen prophet
    return chosen_index


def algo(data, size, trois_quatre=False):


    test_set = data['test_set']
    train_set = data['train_set']

    approximation_error = np.min(data['true_risk'])
    best_prophet_index = np.argmin(data['true_risk'])

    approximation_errors = []
    estimation_errors = []
    test_errors = []
    count = 0
    count_one_pourcent = 0

    # Répéter l'expérience 100 fois
    n_experiments = 100
    for _ in range(n_experiments):
        # one_train_set = np.random.choice(train_set[0], size=size,
        #                                  replace=False)  ## choisis que la premiere ligne verifier ca $$$

        chosen_prophet = calculate_erm_multiple(train_set, data['true_risk'], size)
        if chosen_prophet == best_prophet_index:
            count += 1

        # Étape 2 : Évaluation sur l'ensemble de test
        test_predictions = simulate_prediction(test_set[chosen_prophet], data['true_risk'][chosen_prophet])

        test_error = np.mean(test_predictions != test_set[chosen_prophet])
        # print("prophete avec true risk ",data['true_risk'][chosen_prophet], "moyenne", test_error)
        # estimation error is the difference between the true risk of the best and selected prophet
        true_error = data['true_risk'][chosen_prophet]

        estimation_error = abs(approximation_error - true_error)
        if trois_quatre == True:
            if estimation_error < 0.01:
                count_one_pourcent += 1

        # approximation error is the true risk of the best available prophet

        # Ajouter les erreurs dans les listes
        estimation_errors.append(estimation_error)
        approximation_errors.append(approximation_error)
        test_errors.append(test_error)

    avg_estimation_error = np.mean(estimation_errors)
    avg_approximation_error = np.mean(approximation_errors)


    print("Error of mean estimation :", avg_estimation_error)
    print("Mean approximation  :", avg_approximation_error)
    print("Mean test error on test set", np.mean(test_errors))
    print(f"The best prophete as been choose {count} time on  {n_experiments} ")
    if trois_quatre:
        print(f"The best prophete as been choose with 1% error {count_one_pourcent}"
              f" time on {n_experiments} ")

    # with open(nom_fichier, 'a') as fichier:
    #     fichier.write(f"Erreur d'estimation moyenne : {avg_estimation_error}\n")
    #     fichier.write(f"L'approximation moyenne : {avg_approximation_error}\n")
    #     fichier.write(f"Moyenne test error sur train set : {np.mean(test_errors)}\n")
    #     fichier.write(f"Le meilleur prophète a été choisi {count} fois sur {n_experiments} essais.\n")

        # if trois_quatre:
        #     fichier.write(f"Le meilleur prophète a été choisi avec une erreur de 1% {count_one_pourcent} "
        #                   f"fois sur {n_experiments} essais.\n")


    return avg_estimation_error,avg_approximation_error,np.mean(test_errors),estimation_errors


def print_results(results):
    # Créer les en-têtes du tableau
    headers = ["k", "m", "avg_approximation_error", "avg_estimation_error", "avg_test_error"]
    header_line = f"{headers[0]:<5} | {headers[1]:<5} | {headers[2]:<25} | {headers[3]:<25} | {headers[4]:<15}"
    separator = "-" * len(header_line)

    # Afficher les en-têtes
    print(separator)
    print(header_line)
    print(separator)

    # Initialiser une variable pour suivre la valeur actuelle de k
    previous_k = None

    # Afficher chaque ligne des résultats
    for result in results:
        # Ajouter une ligne vide si on change de groupe k
        if previous_k is not None and result['k'] != previous_k:
            print()  # Saut de ligne

        # Afficher les données de la ligne actuelle
        print(
            f"{result['k']:<5} | {result['m']:<5} | {result['avg_approximation_error']:<25.6f} | {result['avg_estimation_error']:<25.6f} | {result['avg_test_error']:<15.6f}")

        # Mettre à jour la valeur précédente de k
        previous_k = result['k']

    print(separator)


def Scenario_1():
    """
    Question 1.
    2 Prophets 1 Game.
    You may change the input & output parameters of the function as you wish.
    """
    ############### YOUR CODE GOES HERE ###############
    with open('scenario_one_and_two_prophets.pkl', 'rb') as f:
        data = pickle.load(f)
    algo(data, 1)


def Scenario_2():
    """
    Question 2.
    2 Prophets 10 Games.
    You may change the input & output parameters of the function as you wish.
    """
    ############### YOUR CODE GOES HERE ###############
    with open('scenario_one_and_two_prophets.pkl', 'rb') as f:
        data = pickle.load(f)
    algo(data, 10)


def Scenario_3():
    """
    Question 3.
    500 Prophets 10 Games.
    You may change the input & output parameters of the function as you wish.
    """
    ############### YOUR CODE GOES HERE ###############
    with open("scenario_three_and_four_prophets.pkl", 'rb') as f:
        data = pickle.load(f)
    algo(data, 10, True)


def Scenario_4():
    """
    Question 4.
    500 Prophets 1000 Games.
    You may change the input & output parameters of the function as you wish.
    """
    ############### YOUR CODE GOES HERE ###############
    with open("scenario_three_and_four_prophets.pkl", 'rb') as f:
        data = pickle.load(f)
    algo(data, 1000, True)


def Scenario_5():
    """
    Question 5.
    School of Prophets.
    You may change the input & output parameters of the function as you wish.
    """
    ############### YOUR CODE GOES HERE ###############
    with open('scenario_five_prophets.pkl', 'rb') as f:
        data = pickle.load(f)

    test_set = data['test_set']
    train_set = data['train_set']
    results = []
    for k in [2, 5, 10, 50]:


        for m in [1, 10, 50, 1000]:
            random_indices_true_risk = np.random.choice(len(data['true_risk']), size=k, replace=False)
            data_true_risk = data['true_risk'][random_indices_true_risk]

            approximation_error = np.min(data_true_risk)

            approximation_errors = []
            estimation_errors = []
            test_errors = []

            for _ in range(100):
                # one_train_set = np.random.choice(train_set[0], size=size,
                #                                  replace=False)  ## choisis que la premiere ligne verifier ca $$$

                chosen_prophet = calculate_erm_multiple(train_set, data_true_risk, m)

                # Étape 2 : Évaluation sur l'ensemble de test
                test_predictions = simulate_prediction(test_set[chosen_prophet], data_true_risk[chosen_prophet])

                test_error = np.mean(test_predictions != test_set[chosen_prophet])
                # print("prophete avec true risk ",data['true_risk'][chosen_prophet], "moyenne", test_error)
                # estimation error is the difference between the true risk of the best and selected prophet
                true_error = data_true_risk[chosen_prophet]

                estimation_error = abs(approximation_error - true_error)

                # approximation error is the true risk of the best available prophet

                # Ajouter les erreurs dans les listes
                estimation_errors.append(estimation_error)
                approximation_errors.append(approximation_error)
                test_errors.append(test_error)

            avg_estimation_error = np.mean(estimation_errors)
            avg_approximation_error = np.mean(approximation_errors)
            avg_test_error = np.mean(test_errors)

            results.append({
                'k': k,
                'm': m,
                'avg_approximation_error': avg_approximation_error,
                'avg_estimation_error': avg_estimation_error,
                'avg_test_error': avg_test_error
            })

    # Création d'un DataFrame Pandas pour afficher les résultats

    # Affichage du tableau
    print_results(results)


def Scenario_6():
    """
    Question 6.
    The Bias-Variance Tradeoff.
    You may change the input & output parameters of the function as you wish.
    """
    ############### YOUR CODE GOES HERE ###############
    with open('scenario_six_prophets.pkl', 'rb') as f:
        data = pickle.load(f)

    data_1 = data['hypothesis1']
    data_2 = data['hypothesis2']
    print("Hypothesis 1:")
    avg_estimation_error_1, avg_approximation_error_1, avg_test_error_1,estimation_errors_1 = algo(data_1,10)

    print("\nHypothesis 2:")
    avg_estimation_error_2, avg_approximation_error_2, avg_test_error_2,estimation_errors_2 = algo(data_2,10)

    # Comparer les erreurs
    plt.figure(figsize=(8, 6))

    # Histogramme pour Hypothesis 1 (bleu)
    plt.hist(estimation_errors_1, bins=20, range=(0, 0.5), alpha=0.7, color='blue', label='Hypothesis 1')

    # Histogramme pour Hypothesis 2 (jaune)
    plt.hist(estimation_errors_2, bins=20, range=(0, 0.5), alpha=0.7, color='yellow', label='Hypothesis 2')

    # Titre et labels
    plt.title('Estimation Error Comparison - Hypothesis 1 vs Hypothesis 2')
    plt.xlabel('Estimation Error')
    plt.ylabel('Frequency')

    # Légende pour indiquer les couleurs
    plt.legend()

    # Afficher le graphique
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    print(f'Scenario 1 Results:')
    Scenario_1()

    print(f'Scenario 2 Results:')
    Scenario_2()

    print(f'Scenario 3 Results:')
    Scenario_3()

    print(f'Scenario 4 Results:')
    Scenario_4()

    print(f'Scenario 5 Results:')
    Scenario_5()

    print(f'Scenario 6 Results:')
    Scenario_6()
