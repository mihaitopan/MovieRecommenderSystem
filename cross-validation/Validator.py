from CrossCollaborativeFiltering import CrossCollaborativeFiltering
from CrossContentFiltering import CrossContentFiltering
from sys import exit


if __name__ == "__main__":
    '''default constants'''
    userInput = ""
    command = "collaborative"
    noCrossValidationSets = 5


    '''read noCrossValidationSets'''
    try:
        print("number of cross validation sets should be more that 3 and less than 11")
        userInput = input("number of cross validation sets: ")
        noCrossValidationSets = int(userInput)
        if noCrossValidationSets < 3 or noCrossValidationSets > 11:
            raise ValueError
    except ValueError as _:
        print("invalid parameter ", userInput)
        exit()


    '''read command'''
    info = "possible commands: \n"
    # info += "collaborative - collaborative algorithm - collaborative dataset"
    info += "\t doCollaborative - collaborative algorithm \n"# - content dataset"
    info += "\t doContent - content algorithm \n"# - content dataset"
    print(info)
    try:
        command = input("command: ")
        if command != "collaborative" and command != "doCollaborative" and command != "doContent":
            raise ValueError
    except ValueError as _:
        print("invalid parameter ", command)
        exit()


    '''what to validate?'''
    if command == "collaborative":
        print("run: collaborative algorithm - collaborative dataset")
        dataset = "data\\collaborative\\collaborativeRatings.csv"
        trainSetPath = "cross-validation\\sets\\"

        crossValidator = CrossCollaborativeFiltering(dataset, trainSetPath, noCrossValidationSets)
        crossValidator.train()
        mae, rmsd, accuracy = crossValidator.test()
        print("mae: ", mae)
        print("rmsd: ", rmsd)
        print("accuracy: ", accuracy)

    elif command == "doCollaborative":
        print("run: collaborative algorithm - content dataset")
        dataset = "data\\content\\contentRatings.csv"
        trainSetPath = "cross-validation\\sets\\"

        crossValidator = CrossCollaborativeFiltering(dataset, trainSetPath, noCrossValidationSets)
        crossValidator.train()
        mae, rmsd, accuracy = crossValidator.test()
        print("mae: ", mae)
        print("rmsd: ", rmsd)
        print("accuracy: ", accuracy)

    elif command == "doContent":
        print("run: content algorithm - content dataset")
        ratingsFilename = "data\\content\\contentRatings.csv"
        tagsFilename = "data\\content\\contentTags.csv"
        trainSetPath = "cross-validation\\sets\\"

        crossValidator = CrossContentFiltering(ratingsFilename, tagsFilename, trainSetPath, noCrossValidationSets)
        crossValidator.train()
        mae, rmsd, accuracy = crossValidator.test()
        print("mae: ", mae)
        print("rmsd: ", rmsd)
        print("accuracy: ", accuracy)

    else:
        print("invalid parameter ", command)
