from CrossCollaborativeFiltering import CrossCollaborativeFiltering
from CrossContentFiltering import CrossContentFiltering


if __name__ == "__main__":
    # constants
    command = "ColCon"
    noCrossValidationSets = 5

    # what to validate?
    if command == "ColCol":
        print("run: collaborative algorithm - collaborative dataset")
        dataset = "d:\\TreburiSocoteli\\MovieRecommenderSystem\\data\\collaborative\\collaborativeRatings.csv"
        trainSetPath = "d:\\TreburiSocoteli\\MovieRecommenderSystem\\cross-validation\\sets\\"

        crossValidator = CrossCollaborativeFiltering(dataset, trainSetPath, noCrossValidationSets)
        #crossValidator.train()
        mae, rmsd, accuracy = crossValidator.test()
        print("mae: ", mae)
        print("rmsd: ", rmsd)
        print("accuracy: ", accuracy)

    elif command == "ColCon":
        print("run: collaborative algorithm - content dataset")
        dataset = "d:\\TreburiSocoteli\\MovieRecommenderSystem\\data\\content\\contentRatings.csv"
        trainSetPath = "d:\\TreburiSocoteli\\MovieRecommenderSystem\\cross-validation\\sets\\"

        crossValidator = CrossCollaborativeFiltering(dataset, trainSetPath, noCrossValidationSets)
        #crossValidator.train()
        mae, rmsd, accuracy = crossValidator.test()
        print("mae: ", mae)
        print("rmsd: ", rmsd)
        print("accuracy: ", accuracy)

    elif command == "ConCon":
        print("run: content algorithm - content dataset")
        ratingsFilename = "d:\\TreburiSocoteli\\MovieRecommenderSystem\\data\\content\\contentRatings.csv"
        tagsFilename = "d:\\TreburiSocoteli\\MovieRecommenderSystem\\data\\content\\contentTags.csv"
        trainSetPath = "d:\\TreburiSocoteli\\MovieRecommenderSystem\\cross-validation\\sets\\"

        crossValidator = CrossContentFiltering(ratingsFilename, tagsFilename, trainSetPath, noCrossValidationSets)
        #crossValidator.train()
        mae, rmsd, accuracy = crossValidator.test()
        print("mae: ", mae)
        print("rmsd: ", rmsd)
        print("accuracy: ", accuracy)

    else:
        print("bad command")
