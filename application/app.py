from UI import UI



if __name__ == "__main__":
    # paths
    trainSetPath = "training\\sets\\"
    dataset = "data\\collaborative\\collaborativeRatings.csv"
    ratingsFilename = "data\\content\\contentRatings.csv"
    tagsFilename = "data\\content\\contentTags.csv"
    moviesFilename = "data\\content\\contentMovies.csv"

    app = UI(trainSetPath, dataset, ratingsFilename, tagsFilename, moviesFilename)
    app.run()

