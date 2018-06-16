from Controller import Controller
from tkinter import *
from sys import exit


if __name__ == "__main__":
    # paths
    trainSetPath = "d:\\TreburiSocoteli\\MovieRecommenderSystem\\training\\sets\\"
    dataset = "d:\\TreburiSocoteli\\MovieRecommenderSystem\\data\\collaborative\\collaborativeRatings.csv"
    ratingsFilename = "d:\\TreburiSocoteli\\MovieRecommenderSystem\\data\\content\\contentRatings.csv"
    tagsFilename = "d:\\TreburiSocoteli\\MovieRecommenderSystem\\data\\content\\contentTags.csv"
    moviesFilename = "d:\\TreburiSocoteli\\MovieRecommenderSystem\\data\\content\\contentMovies.csv"

    # defaults
    defaultMethod = "Hybrid"
    defaultNoMovies = 6
    defaultMovieId = 722
    defaultUserId = 512

    # controller
    controller = Controller(trainSetPath, dataset, ratingsFilename, tagsFilename, moviesFilename)


    # User Interface
    master = Tk()
    leftFrame = Frame(master)
    leftFrame.pack(side=LEFT)
    middleFrame = Frame(master)
    middleFrame.pack(side=LEFT)
    methodFrame = Frame(middleFrame)
    methodFrame.pack(side=TOP)
    buttonsFrame = Frame(middleFrame)
    buttonsFrame.pack(side=TOP)
    rightFrame = Frame(master)
    rightFrame.pack(side=LEFT)

    movies = controller.getMovies()
    allMoviesListBox = Listbox(leftFrame, width=60, height=40)
    for key in movies.keys():
        # for now, key in list has to be movieId
        allMoviesListBox.insert(key, movies[key])

    # method
    methodVar = StringVar(None)
    methodVar.set(str(defaultMethod))
    colRadioButton = Radiobutton(methodFrame, text="Collaborative", variable=methodVar, value="Collaborative")
    colRadioButton.grid(row=1,column=1)
    conRadioButton = Radiobutton(methodFrame, text="Content", variable=methodVar, value="Content")
    conRadioButton.grid(row=2, column=1)
    hybRadioButton = Radiobutton(methodFrame, text="Hybrid", variable=methodVar, value="Hybrid")
    hybRadioButton.grid(row=3, column=1)

    # number of movies entry
    noMoviesLabelText = StringVar()
    noMoviesLabelText.set("number of movies: ")
    noMoviesLabel = Label(buttonsFrame, textvariable=noMoviesLabelText, height=6)
    noMoviesLabel.grid(row=1,column=1)
    noMoviesEntryText = StringVar(None)
    noMoviesEntry = Entry(buttonsFrame, textvariable=noMoviesEntryText, width=6)
    noMoviesEntry.insert(END, str(defaultNoMovies))
    noMoviesEntry.grid(row=1,column=2)

    # movie id entry
    movieIdLabelText = StringVar()
    movieIdLabelText.set("movie id: ")
    movieIdLabel = Label(buttonsFrame, textvariable=movieIdLabelText, height=6)
    movieIdLabel.grid(row=2,column=1)
    movieIdEntryText = StringVar(None)
    movieIdEntry = Entry(buttonsFrame, textvariable=movieIdEntryText, width=6)
    movieIdEntry.insert(END, str(defaultMovieId))
    movieIdEntry.grid(row=2,column=2)

    # user id entry
    userIdLabelText = StringVar()
    userIdLabelText.set("user id: ")
    userIdLabel = Label(buttonsFrame, textvariable=userIdLabelText, height=6)
    userIdLabel.grid(row=3,column=1)
    userIdEntryText = StringVar(None)
    userIdEntry = Entry(buttonsFrame, textvariable=userIdEntryText, width=6)
    userIdEntry.insert(END, str(defaultUserId))
    userIdEntry.grid(row=3,column=2)

    def onSelect(evt):
        w = evt.widget
        index = int(w.curselection()[0])
        value = w.get(index)
        print(value, " has been selected")
        movieIdEntry.delete(0, END)
        movieIdEntry.insert(END, str(index))

    allMoviesListBox.grid(row=1, column=1)
    allMoviesListBox.bind('<<ListboxSelect>>', onSelect)

    similarMoviesListBox = Listbox(rightFrame, width=60, height=30)
    endExecutionButton = Button(buttonsFrame, text='Exit')
    getSimilarMoviesButton = Button(buttonsFrame, text='Get similar movies')
    getUserMoviesButton = Button(buttonsFrame, text='Get user movies')

    def endExecution(event):
        print("End execution")
        exit()

    def getSimilarMovies(event):
        method = methodVar.get()
        keyMovieId = int(movieIdEntry.get())
        noMovies = int(noMoviesEntry.get())

        allMoviesListBox.see(keyMovieId)
        similarMoviesListBox.delete(0, END)
        idx = 0
        similarMoviesListBox.insert(idx, "SELECTED MOVIE: " + str(movies[keyMovieId]))
        idx += 1
        similarMoviesListBox.insert(idx, " ")
        recommendedMovieIds = controller.getSimilarMovies(method, keyMovieId, noMovies)
        for movieId in recommendedMovieIds:
            idx += 1
            similarMoviesListBox.insert(idx, str(movies[movieId]))

        similarMoviesListBox.grid(row=1,column=1)
        print("getSimilarMovies  ", method, "  ", movies[keyMovieId])

    def getUserMovies(event):
        method = methodVar.get()
        userId = int(userIdEntry.get())
        noMovies = int(noMoviesEntry.get())

        similarMoviesListBox.delete(0, END)
        idx = 0
        similarMoviesListBox.insert(idx, "SELECTED USER ID: " + str(userId))
        idx += 1
        similarMoviesListBox.insert(idx, " ")
        recommendedMovieIds = controller.getUserMovies(method, userId, noMovies)
        for movieId, rating in recommendedMovieIds:
            idx += 1
            similarMoviesListBox.insert(idx, str(movies[movieId]) + "  " + str(round(rating, 2)))

        similarMoviesListBox.insert(1, userIdEntry.get())
        similarMoviesListBox.grid(row=1, column=1)
        print("getUserMovies  ", method, "  userID:", userId)

    endExecutionButton.grid(row=1, column=3)
    endExecutionButton.bind('<Button-1>', endExecution)
    getSimilarMoviesButton.grid(row=2,column=3)
    getSimilarMoviesButton.bind('<Button-1>', getSimilarMovies)
    getUserMoviesButton.grid(row=3,column=3)
    getUserMoviesButton.bind('<Button-1>', getUserMovies)

    master.mainloop()
