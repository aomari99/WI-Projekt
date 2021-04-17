library(Boruta)
read_file <- read.csv('F:\\Projekte\\Pythonscripts\\weekcount.csv')
set.seed(8000)
boruta.train <- Boruta(anzahl~., data = read_file, doTrace = 2) 
plot(boruta.train)

