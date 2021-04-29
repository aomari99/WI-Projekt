library(Boruta)
read_file <- read.csv('F:\\Projekte\\Pythonscripts\\allwithwaether.csv')
set.seed(8000)
boruta.train <- Boruta(anzahl~., data = read_file, doTrace = 10) 
plot(boruta.train)

