# Assumes that the fit modFitRF exists...

pml_write_files = function(x) {
  n = length(x)
  for(i in 1:n) {
    filename = paste0("submission/problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

if (!file.exists("submission")) {
  message("Making submission folder")
  dir.create("submission")
}

submission <- read.csv(testingFile, na.strings=c("NA", "#DIV/0!"))
answers <- predict(modFitRF, submission, type="response")
pml_write_files(answers)
