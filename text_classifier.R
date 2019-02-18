# Load libraries
library(tm)
library(wordcloud)
library(kernlab)
library(plyr)
library(class)
library(SnowballC)

# Set options
options(stringsAsFactors = FALSE)

# Set parameters
candidates <- c("romney","obama")
pathname <- "~/Dataset"

# Function to convert pretty apostrophe
convertPrettyApostrophe <- function(x) gsub("’", "'", x)

# Function to clean Corpus text
cleanCorpus <- function(corpus) {
  
  # Apply Text Mining Clean-up Functions
  corpus.tmp <- tm_map(corpus, content_transformer(convertPrettyApostrophe))
  corpus.tmp <- tm_map(corpus.tmp, removePunctuation)
  corpus.tmp <- tm_map(corpus.tmp, stripWhitespace)
  corpus.tmp <- tm_map(corpus.tmp, content_transformer(tolower))
  corpus.tmp <- tm_map(corpus.tmp, removeWords, stopwords("english"))
  # corpus.tmp <- tm_map(corpus.tmp, stemDocument, language = "english")
  
  return(corpus.tmp)
}

# Function to generate term document matrices
generateTDM <- function(cand, path) {
  # Set directory
  s.dir <- sprintf("%s/%s", path, cand)
  
  # Instantiate Corpus
  s.cor <- Corpus(DirSource(directory = s.dir, encoding = "UTF-8"))
  
  # Clean corpus
  s.cor.cl <- cleanCorpus(s.cor)
  # Create term document matrix
  s.tdm <- TermDocumentMatrix(s.cor.cl)
  # Remove sparse terms
  s.tdm <- removeSparseTerms(s.tdm, 0.7)
  
  # Construct return object
  result <- list(name = cand, tdm = s.tdm)
  
  return(result)
}

# Run term document matrix function on all candidates
tdm <- lapply(candidates, generateTDM, path = pathname)
str(tdm)
# Bind Candidate Name to Term Document Matrices
bindCandidateToTDM <- function(tdm) {
  s.mat <- t(data.matrix(tdm[["tdm"]]))
  s.df <- as.data.frame(s.mat, stringsAsfactors = FALSE)
  s.df <- cbind(s.df, rep(tdm[["name"]], nrow(s.df)))
  colnames(s.df)[ncol(s.df)] <- "targetcandidate"
  return(s.df)
}

# Append Candidate Field to TDM
candTDM <- lapply(tdm, bindCandidateToTDM)

# Rbind Candidate TDMs
tdm.stack <- do.call(rbind.fill, candTDM)
tdm.stack[is.na(tdm.stack)] <- 0

# Random sample 70% for training of data mining model; remainder for test
train.idx <- sample(nrow(tdm.stack), ceiling(nrow(tdm.stack) * .70))
test.idx <- (1:nrow(tdm.stack))[- train.idx]

# Extract candidate name
tdm.cand <- tdm.stack[, "targetcandidate"]
tdm.stack.nl <- tdm.stack[,!colnames(tdm.stack) %in% "targetcandidate"]

# K-nearest Neighbor
knn.pred <- knn(tdm.stack.nl[train.idx, ], tdm.stack.nl[test.idx, ], tdm.cand[train.idx])
knn.train.data <- tdm.stack[train.idx, ]

# Confusion Matrix
conf.mat <- table("Predictions" = knn.pred, Actual = tdm.cand[test.idx])
conf.mat
# Accuracy
(accuracy <- sum(diag(conf.mat))/length(test.idx) * 100)

################################################################################

# Function to generate corpus where each paragraph is
# set as a document.
generateParagraphDocCorpus <- function(cand, path) {
  
  # Set directory and list files
  s.dir <- sprintf("%s/%s", path, cand)
  filelist <- list.files(s.dir, full.names = TRUE)
  
  # Read each paragraph and append to vector
  speech.v <- unlist(sapply(filelist, function(x) {
    speech.tmp <- readLines(x)
    speech.tmp <- speech.tmp[speech.tmp != ""]
    return(speech.tmp)
  }))
  
  # Instantiate Corpus
  s.cor <- Corpus(VectorSource(speech.v, encoding = "ANSI"))
  
  return(s.cor)
}

# Function to generate corpus from a single file
generateSpeechDocCorpus <- function(filepath) {
  
  # Read data from file
  vec <- scan(filepath, what = "", quiet = TRUE)
  
  # Collapse word vector
  vec <- paste(vec, collapse = " ")
  
  # Instantiate Corpus
  s.cor <- Corpus(VectorSource(vec, encoding = "ANSI"))
  
  return(s.cor)
}    

################################################################################

# Create Dendrograms of Policy Topics
generateDendrogram <- function(tdm, sparcity = 0.2) {
  tdm.sub <- removeSparseTerms(tdm, sparcity)
  euc.dist <- dist(tdm.sub, method = "euclidean")
  dendro <- hclust(euc.dist, method = "ward")
  plot(dendro)
}

# Generate Concept Dendrograms
generateDendrogram(tdm[[1]][[2]], 0.2)
generateDendrogram(tdm[[2]][[2]], 0.3)
generateDendrogram(tdm[[2]][[2]], 0.9)


# Find Concept Associations
findAssocs(tdm[[1]][[2]], 'obama', 0.55)
findAssocs(tdm[[2]][[2]], 'obama', 0.85)
