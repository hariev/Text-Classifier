library(RWeka)
require(RWeka)
library("devtools")
require(IDPmisc)
library(cluster)
library(HSAUR)
library(wordcloud)
library(ggplot2)
library("tm")

data <- read.csv("~/Dataset/spam.csv")
data <- as.data.frame(data)
corpus <- Corpus(VectorSource(data$text))
corpus
corpus <- Corpus(VectorSource(data$text))
cleanset <- tm_map(corpus, removeWords,stopwords("english"))
cleanset <- tm_map(cleanset, stripWhitespace)
dtm <- DocumentTermMatrix(cleanset)

rowTotals <- apply(dtm , 1, sum) #Find the sum of words in each Document
dtm.new   <- dtm[rowTotals> 0, ] #remove all docs without words

#Frequency#
freq <- colSums(as.matrix(dtm))
length(freq)
freq

#By ordering the frequencies we can list the most frequent terms and the least frequent terms:
ord <- order(freq)
ord

#Frequency Ordering#
least <- freq[head(ord)] # Least frequent terms
most <- freq[tail(ord)] # Most frequent terms
plot(most)

#frequency of frequencies
head(table(freq), 15) 
tail(table(freq), 15)

#remove sparse terms
inspect(removeSparseTerms(dtm, 0.2))
dtms <- removeSparseTerms(dtm, 0.2)
dim(dtms)
inspect(dtms)

#find frequent items and associations
findFreqTerms(dtm, lowfreq=14)

#find associations with a specific word, specifying a correlation limit 
#If two words always appear together then the correlation would be 1.0 and if they never appear
#together the correlation would be 0.0.

findAssocs(dtm, "support", corlimit=0.6)

#plot correlations

plot(dtm, terms=findFreqTerms(dtm, lowfreq=14)[1:14],corThreshold=0.6)

#plotting word frequencies
freq <- sort(colSums(as.matrix(dtm)), decreasing=TRUE)
head(freq, 20)

wf <- data.frame(word=names(freq), freq=freq)
head(wf)

#plot the frequency of the words that show up at least n times
p <- ggplot(subset(wf, freq>25), aes(word, freq))
p <- p + geom_bar(stat="identity")
p <- p + theme(axis.text.x=element_text(angle=45, hjust=1))
ggplotly(p)

#visualize wordcloud by setting max words
set.seed(142)
wordcloud(names(freq), freq, max.words=100)

#let's add some color and adjust text size for word frequency
set.seed(142)
wordcloud(names(freq), freq, max.words=20, scale=c(5, .1), colors=brewer.pal(6, "Dark2"))
names(dtm.new)


##########################################################

# Trigrams

TrigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 3, max = 3))
tdm <- TermDocumentMatrix(dtm, control = list(tokenize = TrigramTokenizer))
tdm <- removeSparseTerms(tdm, 0.75)
inspect(tdm)
#write to csv and save in directory
m <- as.matrix(tdm)
dim(m)
write.csv(m, file="tdm_3.csv")

# CLUSTERING #

dtm_tfxidf <- weightTfIdf(dtm.new)
summary(dtm_tfxidf)
m <- as.matrix(dtm_tfxidf)
class(m)
rownames(m) <- 1:nrow(m)
norm_eucl <- function(m)
  m/apply(m,1,function(x) sum(x^2)^.5)
m_norm <- norm_eucl(m)
m_norm <-NaRV.omit(m)
results <- kmeans(m_norm,12,30)

##########################################################