library(dplyr)
setwd("./") #change to current directory
assays<-read.table("CYP450_Assays.tsv",header=TRUE, sep="\t",quote="")
assays<-assays[!(assays$ActivityUnit=="None"),]
Bassays<-assays[assays$AssayType=="Binding",]
Fassays<-assays[assays$AssayType=="Functional",]
ADMEassays<-assays[assays$AssayType=="ADME",]

#check frequency of units for Potency measurement
as.data.frame(table(assays[assays$ActivityType=="Potency",]$ActivityUnit))
