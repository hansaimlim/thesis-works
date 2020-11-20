library(stringr)
library(protr)
library(rlist)
convertTriplets<-function(filepath){
  seqlist=list()
  con = file(filepath, "r")
  while ( TRUE ) {
    line = readLines(con, n = 1)
    if ( length(line) == 0 ) {
      break
    }
    if (grepl("PAD",line)){
      next
    }
    if (grepl("GAP",line)){
      next
    }
    s<-toupper(str_replace_all(line,"\\W",""))
    if (grepl("O",s)){
      next
    }
    if (grepl("B",s)){
      next
    }
    if (grepl("U",s)){
      next
    }
    if (grepl("X",s)){
      next
    }
    if (grepl("Z",s)){
      next
    }
    seqlist<-list.append(seqlist,s)
  }
  
  close(con)
  return(seqlist)
}

saveSimMat<-function(simmat,seqlist,filepath) {
  simmat<-as.data.frame(simmat)
  colnames(simmat)<-seqlist
  row.names(simmat)<-seqlist
  write.table(x=simmat,file=filepath,
              col.names=TRUE,row.names=TRUE,quote=FALSE,sep="\t")
}

seqlist<-convertTriplets("./pfam_vocab_triplets_noSpecial.txt")
saveSimMat(parSeqSim(seqlist, cores = 6, type = "local", submat = "BLOSUM45"),
           seqlist,"blo45_local_simmat.txt")
saveSimMat(parSeqSim(seqlist, cores = 6, type = "global", submat = "BLOSUM45"),
           seqlist,"blo45_global_simmat.txt")
saveSimMat(parSeqSim(seqlist, cores = 6, type = "local", submat = "BLOSUM62"),
           seqlist,"blo62_local_simmat.txt")
saveSimMat(parSeqSim(seqlist, cores = 6, type = "global", submat = "BLOSUM62"),
           seqlist,"blo62_global_simmat.txt")
saveSimMat(parSeqSim(seqlist, cores = 6, type = "local", submat = "BLOSUM80"),
           seqlist,"blo80_local_simmat.txt")
saveSimMat(parSeqSim(seqlist, cores = 6, type = "global", submat = "BLOSUM80"),
           seqlist,"blo80_global_simmat.txt")

