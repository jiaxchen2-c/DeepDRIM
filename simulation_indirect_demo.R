



four_genes_theta_cov=function(fileHeader, p=4,sample_n=1000, mu_zero=F)
{
  library(MASS)
  mat=matrix(0,p,p)
  
  edge_a_b=runif(1, min=0.25,max=1)*sample(c(1,-1),1)
  if(runif(1,min=0,max=100)>50)
  {
    edge_b_c=runif(1, min=0.25, max=1)*sample(c(1,-1),1)
  }else{
    edge_b_c=0
  }
  if(runif(1,min=0,max=100)>50)
  {
    edge_a_c=runif(1, min=0.25, max=1)*sample(c(1,-1),1)
  }else{
    edge_a_c=0
  }
  if(runif(1,min=0,max=100)>50)
  {
    edge_a_d=runif(1, min=0.25, max=1)*sample(c(1,-1),1)
  }else{
    edge_a_d=0
  }
  if(runif(1,min=0,max=100)>50)
  {
    edge_b_d=runif(1, min=0.25, max=1)*sample(c(1,-1),1)
  }else{
    edge_b_d=0
  }
  if(runif(1,min=0,max=100)>50)
  {
    edge_c_d=runif(1, min=0.25, max=1)*sample(c(1,-1),1)
  }else{
    edge_c_d=0
  }
  
  
  
  
  mat[1,2]=edge_a_b
  mat[2,1]=edge_a_b
  mat[2,3]=edge_b_c
  mat[3,2]=edge_b_c
  mat[1,3]=edge_a_c
  mat[3,1]=edge_a_c
  
  mat[1,4]=edge_a_d
  mat[4,1]=edge_a_d
  mat[2,4]=edge_b_d
  mat[4,2]=edge_b_d
  mat[3,4]=edge_c_d
  mat[4,3]=edge_c_d
  
  
  
  theta=mat
  
  ee <- min(eigen(theta,only.values=T)$values)
  diag(theta) <- ifelse(ee < 0, -ee + 0.1, 0.1)
  
  cov=ginv(theta)
  
  print(theta)
  print(cov)
  
  write.table(theta,file =paste(fileHeader,"_theta.csv",sep=""),quote=F,sep=",",col.names = F,row.names = F)
  
  write.table(cov,file =paste(fileHeader,"_cov.csv",sep=""),quote=F,sep=",",col.names = F,row.names = F)
  
  if(mu_zero)
  {
    sim_data=mvrnorm(sample_n,mu=rep(0,p),cov,tol = 1e-6)
  }else{
    mu_vector=runif(p, 10, 10000)
    sim_data=mvrnorm(sample_n,mu=mu_vector,cov,tol = 1e-6)
    sim_data=abs(sim_data)
  }
  
  write.table(sim_data,file=paste(fileHeader,"_sim_data.csv",sep=''),quote=F,sep=",")
  
}




more_genes_theta_cov=function(fileHeader,p=10, sparsity=0.7,mu_zero=F,sample_n=1000)
{
  sparse.base<- rbinom(p*p,1,1-sparsity) * sample(c(-1,1), p*p,replace=TRUE)*runif(p*p,0.25,0.75)
  
  Theta=matrix(data=sparse.base,p,p)
  Theta[lower.tri(Theta,diag=FALSE)] <- 0
  Theta <- Theta+t(Theta)

  Theta=ifelse(abs(Theta)<1e-5,0,Theta)
  diag(Theta) <- 0
  ee <- min(eigen(Theta,only.values=T)$values)
  diag(Theta) <- ifelse(ee < 0, -ee + 0.1, 0.1)
  
  
  library(MASS)
  cov = ginv(Theta)
  
  write.table(Theta,file =paste(fileHeader,"_theta.csv",sep=""),quote=F,sep=",",col.names = F,row.names = F)
  
  write.table(cov,file =paste(fileHeader,"_cov.csv",sep=""),quote=F,sep=",",col.names = F,row.names = F)
  
  if(mu_zero)
  {
    sim_data=mvrnorm(sample_n,mu=rep(0,p),cov,tol = 1e-6)
  }else{
    mu_vector=runif(p, 10, 10000)
    sim_data=mvrnorm(sample_n,mu=mu_vector,cov,tol = 1e-6)
    sim_data=abs(sim_data)
  }
  
  write.table(sim_data,file=paste(fileHeader,"_sim_data.csv",sep=''),quote=F,sep=",")
  
}






#####generate 4 node indirect...
output_dir=paste("simulation_node4_indirect_reshow/",sep="")
dir.create(output_dir)

for(i in c(1:2500))
{
  
  fileHeader=paste(output_dir,i,sep="")
  four_genes_theta_cov(fileHeader)
  
  
}








