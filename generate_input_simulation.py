import pandas as pd
from numpy import *
import numpy as np
import json, re,os, sys


###output
##x, y, z fit the model(single stage), and stored in batch



class ExprCollection:

    def __init__(self,output_dir,x_method_version=1, max_col=None, pair_in_batch_num=None,getlog=False, node_num=10,divide_part=10,pair_in_each_net=2):
        # input
        self.expr = None  # a table
        self.geneIDs = None  #
        self.geneID_to_index_in_expr = {}
        self.networki_geneID_to_expr = {}
        self.networki_genepair_to_cov = {}
        self.networki_to_corr_mat = {}
        self.networki_to_cov_mat = {}
        self.networki_genepair_to_corr = {}
        self.sampleIDs = None  # not necessary
        self.geneID_map = None  # not necessary, ID in expr to ID in gold standard
        self.gold_standard = {}  # geneA;geneB -> 0,1,2 #note direction, geneA,geneB is diff with geneB,geneA
        self.output_dir = output_dir
        self.start_batch_num = 0
        self.x_method_version = x_method_version
        self.pair_in_batch_num = pair_in_batch_num
        self.getlog=getlog
        self.node_num=node_num
        self.divide_part=divide_part
        if max_col is None:
            #self.max_col = 2 * len(self.geneIDs)
            self.max_col = math.ceil(sqrt(2 * node_num))
            print("max_col=",self.max_col)
        else:
            self.max_col = max_col
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)
            os.mkdir(self.output_dir + "version0/")
            os.mkdir(self.output_dir + "version1/")
            os.mkdir(self.output_dir + "version11/")

        ##setting for one pair
        self.flag_removeNoise = None
        self.top_num = None
        self.top_or_random_or_all = None
        self.flag_ij_repeat = None
        self.flag_ij_compress = None
        self.cat_option = None  # flat, or multiple channel "multi_channel"
        self.flag_multiply_weight = None

        self.hub_TF = {}
        self.pair_in_each_net = pair_in_each_net

    def load_abundance_file(self, file_path, networki=0):
        if file_path is not None:
            if os.path.isfile(file_path):
                df = pd.read_csv(file_path, header='infer',index_col=0)
                if file_path.endswith(".txt"):
                    df = pd.read_csv(file_path, header='infer', index_col=0, delimiter="\t")
                m1 = df.values
                self.sampleIDs = df.index
                self.geneIDs = df.columns.values
                expr = np.transpose(m1)
                for i in range(0,len(self.geneIDs)):
                    self.geneID_to_index_in_expr[self.geneIDs[i]] = i
                    networki_geneID = str(networki)+":"+self.geneIDs[i]
                    self.networki_geneID_to_expr[networki_geneID] = expr[i,:]
                return expr
            else:
                print("Do not exist file " + file_path + " ,please check")


    def load_adj_matrix(self, file_path):
        if file_path is not None:
            if os.path.isfile(file_path):
                df = pd.read_csv(file_path, header=None)
                theta = df.values
                return theta
            else:
                print("Do not exist file " + file_path + " ,please check")

    def load_data_expr_answer_adj(self,rpkm_file, gold_standard_file, covariance_file=None, networki=0):
        #for simulation data, gold_standard_file is theta #need some transformation to gold_standard
        #initialize all
        expr = self.load_abundance_file(rpkm_file,networki)
        theta = self.load_adj_matrix(gold_standard_file)
        cov = self.load_adj_matrix(covariance_file)
        corr_mat = np.corrcoef(expr)
        print("dim corr_mat",shape(corr_mat))
        #self.networki_to_corr_mat[networki] = corr_mat
        #self.networki_to_cov_mat[networki] = cov
        for i in range(0,shape(cov)[0]):
            for j in range(0,shape(cov)[1]):
                keyij=str(networki)+":"+self.geneIDs[i]+","+str(networki)+":"+self.geneIDs[j]
                self.networki_genepair_to_cov[keyij] = cov[i,j]
                self.networki_genepair_to_corr[keyij] = corr_mat[i,j]

        A=np.nonzero(theta)
        geneA_list = A[0]
        geneB_list = A[1]

        B = np.where(theta == 0)
        random_B = np.random.randint(len(B[0]), size=(1, len(A[0])))
        geneC_list = B[0][random_B][0]
        geneD_list = B[1][random_B][0]
        for i in range(0,len(geneA_list)):
            if int(geneA_list[i]) < int(geneB_list[i]): ###modify if require direction
                key = str(networki)+":"+self.geneIDs[geneA_list[i]]+','+str(networki)+":"+self.geneIDs[geneB_list[i]]
                self.gold_standard[key] = int(1)
                #key2 = str(self.geneIDs[geneB_list[i]])+','+str(self.geneIDs[geneA_list[i]])
                #self.gold_standard[key2] = int(1)
        for k in range(0,len(geneC_list)):
            if int(geneC_list[k]) < int(geneD_list[k]):
                #print(theta[geneC_list[k],geneD_list[k]])
                key = str(networki)+":"+self.geneIDs[geneC_list[k]] + ',' +str(networki)+":" + self.geneIDs[geneD_list[k]]
                self.gold_standard[key] = int(0)
                #key2 = str(self.geneIDs[geneD_list[k]])+','+str(self.geneIDs[geneC_list[k]])
                #self.gold_standard[key2] = int(0)
        print("gold standard", self.gold_standard)

    def load_data_simulation_indirect(self, network_num=1000):
        #initialize gold standard and network_geneID_to_expr
        #generate random simulation indirect network
        #record theta and cov
        for i in range(0,network_num):
            theta_mat = np.zeros((3,3))

            import random

            for j in range(0,3):
                for k in range(0,3):
                    if k > j:
                        rand = random.random()
                        rand = min(rand+0.25,0.75)

                        theta_mat[j,k]=rand
                        theta_mat[k,j]=rand

            if random.random() > 0.5:
                theta_mat[0,2]=0

            w, v= np.linalg.eigh(theta_mat)
            eig=min(w)
            np.fill_diagonal(theta_mat, eig)

            cov_mat = np.linalg.inv(theta_mat)

            #write theta_mat and cov_mat

            #to be continue


    def load_data_expr_answer_indirect(self, rpkm_file, gold_standard_file, covariance_file,networki=0):
        # for simulation data, gold_standard_file is theta #need some transformation to gold_standard
        # initialize all
        expr = self.load_abundance_file(rpkm_file, networki)
        theta = self.load_adj_matrix(gold_standard_file)
        cov = self.load_adj_matrix(covariance_file)
        corr_mat = np.corrcoef(expr)
        print("dim corr_mat", shape(corr_mat))
        #self.networki_to_corr_mat[networki] = corr_mat
        #self.networki_to_cov_mat[networki] = cov
        #print(shape(cov))
        for i in range(0, shape(cov)[0]):
            for j in range(0, shape(cov)[1]):
                keyij = str(networki) + ":" + self.geneIDs[i] + "," + str(networki) + ":" + self.geneIDs[j]
                self.networki_genepair_to_cov[keyij] = cov[i, j]
                self.networki_genepair_to_corr[keyij] = corr_mat[i,j]

        #initialize networki TF hub by top cov sum
        TF_cov_sum=np.zeros((shape(cov)[0]))
        for i in range(0, shape(cov)[0]):
            cov_list_TFi = cov[i, :]
            TF_cov_sum[i] = sum(abs(cov_list_TFi))
        the_order = np.argsort(-TF_cov_sum)
        select_index = the_order[0:self.top_num]
        hub_TF_networki = [self.geneIDs[j] for j in select_index]

        self.hub_TF[str(networki)] = hub_TF_networki

        ###end initial networki TF


        import random
        tmp = random.sample(range(0,self.node_num), 2)

        geneA_list=[]
        geneB_list=[]
        geneC_list=[]
        geneD_list=[]
        geneE_list=[]
        geneF_list=[]

        A = np.nonzero(theta)

        if self.pair_in_each_net==2:
            select_pair=[0]
        elif self.pair_in_each_net is None:
            select_pair=range(0,len(A[0]))

        if len(A[0])>0:
            random_A = np.random.randint(len(A[0]), size=(1, len(A[0])))
            geneA_list = A[0][random_A][0]
            geneB_list = A[1][random_A][0]
            print("geneA_list", geneA_list)
            for i in select_pair:
                geneE_list.append(geneA_list[i])
                geneF_list.append(geneB_list[i])
            print("geneE_list",geneE_list)

        B = np.where(theta == 0)
        if len(B[0])>0:
            random_B = np.random.randint(len(B[0]), size=(1, len(A[0])))
            geneC_list = B[0][random_B][0]
            geneD_list = B[1][random_B][0]

            for i in select_pair:
                geneE_list.append(geneC_list[i])
                geneF_list.append(geneD_list[i])


        for i in range(0, len(geneE_list)):

            key = str(networki) + ":" + self.geneIDs[geneE_list[i]] + ',' + str(networki) + ":" + self.geneIDs[geneF_list[i]]
            if theta[geneE_list[i], geneF_list[i]] != 0:
                self.gold_standard[key] = int(1)
            else:
                self.gold_standard[key] = int(0)
            # key2 = str(self.geneIDs[geneB_list[i]])+','+str(self.geneIDs[geneA_list[i]])
            # self.gold_standard[key2] = int(1)
        #print(self.gold_standard)

    def load_data_expr_answer_cov(self,rpkm_file, gold_standard_file, networki=0):
        #for simulation data, gold_standard_file is theta #need some transformation to gold_standard
        #initialize all
        self.load_abundance_file(rpkm_file)
        cov = self.load_adj_matrix(gold_standard_file)
        abs_cov = abs(cov)
        A=np.where(abs_cov >= 0.25)
        geneA_list = A[0]
        geneB_list = A[1]

        B = np.where(abs_cov < 0.25)
        random_B = np.random.randint(len(B[0]), size=(1, len(A[0])))
        geneC_list = B[0][random_B][0]
        geneD_list = B[1][random_B][0]
        for i in range(0,len(geneA_list)):
            if int(geneA_list[i]) < int(geneB_list[i]): ###modify if require direction
                key = str(networki)+":"+self.geneIDs[geneA_list[i]]+','+ str(networki) + ":" + self.geneIDs[geneB_list[i]]
                self.gold_standard[key] = int(1)
                #key2 = str(self.geneIDs[geneB_list[i]])+','+str(self.geneIDs[geneA_list[i]])
                #self.gold_standard[key2] = int(1)
        for k in range(0,len(geneC_list)):
            if int(geneC_list[k]) < int(geneD_list[k]):
                #print(theta[geneC_list[k],geneD_list[k]])
                key = str(networki)+":"+self.geneIDs[geneC_list[k]] + ',' + str(networki) + ":" + self.geneIDs[geneD_list[k]]
                self.gold_standard[key] = int(0)
                #key2 = str(self.geneIDs[geneD_list[k]])+','+str(self.geneIDs[geneC_list[k]])
                #self.gold_standard[key2] = int(0)
        print("gold standard",self.gold_standard)

    def get_histogram_bins(self,geneA,geneB):
        #print("geneA",geneA,"geneB",geneB)
        #input gene A name, geneB name, expression data,
        x_geneA = self.networki_geneID_to_expr.get(geneA)
        x_geneB = self.networki_geneID_to_expr.get(geneB)
        if self.getlog:
            x_geneA = log10(x_geneA+ 10 ** -2)
            x_geneB = log10(x_geneB+ 10 ** -2)

        n = len(self.sampleIDs) #number of samples
        H_T = np.histogram2d(x_geneA, x_geneB, bins=32)
        H = H_T[0].T
        HT = (np.log10(H / n + 10 ** -4) + 4) / 4 #pesudo count?
        return HT


    def get_x_for_one_pair_version0(self,geneA,geneB):####change here, important.
        #input geneA, geneB, get corresponding expr and get histgram
        #return x, y, z
        x = self.get_histogram_bins(geneA,geneB)
        return x


    def get_x_for_one_pair_version1(self, geneA,geneB):#### change here, important. # need complete
        # for each gene pair, x contain all images in the corresponding rows and columns
        one_row  =None
        rows = None
        index=0
        one_image = self.get_histogram_bins(geneA, geneB)
        one_row=one_image
        index=index+1
        one_image = self.get_histogram_bins(geneB, geneA)
        one_row = np.concatenate((one_row, one_image), axis=1)
        index=index+1
        networki=geneA.split(":")[0]
        for j in range(0, len(self.geneIDs)):
            if str(networki)+":"+self.geneIDs[j] != geneB:
                one_image = self.get_histogram_bins(geneA, str(networki)+":"+self.geneIDs[j])
                if index >= self.max_col:
                    if rows is None:
                        rows = one_row
                    else:
                        rows = np.concatenate((rows,one_row),axis=0)
                    one_row=one_image
                    index=1
                else:
                    one_row = np.concatenate((one_row, one_image), axis=1)
                    index=index+1
        for j in range(0, len(self.geneIDs)):
            if str(networki)+":"+self.geneIDs[j] != geneA:
                one_image = self.get_histogram_bins(str(networki)+":"+self.geneIDs[j], geneB)
                if index >= self.max_col:
                    if rows is None:
                        rows = one_row
                    else:
                        rows = np.concatenate((rows, one_row), axis=0)
                    one_row = one_image
                    index = 1
                else:
                    one_row = np.concatenate((one_row, one_image), axis=1)
                    index = index + 1

        if shape(one_row)[0] > 0:
            if shape(one_row)[1] < (self.max_col * 32):
                print("rest dimension", ((self.max_col * 32)-shape(one_row)[1]))
                rest_image=np.zeros((32,((self.max_col * 32)-shape(one_row)[1])))
                one_row = np.concatenate((one_row,rest_image), axis=1)
                rows = np.concatenate((rows,one_row),axis=0)
            else:
                rows = np.concatenate((rows, one_row), axis=0)
        if rows is None:
            x = one_row
        else:
            x=rows
        #print("x",shape(x))
        return x


    def get_gene_pair_data(self,geneA,geneB,x_method_version):
        # input geneA, geneB, get corresponding expr and get histogram
        # return x, y, z
        if x_method_version==11:
            x = self.get_x_for_one_pair_version11(geneA,geneB)
        elif x_method_version==0:
            x = self.get_x_for_one_pair_version0(geneA,geneB)
        elif x_method_version==1:
            x = self.get_x_for_one_pair_version1(geneA,geneB)
        key = str(geneA)+','+str(geneB)
        y = self.gold_standard.get(key)
        z = key
        # if not os.path.isdir(self.output_dir +str(x_method_version)+"_histogram/"):
        #    os.mkdir(self.output_dir+str(x_method_version) +"_histogram/")
        # np.savetxt(self.output_dir +str(x_method_version)+"_histogram/" + z+'_'+str(y)+'_histogram.csv', x, delimiter=",")

        return [x,y,z]

    def get_batch(self,gene_list,save_header,x_method_version):
        xdata = []  # numpy arrary [k,:,:,1], k is number o fpairs
        ydata = []  # numpy arrary shape k,1
        zdata = []  # record corresponding pairs
        # for each term in list, split it into two
        # call get_gene_pair_data and append together
        for i in range(0,len(gene_list)):
            geneA=gene_list[i].split(',')[0]
            geneB=gene_list[i].split(',')[1]
            [x,y,z] = self.get_gene_pair_data(geneA,geneB,x_method_version)
            xdata.append(x)
            ydata.append(y)
            zdata.append(z)

        print("xdata",shape(xdata))

        xx=xdata
        if (len(xdata) > 0):
            if len(shape(xdata))==4:
                #xx = np.array(xdata)[:, :, :, :, np.newaxis]
                xx = xdata
            if len(shape(xdata))==3:
                xx = np.array(xdata)[:, :, :, np.newaxis]
        print("xx",shape(xx))

        np.save(save_header+'_xdata.npy',xx)
        np.save(save_header + '_ydata.npy', np.array(ydata))
        np.save(save_header + '_zdata.npy', np.array(zdata))
        #save as npy
        pass

    def get_train_test(self,batch_index=None,generate_multi=True):
        #deal with cross validation or train test batch partition,mini_batch
        key_list = list(self.gold_standard.keys())
        #random_n = list(np.random.randint(len(key_list), size=(10)))
        from random import shuffle
        #print(key_list)
        shuffle(key_list)
        shuffle(key_list)
        print("len key list",len(key_list))
        #print(key_list)
        index_start = 0

        if self.pair_in_batch_num is None:
            self.pair_in_batch_num = math.floor(len(key_list)/self.divide_part)
        batches = int(round(len(key_list)/self.pair_in_batch_num))

        print(batches)
        tmp = self.start_batch_num
        for i in range(self.start_batch_num,(tmp+batches)):
            print("index_start", index_start)

            index_end = index_start + self.pair_in_batch_num
            print("index_end", index_end)
            if index_end <= len(key_list):
                select_list = list(key_list[j] for j in range(index_start,index_end))

                if generate_multi:
                    self.get_batch(select_list, self.output_dir+"version0/" + str(i),0)
                    self.get_batch(select_list, self.output_dir+"version"+str(self.x_method_version)+"/" + str(i), self.x_method_version)
                    #self.get_batch(select_list, self.output_dir + "version1/" + str(i), 1)
                else:
                    self.get_batch(select_list, self.output_dir+str(i),self.x_method_version)

                index_start = index_end
                self.start_batch_num = i+1


    def get_all_related_pairs(self, geneA, geneB):
        histogram_list = []
        #if from simulation data, we need record network id, and get it by networkid
        networki = geneA.split(":")[0]
        for j in range(0, len(self.geneIDs)):
            if str(networki)+":"+self.geneIDs[j] != geneB:
                x = self.get_histogram_bins(geneA, str(networki)+":"+self.geneIDs[j])
                
                histogram_list.append(x)
        for j in range(0, len(self.geneIDs)):
            if str(networki)+":"+self.geneIDs[j] != geneA:
                x = self.get_histogram_bins(str(networki)+":"+self.geneIDs[j], geneB)
                
                histogram_list.append(x)
        return histogram_list


    def get_top_cov_pairs(self,geneA,geneB,cov_or_corr="corr"):
        # get cov value first
        histogram_list = []
        networki = geneA.split(":")[0]

        x = self.get_histogram_bins(geneA, geneA)
        
        histogram_list.append(x)

        x = self.get_histogram_bins(geneB, geneB)
        
        histogram_list.append(x)

        index = geneA.split(":")[1]
        cov_list_geneA = []
        for j in range(0, len(self.geneIDs)):
            #if self.geneIDs[j] != geneB:
            if cov_or_corr=="cov":
                cov_value = self.networki_genepair_to_cov.get(geneA + "," + str(networki)+":"+self.geneIDs[j])
            else:
                cov_value = self.networki_genepair_to_corr.get(geneA + "," + str(networki)+":"+self.geneIDs[j])
            cov_list_geneA.append(cov_value)
        cov_list_geneA = np.asarray(cov_list_geneA)
        cov_list_geneA = cov_list_geneA.ravel()
        the_order = np.argsort(-cov_list_geneA)
        select_index = the_order[0:self.top_num]
        for j in select_index:
            #if self.ID_to_name_map.get(self.geneIDs[j]) != geneA:
            x = self.get_histogram_bins(geneA, str(networki)+":"+self.geneIDs[j])
            
            histogram_list.append(x)
        ####
        indexB = geneB.split(":")[1]
        cov_list_geneB = []
        for j in range(0, len(self.geneIDs)):
            #if self.geneIDs[j] != geneB:
            if cov_or_corr=="cov":
                cov_value = self.networki_genepair_to_cov.get(geneB + "," + str(networki) + ":" + self.geneIDs[j])
            else:
                cov_value = self.networki_genepair_to_corr.get(geneB + "," + str(networki) + ":" + self.geneIDs[j])
            cov_list_geneB.append(cov_value)
        cov_list_geneB=np.asarray(cov_list_geneB)
        cov_list_geneB = cov_list_geneB.ravel()
        the_order = np.argsort(-cov_list_geneB)
        select_index = the_order[0:self.top_num]
        for j in select_index:
            #if self.ID_to_name_map.get(self.geneIDs[j]) != geneB:
            x = self.get_histogram_bins(str(networki)+":"+self.geneIDs[j], geneB)
            
            histogram_list.append(x)
        return histogram_list



    def get_TF_hub_pair(self, geneA, geneB):
        pair_list = []
        histogram_list = []
        networki = geneA.split(":")[0]

        TF_hub_networki=self.hub_TF.get(networki)
        for j in range(0, len(TF_hub_networki)):
            #if TF_hub_networki[j] != geneB:
            pair_list.append((str(geneA) + "," + str(networki)+":"+TF_hub_networki[j]))
        for j in range(0, len(TF_hub_networki)):
            #if TF_hub_networki[j] != geneA:
            pair_list.append(str(networki)+":"+TF_hub_networki[j] + "," + str(geneB))
        for i in range(0, len(pair_list)):
            tmp = pair_list[i].split(',')
            x = self.get_histogram_bins(tmp[0], tmp[1])
            
            histogram_list.append(x)
        return histogram_list

    def get_random_pairs(self, geneA, geneB):
        pair_list = []
        histogram_list = []
        networki = geneA.split(":")[0]
        for j in range(0, len(self.geneIDs)):
            if self.geneIDs[j] != geneB:
                pair_list.append(str(geneA) + "," + str(networki)+":"+self.geneIDs[j])
        for j in range(0, len(self.geneIDs)):
            if self.geneIDs[j] != geneA:
                pair_list.append(str(networki)+":"+self.geneIDs[j] + "," + str(geneB))
        # order the cov_list
        from random import shuffle
        shuffle(pair_list)
        selected = pair_list[1:self.top_num]
        print(selected)
        for i in range(0, len(selected)):
            tmp = selected[i].split(',')
            x = self.get_histogram_bins(tmp[0], tmp[1])
            
            histogram_list.append(x)
        return histogram_list

    def get_x_for_one_pair_version11(self, geneA, geneB):
        

        # get the first i,j pair, compress or not.
        x = self.get_histogram_bins(geneA, geneB)
        

        #for j,i? get histogram?
        # generate a list to restore histogram
        # get top or random or all?
        histogram_list=[]

        if self.top_or_random_or_all == "all":
            histogram_list = self.get_all_related_pairs(geneA, geneB)
        else:
            if self.top_or_random_or_all == "top":
                # order by decrease cov
                histogram_list=self.get_top_cov_pairs(geneA,geneB)
            elif self.top_or_random_or_all == "random":
                histogram_list=self.get_random_pairs(geneA, geneB)
            elif self.top_or_random_or_all == "TF_hub":
                histogram_list=self.get_TF_hub_pair(geneA,geneB)
        print(len(histogram_list))
        # concantate together, if ij not compress, consider the way put together. or consider multiple channel
        if self.cat_option == "flat":
            if self.flag_ij_compress:
                if self.flag_ij_repeat:
                    #call repeat one
                    multi_image=self.repeat_ij_multiple(x, histogram_list)
                else:
                    #normally cat...
                    print("run cat normal")
                    multi_image=self.normal_cat(x, histogram_list)
            else:
                multi_image = self.normal_cat(x, histogram_list)
        elif self.cat_option == "multi_channel":
            #call multiple channel, by each image and by each pixel
            if self.flag_multiply_weight:
                x=self.multiply_weight_x_ij(x)
            multi_image=self.cat_multiple_channel(x, histogram_list)
        elif self.cat_option == "multi_channel_zhang":
            if self.flag_multiply_weight:
                x=self.multiply_weight_x_ij(x)
            multi_image=self.cat_multiple_channel_zhang(x, histogram_list)

        # multiply by weight
        if self.flag_multiply_weight:
            #call multiple weight, sepearte case of flat and multiple channel
            multi_image=self.multiply_weight(multi_image)
        return multi_image

    def repeat_ij_multiple(self, x_ij, histogram_list):
        index = 0
        one_row = x_ij
        rows = None
        index = index + 1
        for i in range(0, len(histogram_list)):
            one_image = histogram_list[i]
            if index >= self.max_col:
                if rows is None:
                    rows = one_row
                else:
                    rows = np.concatenate((rows, one_row), axis=0)
                one_row = one_image
                index = 1
            else:
                print(shape(one_row))
                print(shape(one_image))
                one_row = np.concatenate((one_row, one_image), axis=1)
                index = index + 1
            #####
            one_image = x_ij
            if index >= self.max_col:
                if rows is None:
                    rows = one_row
                else:
                    rows = np.concatenate((rows, one_row), axis=0)
                one_row = one_image
                index = 1
            else:
                one_row = np.concatenate((one_row, one_image), axis=1)
                index = index + 1

            print(shape(one_row))
        if shape(one_row)[0] > 0:
            if shape(one_row)[0] > 0:
                dim2 = 32
                if shape(one_row)[1] < (self.max_col * dim2):
                    print("rest dimension", ((self.max_col * dim2) - shape(one_row)[1]))
                    rest_image = np.zeros((dim2, ((self.max_col * dim2) - shape(one_row)[1])))
                    one_row = np.concatenate((one_row, rest_image), axis=1)
                    rows = np.concatenate((rows, one_row), axis=0)
                else:
                    rows = np.concatenate((rows, one_row), axis=0)
        if rows is None:
            x = one_row
        else:
            x = rows
        print("x", shape(x))
        return x

    def normal_cat(self, x_ij, histogram_list):
        index = 0
        print("shape x_ij",shape(x_ij))
        one_row = x_ij
        rows = None
        index = index + 1
        for i in range(0,len(histogram_list)):
            one_image=histogram_list[i]
            print("shape one image", shape(one_image))
            if index >= self.max_col:
                if rows is None:
                    rows = one_row
                else:
                    rows = np.concatenate((rows, one_row), axis=0)
                one_row = one_image
                index = 1
            else:
                one_row = np.concatenate((one_row, one_image), axis=1)
                index = index + 1

        print("shape rows",shape(rows))
        if shape(one_row)[0] > 0:
            if shape(one_row)[0] > 0:
                dim2=32
                if shape(one_row)[1] < (self.max_col * dim2):
                    print("rest dimension", ((self.max_col * dim2) - shape(one_row)[1]))
                    rest_image = np.zeros((dim2, ((self.max_col * dim2) - shape(one_row)[1])))
                    one_row = np.concatenate((one_row, rest_image), axis=1)
                    rows = np.concatenate((rows, one_row), axis=0)
                else:
                    rows = np.concatenate((rows, one_row), axis=0)
        if rows is None:
            x = one_row
        else:
            x = rows
        print("x", shape(x))
        return x


    def cat_not_compressed_ij(self, x_ij, histogram_list):
        pass

    def cat_multiple_channel_zhang(self, x_ij, histogram_list):
        x=[]
        x.append(x_ij)
        for i in range(0, len(histogram_list)):
            x.append(histogram_list[i])
        print("x shape",shape(x))
        return x


    def cat_multiple_channel(self, x_ij, histogram_list):
        #calculate the size of each channel by total num
        index = 0
        if len(shape(x_ij)) == 2:
            reshape_size=shape(x_ij)[0]*shape(x_ij)[1]
        elif len(shape(x_ij)) == 1:
            reshape_size = shape(x_ij)[0]
        elif len(shape(x_ij)) == 3:
            reshape_size=shape(x_ij)[0]*shape(x_ij)[1]*shape(x_ij)[2]
        totoal_num = 1+len(histogram_list)
        self.max_col = math.ceil(sqrt(totoal_num))
        one_image = x_ij.reshape(1,1,reshape_size)
        one_row = one_image
        index = index + 1
        rows = None

        for i in range(0,len(histogram_list)):
            one_image=histogram_list[i]
            one_image = one_image.reshape(1, 1, reshape_size)
            print("shape one image", shape(one_image))
            if index >= self.max_col:
                if rows is None:
                    rows = one_row
                else:
                    rows = np.concatenate((rows, one_row), axis=0)
                one_row = one_image
                index = 1
            else:
                one_row = np.concatenate((one_row, one_image), axis=1)
                index = index + 1

        print("shape rows", shape(rows))
        if shape(one_row)[0] > 0:
            if shape(one_row)[0] > 0:
                if shape(one_row)[1] < self.max_col:
                    print("rest dimension", ((self.max_col) - shape(one_row)[1]))
                    rest_image = np.zeros((1, ((self.max_col) - shape(one_row)[1]),reshape_size))
                    one_row = np.concatenate((one_row, rest_image), axis=1)
                    rows = np.concatenate((rows, one_row), axis=0)
                else:
                    rows = np.concatenate((rows, one_row), axis=0)
        if rows is None:
            x = one_row
        else:
            x = rows
        print("x", shape(x))
        return x

    def multiply_weight(self, multi_image):
        if self.cat_option == "flat":
            #mulpiply a weight matrix element wise
            weight_matrix = np.ones(shape(multi_image))
            dim1 = 32
            dim2 = 32
            weight_matrix[0:dim1, 0:dim2] = 10

            multi_image=np.multiply(multi_image, weight_matrix)
        else:
            #deal with it
            pass
        return multi_image

    def multiply_weight_x_ij(self, x_ij):
        #mulpiply a weight matrix element wise
        weight_matrix = np.ones(shape(x_ij))
        dim1 = 32
        dim2 = 32
        weight_matrix[0:dim1, 0:dim2] = 10

        x_ij=np.multiply(x_ij, weight_matrix)
        return x_ij


    def setting_for_one_pair(self, flag_removeNoise=False, top_num=1000, top_or_random_or_all="top", flag_ij_repeat=True,
                             flag_ij_compress=False, cat_option="flat", flag_multiply_weight=True):
        self.flag_removeNoise = flag_removeNoise
        self.top_num = top_num
        if top_or_random_or_all== "all": #"all","top","random","TF_hub"
            self.max_col# assign by node num
        elif top_or_random_or_all == "TF":
            pass
        elif top_or_random_or_all=="TF_hub":
            self.max_col = math.ceil(sqrt(2 * top_num + 1))
            print("max_col=", self.max_col)
        elif top_or_random_or_all == "top":
            #assign by top num/ or TF num
            self.max_col = math.ceil(sqrt(2 * top_num+1))
            print("max_col=", self.max_col)
        elif top_or_random_or_all == "random":
            self.max_col = math.ceil(sqrt(2 * top_num + 1))
            print("max_col=", self.max_col)
        self.top_or_random_or_all = top_or_random_or_all
        self.flag_ij_repeat = flag_ij_repeat
        self.flag_ij_compress = flag_ij_compress
        self.cat_option = cat_option  # "flat", or multiple channel "multi_channel", "multi_channel_zhang"
        self.flag_multiply_weight = flag_multiply_weight

def work_for_one_setup(network_num, node_num, output_dir, input_dir,divide_part,pair_in_each_net):
    #num_in_each_batch = round((network_num * 2) / 5)
    num_in_each_batch = None
    ec = ExprCollection(
        output_dir,
        x_method_version=11, max_col=None, pair_in_batch_num=num_in_each_batch, getlog=True,
        node_num=node_num, divide_part=divide_part,pair_in_each_net=pair_in_each_net)  # 650,1300,2000

    ec.setting_for_one_pair(flag_removeNoise=False, top_num=10, top_or_random_or_all="top", flag_ij_repeat=False,
                            flag_ij_compress=False, cat_option="multi_channel_zhang", flag_multiply_weight=False)


    for i in range(1, (network_num + 1)):
        simulate_expr_file = input_dir + str(i) + "_sim_data.csv"
        simulate_theta_file = input_dir + str(i) + "_theta.csv"
        simulate_cov_file = input_dir + str(i) + "_cov.csv"
        ec.load_data_expr_answer_indirect(simulate_expr_file, simulate_theta_file, simulate_cov_file, i)

    ec.get_train_test()


def main():
    node_num_vector = [4]
    network_num = 2500
    sparsity = 0.5
    replicate = 1
    divide_part = 5
    node_num = 4
    pair_in_each_net = 2 #2, None for all pos + equal number neg


    output_dir="simulation_data/"
    input_dir="simulation_node4_indirect_reshow/"

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    work_for_one_setup(network_num, node_num, output_dir, input_dir,divide_part, pair_in_each_net)


if __name__ == '__main__':
    main()



