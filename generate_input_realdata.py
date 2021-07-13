
from __future__ import print_function


import pandas as pd
from numpy import *
import numpy as np
import json, re,os, sys
#from GENIE3 import *
import argparse




parser = argparse.ArgumentParser(description="")

parser.add_argument('-out_dir', required=True, help='Indicate the path for output.')
parser.add_argument('-expr_file', required=True, help='The file of the gene expression profile. Can be h5 or csv file, the format please refer the example data.')
parser.add_argument('-pairs_for_predict_file', required=True, help='The file of the training gene pairs and their labels.')
parser.add_argument('-geneName_map_file', required=True, default=None, help='The file to map the name of gene in expr_file to the pairs_for_predict_file')

parser.add_argument('-flag_load_from_h5', default=False, help='Is the expr_file is a h5 file. True or False.')
parser.add_argument('-flag_load_split_batch_pos', default=True, help='Is there a file that indicate the position in pairs_for_predict_file to divide pairs into different TFs.')
parser.add_argument('-TF_divide_pos_file', default=None, help='File that indicate the position in pairs_for_predict_file to divide pairs into different TFs.')

parser.add_argument('-TF_num', type=int, default=None, help='To generate representation for this number of TFs. Should be a integer that equal or samller than the number of TFs in the pairs_for_predict_file.')
parser.add_argument('-TF_order_random', default=False, help='If the TF_num samller than the number of TFs in the pairs_for_predict_file, we need to indicate TF_order_random, if TF_order_random=True, then the code will generate representation for randomly selected TF_num TFs.')

parser.add_argument('-top_or_random', default="top_cov", help='Decide how to select the neighbor images. Can be set as "top_cov","top_corr", "random")
parser.add_argument('-get_abs', default=False, help='Select neighbor images by considering top value or top absolute value.)



args = parser.parse_args()

class RepresentationTest2:
    def __init__(self,output_dir,x_method_version=1, max_col=None, pair_in_batch_num=250, start_batch_num=0,
                 end_batch_num=None,getlog=False,plot_histogram=False, load_batch_split_pos=False, print_xdata=True, cellNum=None):
        # input
        self.load_batch_split_pos = load_batch_split_pos
        self.expr = None  # a table
        self.geneIDs = None  #
        self.geneID_to_index_in_expr = {}
        self.rpkm = None
        self.sampleIDs = None  # not necessary
        self.geneID_map = None  # not necessary, ID in expr to ID in gold standard
        self.ID_to_name_map = None
        self.geneIDs_lists_for_goldkey = []
        self.split_batch_pos = None
        self.gold_standard = {}  # geneA;geneB -> 0,1,2 #note direction, geneA,geneB is diff with geneB,geneA
        self.networki_geneID_to_expr = {}
        self.networki_genepair_to_cov = {}
        self.geneID_to_candidate_genes = None
        self.output_dir = output_dir
        self.base_out_dir = output_dir
        self.x_method_version = x_method_version
        self.pair_in_batch_num = pair_in_batch_num
        self.getlog=getlog
        self.trained_singleImage_model = None
        self.max_col = max_col
        self.print_xdata = print_xdata
        self.chipseq_filter_corr_cutoff = None
        self.key_list = []
        self.generate_key_list = []
        self.file_generate_key_list = None

        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
            os.mkdir(output_dir+"version11")
            os.mkdir(output_dir+"version0")

        ##setting for one pair
        self.flag_removeNoise = None
        self.top_num = None
        self.top_or_random_or_all = None
        self.flag_ij_repeat = None
        self.flag_ij_compress = None
        self.cat_option = None  # flat, or multiple channel "multi_channel"
        self.flag_multiply_weight = None
        self.add_self_image = None
        self.get_abs = None

        self.plot_histogram = plot_histogram
        self.geneIDs_TF = []

        self.end_batch_num= end_batch_num
        self.start_batch_num = start_batch_num

        self.cov_matrix = None
        self.corr_matrix = None
        self.hub_TF = None
        self.hub_TF_num = 30
        self.TF_index = None

        self.cellNum = cellNum


    def get_expr_by_networki_geneName(self, geneA):  #for simulation, for liver
        index=self.geneID_map.get(str(geneA))
        if index is None:
            index = int(geneA)
            geneA_x = self.rpkm.iloc[:,index]

        else:
            #index=int(index)
            if type(self.geneIDs[0])==int:
                index=int(index)
            index2 = np.where(self.geneIDs==index)
            index = index2[0]
            geneA_x = self.rpkm.iloc[:,index]
        geneA_x=geneA_x.to_numpy().reshape(-1)
        return geneA_x

    def get_index_by_networki_geneName(self, geneA):  #for simulation, for liver
        index=self.geneID_map.get(str(geneA))
        if index is None:
            index = int(geneA)
            #print("gene", geneA, "not found")
            #return None
        else:
            if type(self.geneIDs[0])==int:
                index=int(index)
            index2 = np.where(self.geneIDs==index)
            index = index2[0]
        return index

    def get_expr_by_networki_geneName_version2(self, geneA):#for boneMarrow
        index=self.geneID_map.get(geneA)
        if index is None:
            index = int(geneA)
            geneA_x = self.rpkm[self.geneIDs[index]][:]
        else:
            index=int(index)
            geneA_x = self.rpkm[index][:]
        return geneA_x

    def get_index_by_networki_geneName_version2(self, geneA):#for boneMarrow
        index=self.geneID_map.get(geneA)
        if index is None:
            index = int(geneA)
        else:
            index=int(index)
            index2 = np.where(self.geneIDs==index)
            index = index2[0]
        return index

    def get_gene_list(self, file_name):
        import re
        h = {}
        h2 = {}
        s = open(file_name, 'r')  # gene symbol ID list of sc RNA-seq
        for line in s:
            search_result = re.search(r'^([^\s]+)\s+([^\s]+)', line)
            #print("-",search_result.group(1),"-",search_result.group(2),"-")
            h[str(search_result.group(1).lower())] = str(search_result.group(2).lower())  # h [gene symbol] = gene ID
            h2[str(search_result.group(2).lower())] = str(search_result.group(1).lower()) #h2 geneID = gene symbol
        self.geneID_map = h
        self.ID_to_name_map = h2
        s.close()

    def load_real_data(self, filename):
        # '/home/yey3/sc_process_1/rank_total_gene_rpkm.h5')    # scRNA-seq expression data )#
        store = pd.HDFStore(filename)
        print(store.keys())
        rpkm = store['/RPKMs']

        if self.cellNum is None:
            self.rpkm = rpkm
        else:
            import random
            print("len(rpkm.index)",len(rpkm.index))
            select_cells = np.asarray(random.sample(range(len(rpkm.index)),self.cellNum))
            self.rpkm = rpkm.iloc[select_cells][:]

        store.close()
        self.geneIDs=rpkm.columns
        self.geneIDs=np.asarray(self.geneIDs,dtype=str)
        print("self.geneIDs",self.geneIDs)
        #for i in range(0, len(self.geneIDs)):
        #    self.geneIDs[i]=str(self.geneIDs[i])
        #    self.geneIDs[i] = self.geneIDs[i].lower()
        print("gene nums",len(rpkm.columns))
        print("cell nums", len(rpkm.index))
        #out_table_df = rpkm.T
        #out_table_df.to_csv(self.output_dir+"bone_marrow_rpkm.csv",sep=",")
        #print(rpkm.index)

    def load_real_data_sparse(self,filename,geneNameFile,study_info_file, select_str='healthy_liver'):
        import scipy.sparse
        df2 = pd.read_table(study_info_file,header=None, dtype=str)
        X = df2.iloc[:, 0]
        index = np.where(X == select_str)
        index = index[0]

        sparse_matrix = scipy.sparse.load_npz(filename)
        selected_cells_sparse_matrix = sparse_matrix.tocsr()[:, index]
        self.rpkm = pd.DataFrame.sparse.from_spmatrix(selected_cells_sparse_matrix)####for pandas v0.25
        #self.rpkm = pd.SparseDataFrame(selected_cells_sparse_matrix) #for pandas v0.24

        self.rpkm = self.rpkm.T
        print("rpkm[0,]",self.rpkm.iloc[0,:])
        print("rpkm[,0]",self.rpkm.iloc[:,0])
        print("shape self.rpkm",self.rpkm.shape)
        df=pd.read_table(geneNameFile,header=None)
        self.geneIDs=df.iloc[:,0]
        self.geneIDs = np.asarray(self.geneIDs, dtype=str)
        print("len geneIDs", len(self.geneIDs))
        for i in range(0,len(self.geneIDs)):
            self.geneIDs[i]=self.geneIDs[i].lower()
        print("geneIDs",self.geneIDs)


    def load_real_data_from_csv(self,filename):
        df = pd.read_csv(filename)
        print("load data shape",df.shape)
        print("df columns:",df.columns)
        print("df index", df.index)
        self.rpkm=df
        self.geneIDs=df.columns
        self.geneIDs = np.asarray(self.geneIDs, dtype=str)
        print("len geneIDs", len(self.geneIDs))
        for i in range(0, len(self.geneIDs)):
            self.geneIDs[i] = self.geneIDs[i].lower()
        print("geneIDs", self.geneIDs)

    def load_real_data_singlecelltype(self, expr_file):
        df = pd.read_csv(expr_file,header='infer',index_col=0)
        print("load data shape", df.shape)
        print("df columns:", df.columns)
        print("df index", df.index)
        self.rpkm = df.T
        self.geneIDs = df.index
        self.geneIDs = np.asarray(self.geneIDs, dtype=str)
        print("len geneIDs", len(self.geneIDs))
        for i in range(0, len(self.geneIDs)):
            self.geneIDs[i] = self.geneIDs[i].lower()
        print("geneIDs", self.geneIDs)

    def get_gold_standard(self,filename):
        unique_keys={}
        s = open(filename)  # 'mmukegg_new_new_unique_rand_labelx.txt')#)   ### read the gene pair and label file
        for line in s:
            separation = line.split()
            geneA_name, geneB_name, label = separation[0], separation[1], separation[2]
            geneA_name = geneA_name.lower()
            geneB_name = geneB_name.lower()
            key=str(geneA_name)+","+str(geneB_name)
            key2=str(geneB_name)+","+str(geneA_name)
            #if key not in self.gold_standard:
                #if key2 not in self.gold_standard:
            if self.load_batch_split_pos:
                if key in self.gold_standard.keys():
                    if label == int(2):
                        pass
                    else:
                        self.gold_standard[key] = int(label)
                    self.key_list.append(key)
                else:
                    self.gold_standard[key] = int(label)
                    self.key_list.append(key)
            else:
                if int(label) != 2:
                    unique_keys[geneA_name] = self.geneID_map.get(geneA_name)
                    unique_keys[geneB_name] = self.geneID_map.get(geneB_name)

                    if geneA_name in unique_keys:
                        if geneB_name in unique_keys:
                            print(key,label,int(label))
                            self.gold_standard[key] = int(label)
        s.close()
        print("gold standard length",len(self.gold_standard.keys()))

    def load_split_batch_pos(self,filename):
        self.split_batch_pos = []
        s = open(filename)  # 'mmukegg_new_new_unique_rand_labelx.txt')#)   ### read the gene pair and label file
        for line in s:
            separation = line.split()
            #print("line",line)
            #print("separation",separation)
            self.split_batch_pos.append(separation[0])

        #print(self.split_batch_pos)
        s.close()

    def load_candidate_gene_list(self,filename):
        self.geneID_to_candidate_genes = {}
        s = open(filename)  # 'mmukegg_new_new_unique_rand_labelx.txt')#)   ### read the gene pair and label file
        for line in s:
            separation = line.split(":")
            gene_i = separation[0]
            tmp = separation[1].split("\n")
            candidate_genes = tmp[0].split(",")
            self.geneID_to_candidate_genes[gene_i] = candidate_genes
        s.close()



    def get_cov_for_genepair(self,geneA,geneB):
        x_geneA = self.get_expr_by_networki_geneName(geneA)
        x_geneB = self.get_expr_by_networki_geneName(geneB)

        return np.cov(x_geneA,x_geneB)[0,1]

    def get_corr_for_genepair(self,geneA,geneB):
        x_geneA = self.get_expr_by_networki_geneName(geneA)
        x_geneB = self.get_expr_by_networki_geneName(geneB)

        return np.corrcoef(x_geneA,x_geneB)[0,1]

    def get_histogram_bins(self, geneA, geneB):
        x_geneA=self.get_expr_by_networki_geneName(geneA)
        x_geneB=self.get_expr_by_networki_geneName(geneB)

        if x_geneA is not None:
            if x_geneB is not None:
                #print("x_geneA",x_geneA)
                #print("x_geneB",x_geneB)
                x_tf = log10(x_geneA + 10 ** -2)  # ## 43261 means the number of samples in the sc data, we also have one row that is sum of all cells, so the real size is 43262, that is why we use [0:43261]. For TF target prediction or other data, just remove "[0:43261]"
                x_gene = log10(x_geneB + 10 ** -2)  # For TF target prediction, remove "[0:43261]"

                H_T = histogram2d(x_tf, x_gene, bins=32)
                H = H_T[0].T
                HT = (log10(H / len(x_tf) + 10 ** -4) + 4) / 4

                return HT
            else:
                return None
        else:
            return None

    def get_x_for_one_pair_version0(self,geneA,geneB):####change here, important.
        #input geneA, geneB, get corresponding expr and get histgram
        #return x, y, z
        x = self.get_histogram_bins(geneA,geneB)
        return x

    def get_gene_pair_data(self,geneA,geneB, x_method_version):
        # input geneA, geneB, get corresponding expr and get histogram
        # return x, y, z
        if self.x_method_version != 0:
            if self.max_col is None:
                self.max_col = 2 * len(self.geneIDs)


        if x_method_version==0:
            x = self.get_x_for_one_pair_version0(geneA,geneB)
        elif x_method_version==11:
            x = self.get_x_for_one_pair_version11(geneA, geneB)

        if x is not None:
            key = str(geneA)+','+str(geneB)
            y = self.gold_standard.get(key)
            z = key
            if self.plot_histogram:
                if not os.path.isdir(self.output_dir +str(x_method_version)+"_histogram/"):
                    os.mkdir(self.output_dir+str(x_method_version) +"_histogram/")
            #print("x.shape", x.shape)
            if self.plot_histogram:
                np.savetxt(self.output_dir +str(x_method_version)+"_histogram/" + z+'_'+str(y)+'_histogram.csv', x, delimiter=",")
            return [x,y,z]
        else:
            return [x,y,z]

    def get_batch(self,gene_list,save_header,x_method_version):
        xdata = []  # numpy arrary [k,:,:,1], k is number o fpairs
        ydata = []  # numpy arrary shape k,1
        zdata = []  # record corresponding pairs
        # for each term in list, split it into two
        # call get_gene_pair_data and append together
        print("x_method_version",x_method_version)
        print("gene_list",gene_list)
        if len(gene_list)>0:
            for i in range(0,len(gene_list)):
                geneA=gene_list[i].split(',')[0]
                geneB=gene_list[i].split(',')[1]
                key = str(geneA) + ',' + str(geneB)
                gold_y = self.gold_standard.get(key)
                print("gene_list[i]", gene_list[i])
                print("gold_y", gold_y)
                if int(gold_y) != 2:
                    [x,y,z] = self.get_gene_pair_data(geneA,geneB,x_method_version)
                    if x is not None:
                        xdata.append(x)
                        ydata.append(y)
                        zdata.append(z)

            if self.print_xdata:
                print("xdata",shape(xdata))

                if (len(xdata) > 0):
                    if len(shape(xdata)) == 4:
                        # xx = np.array(xdata)[:, :, :, :, np.newaxis]
                        xx = xdata
                    elif len(shape(xdata)) == 3:
                        xx = np.array(xdata)[:, :, :, np.newaxis]
                    else:
                        xx = np.array(xdata)[:, :, :, np.newaxis]

                print("xx",shape(xx))

                print("save",save_header)
                np.save(save_header+'_xdata.npy',xx)
                np.save(save_header + '_ydata.npy', np.array(ydata))
                np.save(save_header + '_zdata.npy', np.array(zdata))

    def get_train_test(self,batch_index=None, generate_multi=True,
                       TF_pairs_num_lower_bound=0,TF_pairs_num_upper_bound=None,TF_order_random=False):
        #deal with cross validation or train test batch partition,mini_batch
        self.generate_key_list=[]
        if self.split_batch_pos is not None:
            #from collections import OrderedDict
            #self.gold_standard = OrderedDict(self.gold_standard)
            key_list = self.key_list
        else:
            from collections import OrderedDict
            self.gold_standard = OrderedDict(self.gold_standard)
            key_list = list(self.gold_standard.keys())
            #key_list = list(sorted(self.gold_standard.keys()))

        print("gold standard len:",len(key_list))
        if self.split_batch_pos is not None:
            print(len(self.split_batch_pos))
            print("self.split_batch_pos",self.split_batch_pos)
            index_start_list=[]
            index_end_list=[]

            for i in range(0, (len(self.split_batch_pos)-1)):
                index_start = int(self.split_batch_pos[i])
                index_end = int(self.split_batch_pos[i + 1])

                if (index_end-index_start)>=TF_pairs_num_lower_bound:
                    if TF_pairs_num_upper_bound is None:
                        index_start_list.append(index_start)
                        index_end_list.append(index_end)
                    else:
                        if (index_end-index_start)<=TF_pairs_num_upper_bound:
                            index_start_list.append(index_start)
                            index_end_list.append(index_end)
            if TF_order_random:
                from random import shuffle
                TF_order=list(range(0,len(index_start_list)))
                print("TF_order",TF_order)
                shuffle(TF_order)
                print("TF_order", TF_order)
                TF_order=list(TF_order)
                print("TF_order",TF_order)
                index_start_list=np.asarray(index_start_list)
                index_end_list=np.asarray(index_end_list)
                index_start_list=index_start_list[TF_order]
                index_end_list=index_end_list[TF_order]

            if self.end_batch_num is None:
                self.end_batch_num = len(index_start_list)
            else:
                if self.end_batch_num > len(index_start_list):
                    self.end_batch_num = len(index_start_list) ####?????!!!!!

            print("(len(self.split_batch_pos)-1):",(len(self.split_batch_pos)-1))
            print("len(index_start_list):", len(index_start_list))
            print("self.start_batch_num:", self.start_batch_num)
            print("self.end_batch_num:", self.end_batch_num)

            for i in range(self.start_batch_num, self.end_batch_num):
                print(i)
                index_start = int(index_start_list[i])
                index_end = int(index_end_list[i])
                print("index_start",index_start)
                print("index_end", index_end)
                if index_end <= len(key_list):
                    select_list = list(key_list[j] for j in range(index_start, index_end))

                    for j in range(index_start,index_end):
                        self.generate_key_list.append(key_list[j]+','+str(self.gold_standard.get(key_list[j])))

                    print("select_list",select_list)
                    if generate_multi:
                        self.get_batch(select_list, self.output_dir+"version11/" + str(i),11)
                        self.get_batch(select_list, self.output_dir + "version0/" + str(i), 0)
                    else:
                        self.get_batch(select_list, self.output_dir + str(i), self.x_method_version)
            self.generate_key_list=np.asarray(self.generate_key_list)
            np.savetxt(self.file_generate_key_list, self.generate_key_list,fmt="%s", delimiter='\n')


        else:
            if self.shulffle_key:
                from random import shuffle
                print(key_list)
                shuffle(key_list)
                shuffle(key_list)
            print(len(key_list))
            #print(key_list)

            batches = int(round(len(key_list)/self.pair_in_batch_num))

            print(batches)
            tmp = self.start_batch_num
            if self.end_batch_num is None:
                self.end_batch_num=(tmp+batches)+1

            if self.start_batch_num==0:
                index_start = 0
            else:
                index_start=self.pair_in_batch_num*self.start_batch_num

            for i in range(self.start_batch_num,self.end_batch_num):
                index_end = index_start + self.pair_in_batch_num

                if index_end > len(key_list):
                    index_end = len(key_list)

                select_list = list(key_list[j] for j in range(index_start,index_end))
                print(select_list)
                if generate_multi:
                    self.get_batch(select_list, self.output_dir+"version11/" + str(i),11)
                    self.get_batch(select_list, self.output_dir+"version0/" + str(i), 0)
                else:
                    self.get_batch(select_list, self.output_dir+str(i),self.x_method_version)
                index_start = index_end+1
                self.start_batch_num = i+1



    def get_all_related_pairs(self, geneA, geneB):
        histogram_list = []
        #if from simulation data, we need record network id, and get it by networkid
        networki = geneA.split(":")[0]
        for j in range(0, len(self.geneIDs)):
            if self.ID_to_name_map.get(self.geneIDs[j]) != geneB:
                x = self.get_histogram_bins(geneA, j)
                
                histogram_list.append(x)
        for j in range(0, len(self.geneIDs)):
            if self.ID_to_name_map.get(self.geneIDs[j]) != geneA:
                x = self.get_histogram_bins(j, geneB)
                
                histogram_list.append(x)
        return histogram_list



    def get_top_cov_pairs(self,geneA,geneB,cov_or_corr="cov"):
        # get cov value first
        if self.corr_matrix is None or self.cov_matrix is None:
            self.calculate_cov()
        if cov_or_corr=="corr":
            np.fill_diagonal(self.corr_matrix, 0)


        histogram_list = []
        networki = geneA.split(":")[0]


        x = self.get_histogram_bins(geneA, geneA)
        
        if self.add_self_image:
            histogram_list.append(x)

        x = self.get_histogram_bins(geneB, geneB)
        
        if self.add_self_image:
            histogram_list.append(x)

        index = self.get_index_by_networki_geneName(geneA)
        if cov_or_corr=="cov":
            cov_list_geneA = self.cov_matrix[index, :]
        else:
            cov_list_geneA = self.corr_matrix[index, :]
        cov_list_geneA = cov_list_geneA.ravel()
        if self.get_abs:
            cov_list_geneA = np.abs(cov_list_geneA)
        the_order = np.argsort(-cov_list_geneA)
        select_index = the_order[0:self.top_num]
        for j in select_index:
            #if self.ID_to_name_map.get(self.geneIDs[j]) != geneA:
            x = self.get_histogram_bins(geneA, str(j))
            
            histogram_list.append(x)
        ####
        indexB = self.get_index_by_networki_geneName(geneB)
        if cov_or_corr=="cov":
            cov_list_geneB = self.cov_matrix[indexB, :]
        else:
            cov_list_geneB = self.corr_matrix[indexB, :]
        cov_list_geneB = cov_list_geneB.ravel()
        if self.get_abs:
            cov_list_geneB = np.abs(cov_list_geneB)
        the_order = np.argsort(-cov_list_geneB)
        select_index = the_order[0:self.top_num]
        for j in select_index:
            #if self.ID_to_name_map.get(self.geneIDs[j]) != geneB:
            x = self.get_histogram_bins(str(j), geneB)
            
            histogram_list.append(x)
        return histogram_list

    def get_random_pairs(self, geneA, geneB):
        pair_list = []
        histogram_list = []
        networki = geneA.split(":")[0]
        for j in range(0, len(self.geneIDs)):
            if self.ID_to_name_map.get(self.geneIDs[j]) != geneB:
                pair_list.append(str(geneA) + "," + str(j))
        for j in range(0, len(self.geneIDs)):
            if self.ID_to_name_map.get(self.geneIDs[j]) != geneA:
                pair_list.append(str(j) + "," + str(geneB))
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

    def load_geneIDs_TF(self,filename):
        unique_keys = {}
        index = 0
        window = 7
        s = open(filename)  # 'mmukegg_new_new_unique_rand_labelx.txt')#)   ### read the gene pair and label file
        for line in s:
            index = index +1
            separation = line.split()
            geneA_name, geneB_name, label = separation[0], separation[1], separation[2]
            key = str(geneA_name) + "," + str(geneB_name)
            key2 = str(geneB_name) + "," + str(geneA_name)
            # if key not in self.gold_standard:
            # if key2 not in self.gold_standard:
            if index < window:
                if geneA_name in unique_keys:
                    unique_keys[geneA_name] = unique_keys[geneA_name] + 1
                else:
                    unique_keys[geneA_name] = 1
                if geneB_name in unique_keys:
                    unique_keys[geneB_name] = unique_keys[geneB_name] + 1
                else:
                    unique_keys[geneB_name] = 1
            if index == window:
                for key in unique_keys.keys():
                    print(unique_keys.get(key))
                    if unique_keys.get(key) > 3:
                        if key not in self.geneIDs_TF:
                            self.geneIDs_TF.append(key)
                index = 0
                unique_keys = {}

        print("len(geneIDs_TF)",len(self.geneIDs_TF))
        print("geneIDs_TF",self.geneIDs_TF)

        df = pd.DataFrame(self.geneIDs_TF)
        df.to_csv("TF_list.csv")


    def calculate_cov(self):
        expr = self.rpkm.iloc[:][:]
        print("expr shape", shape(expr))
        expr=np.asarray(expr)
        #expr=expr[0:43261,0:2000] ###need remove
        print("expr shape", shape(expr))
        expr = expr.transpose()
        print("expr shape",shape(expr))
        self.cov_matrix = np.cov(expr)
        self.corr_matrix = np.corrcoef(expr)
        print("cov matrix dim", shape(self.cov_matrix))

    def get_hub_TF(self):
        if self.cov_matrix is None:
            self.calculate_cov()
        TF_cov_sum = np.zeros((len(self.geneIDs_TF)))
        i=0
        for TFi in self.geneIDs_TF:
            index = self.get_index_by_networki_geneName(TFi)
            #if index > 2000: ###need remove
            #    index = 1500
            cov_list_TFi= self.cov_matrix[index,:]
            TF_cov_sum[i]=sum(abs(cov_list_TFi))
            i=i+1
        the_order = np.argsort(-TF_cov_sum)
        select_index = the_order[0:self.hub_TF_num]
        self.hub_TF = [self.geneIDs_TF[j] for j in select_index]

    def get_TF_index(self):
        self.TF_index = []
        for TFi in self.geneIDs_TF:
            index = self.get_index_by_networki_geneName(TFi)
            self.TF_index.append(index)
        #self.TF_index = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19] #need remove

    def get_top_cov_TF(self,geneA):
        if self.TF_index is None:
            self.get_TF_index()
        top_TF=None
        index = self.get_index_by_networki_geneName(geneA)
        #if index > 2000:  ###need remove
        #    index = 1500
        cov_list_geneA = self.cov_matrix[index, :]
        #tmp = np.where(np.asarray(self.TF_index)>2000) ##need remove
        #np.asarray(self.TF_index)[tmp[0]]=1500 ##need remove
        # filter by TF

        filtered_cov = cov_list_geneA[self.TF_index] #or remove if get top gene

        # get top cov TF by sort
        the_order = np.argsort(-filtered_cov)
        select_index = the_order[0:self.hub_TF_num]
        #or replace geneIDs_TF to self.geneIDs

        top_TF = [self.geneIDs_TF[j] for j in select_index]
        print("top TF", top_TF)
        return top_TF

    def get_top_corr_TF(self, geneA):
        if self.TF_index is None:
            self.get_TF_index()
        top_TF=None
        index = self.get_index_by_networki_geneName(geneA)
        #if index > 2000:  ###need remove
        #    index = 1500
        corr_list_geneA = self.corr_matrix[index, :]
        #tmp = np.where(np.asarray(self.TF_index)>2000) ##need remove
        #np.asarray(self.TF_index)[tmp[0]]=1500 ##need remove
        # filter by TF

        filtered_corr = corr_list_geneA[self.TF_index] #or remove if get top gene

        # get top cov TF by sort
        the_order = np.argsort(-filtered_corr)
        select_index = the_order[0:self.hub_TF_num]
        #or replace geneIDs_TF to self.geneIDs

        top_TF = [self.geneIDs_TF[j] for j in select_index]
        print("top TF", top_TF)
        return top_TF


    def get_TF_pairs(self, geneA, geneB):
        if self.geneIDs_TF is None:
            self.load_geneIDs_TF()
        histogram_list = []

        for j in range(0, len(self.geneIDs_TF)):
            #if geneA != self.geneIDs_TF[j]:
            x = self.get_histogram_bins(geneA, self.geneIDs_TF[j])
            histogram_list.append(x)

        for j in range(0,len(self.geneIDs_TF)):
            #if geneB != self.geneIDs_TF[j]:
            x = self.get_histogram_bins(geneB, self.geneIDs_TF[j])
            histogram_list.append(x)

        return histogram_list

    def get_TF_hub_pairs(self, geneA, geneB):
        if self.geneIDs_TF is None:
            self.load_geneIDs_TF()
        if self.hub_TF is None:
            self.get_hub_TF()

        histogram_list = []

        for j in range(0, len(self.hub_TF)):
            if geneA != self.hub_TF[j]:
                x = self.get_histogram_bins(geneA, self.hub_TF[j])
                histogram_list.append(x)

        for j in range(0,len(self.hub_TF)):
            if geneB != self.hub_TF[j]:
                x = self.get_histogram_bins(geneB, self.hub_TF[j])
                histogram_list.append(x)

        return histogram_list


    def get_TF_top_cov_pairs(self, geneA, geneB, cov_or_corr="cov"):
        if self.geneIDs_TF is None:
            self.load_geneIDs_TF()
        if self.hub_TF is None:
            self.get_hub_TF()

        if cov_or_corr=="cov":
            top_TF = self.get_top_cov_TF(geneA)
        else:
            top_TF = self.get_top_corr_TF(geneA)

        histogram_list = []
        for j in range(0, len(top_TF)):
            if geneA != top_TF[j]:
                x = self.get_histogram_bins(geneA, top_TF[j])
                histogram_list.append(x)

        for j in range(0, len(top_TF)):
            if geneB != top_TF[j]:
                x = self.get_histogram_bins(geneB, top_TF[j])
                histogram_list.append(x)

        if cov_or_corr=="cov":
            top_TF = self.get_top_cov_TF(geneB)
        else:
            top_TF = self.get_top_corr_TF(geneB)
        for j in range(0, len(top_TF)):
            if geneA != top_TF[j]:
                x = self.get_histogram_bins(geneA, top_TF[j])
                histogram_list.append(x)

        for j in range(0, len(top_TF)):
            if geneB != top_TF[j]:
                x = self.get_histogram_bins(geneB, top_TF[j])
                histogram_list.append(x)
        return histogram_list

    def get_candidate_genes_pairs(self,geneA, geneB):
        if self.geneID_to_candidate_genes is None:
            print("error, should load geneID_to_candidate_genes first")
        geneA_candidate = []
        geneID_A = self.geneID_map.get(geneA)
        if geneID_A in self.geneID_to_candidate_genes:
            geneA_candidate_ID = self.geneID_to_candidate_genes.get(geneID_A)
            for i in range(0, len(geneA_candidate_ID)):
                name_i = self.ID_to_name_map.get(str(geneA_candidate_ID[i]))
                geneA_candidate.append(name_i)

        geneB_candidate = []
        geneID_B = self.geneID_map.get(geneB)
        if geneID_B in self.geneID_to_candidate_genes:
            geneB_candidate_ID = self.geneID_to_candidate_genes.get(geneID_B)
            for i in range(0,len(geneB_candidate_ID)):
                name_i = self.ID_to_name_map.get(str(geneB_candidate_ID[i]))
                geneB_candidate.append(name_i)

        histogram_list = []

        for j in range(0, len(geneA_candidate)):
            if geneA != geneA_candidate[j]:
                x = self.get_histogram_bins(geneA, geneA_candidate[j])
                histogram_list.append(x)

        for j in range(0,len(geneB_candidate)):
            if geneB != geneB_candidate[j]:
                x = self.get_histogram_bins(geneB, geneB_candidate[j])
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
            if self.top_or_random_or_all == "top_cov":
                # order by decrease cov
                histogram_list=self.get_top_cov_pairs(geneA,geneB,"cov")
            elif self.top_or_random_or_all == "top_corr":
                histogram_list = self.get_top_cov_pairs(geneA, geneB,"corr")
            elif self.top_or_random_or_all == "random":
                histogram_list=self.get_random_pairs(geneA, geneB)
            elif self.top_or_random_or_all == "TF":
                histogram_list=self.get_TF_pairs(geneA, geneB)
            elif self.top_or_random_or_all == "TF_hub":
                self.hub_TF_num = self.top_num
                histogram_list=self.get_TF_hub_pairs(geneA, geneB)
            elif self.top_or_random_or_all == "TF_top_cov":
                self.hub_TF_num = self.top_num
                histogram_list=self.get_TF_top_cov_pairs(geneA, geneB)
            elif self.top_or_random_or_all == "TF_top_corr":
                self.hub_TF_num = self.top_num
                histogram_list = self.get_TF_top_cov_pairs(geneA, geneB,"corr")
            elif self.top_or_random_or_all == "candidate_gene":
                histogram_list = self.get_candidate_genes_pairs(geneA, geneB)


        if len(histogram_list)>0:
            print("len histogram", len(histogram_list))
            # concantate together, if ij not compress, consider the way put together. or consider multiple channel
            if self.cat_option == "flat":
                if self.flag_ij_compress:
                    if self.flag_ij_repeat:
                        #call repeat one
                        multi_image=self.repeat_ij_multiple(x, histogram_list)
                    else:
                        #normally cat...
                        multi_image=self.normal_cat(x, histogram_list)
                else:
                    multi_image = self.normal_cat(x, histogram_list)
            elif self.cat_option == "multi_channel":
                #call multiple channel, by each image and by each pixel
                if self.flag_multiply_weight:
                    x = self.multiply_weight_x_ij(x)
                multi_image=self.cat_multiple_channel(x, histogram_list)
            elif self.cat_option == "multi_channel_zhang":
                if self.flag_multiply_weight:
                    x = self.multiply_weight_x_ij(x)
                multi_image = self.cat_multiple_channel_zhang(x, histogram_list)

            # multiply by weight
            if self.flag_multiply_weight:
                #call multiple weight, sepearte case of flat and multiple channel
                multi_image=self.multiply_weight(multi_image)
        else:
            multi_image = x
        return multi_image

    def repeat_ij_multiple(self, x_ij, histogram_list):
        index = 0
        one_row = [x_ij]
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
        print(shape(x_ij))
        one_row = x_ij
        rows = None
        if len(histogram_list)>0:
            index = index + 1
            for i in range(0,len(histogram_list)):
                one_image=histogram_list[i]
                print(shape(one_image))
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

            print(shape(rows))
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

    def cat_multiple_channel(self, x_ij, histogram_list):
        # calculate the size of each channel by total num
        index = 0
        if len(shape(x_ij)) == 2:
            reshape_size = shape(x_ij)[0] * shape(x_ij)[1]
        elif len(shape(x_ij)) == 1:
            reshape_size = shape(x_ij)[0]
        elif len(shape(x_ij)) == 3:
            reshape_size = shape(x_ij)[0] * shape(x_ij)[1] * shape(x_ij)[2]
        totoal_num = 1 + len(histogram_list)
        self.max_col = math.ceil(sqrt(totoal_num))
        one_image = x_ij.reshape(1, 1, reshape_size)
        one_row = one_image
        index = index + 1
        rows = None

        for i in range(0, len(histogram_list)):
            one_image = histogram_list[i]
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
                    rest_image = np.zeros((1, ((self.max_col) - shape(one_row)[1]), reshape_size))
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

    def cat_multiple_channel_zhang(self, x_ij, histogram_list):
        x=[]
        x.append(x_ij)
        for i in range(0, len(histogram_list)):
            if histogram_list[i] is not None:
                x.append(histogram_list[i])
        print("x shape",shape(x))
        return x




    def quantileNormalize(self, df_input):
        df = df_input.copy()
        print("shape df", shape(df))
        # compute rank
        dic = {}
        for col in df:

            print(col)
            dic.update({col: sorted(df[col])})
        sorted_df = pd.DataFrame(dic)
        rank = sorted_df.mean(axis=1).tolist()
        # sort
        for col in df:
            t = np.searchsorted(np.sort(df[col]), df[col])
            df[col] = [rank[i] for i in t]
        return df




    def output_goldstandard_corr(self,outfile="filtered_gold_standard",print_gold_standard_and_corr=True, chipseq_filter_corr_cutoff = 0.1):
        from collections import OrderedDict
        self.gold_standard = OrderedDict(self.gold_standard)
        gold_standard_keys = list(self.gold_standard.keys())
        even_flag = 1
        if print_gold_standard_and_corr:
            output_table = []

            if self.corr_matrix is None:
                self.calculate_cov()

            #outF = open(self.output_dir+"co-expression_gold_standard,csv", 'w')
            for i in range(0,len(gold_standard_keys)):
                keyi = gold_standard_keys[i]
                separation = keyi.split(",")
                geneA, geneB = separation[0], separation[1]
                #corri = self.get_corr_for_genepair(geneA, geneB)
                indexA = self.get_index_by_networki_geneName(geneA)
                indexB = self.get_index_by_networki_geneName(geneB)
                corri = self.corr_matrix[indexA, indexB]

                #if type(corri)==np.ndarray:
                    #corri=corri[0]
                labeli = self.gold_standard.get(keyi)
                oneline=[geneA, geneB, str(labeli),  str(corri)]
                if chipseq_filter_corr_cutoff is not None:
                    if labeli == 0:
                        if even_flag > 0:
                            if abs(corri) > 0.5:
                                output_table.append(oneline)
                                #even_flag = even_flag - 1
                    else:
                        if abs(corri) > chipseq_filter_corr_cutoff:
                            output_table.append(oneline)
                            #even_flag = even_flag + 1
                else:
                    output_table.append(oneline)
                #outF.write(keyi + "," + str(self.gold_standard.get(keyi)) + "," + str(corri) + '\n')
            #outF.close()
            output_table=np.asarray(output_table)
            print("shape output_table", shape(output_table))
            np.savetxt(self.output_dir+outfile,output_table,delimiter="\t",fmt="%s")



    def setting_for_one_pair(self, flag_removeNoise=False, top_num=1000, top_or_random_or_all="top_cov", flag_ij_repeat=True,
                             flag_ij_compress=False, cat_option="flat", flag_multiply_weight=True, shulffle_key=True, add_self_image=True, get_abs=False):
        self.flag_removeNoise = flag_removeNoise

        self.top_num = top_num
        self.top_or_random_or_all = top_or_random_or_all #"top_cov", "random", "TF", "TF_hub", "TF_top_cov", "candidate_gene", "top_corr",  "nmf_signature", "nmf_genes", "nmf_similarity_genes"
        self.flag_ij_repeat = flag_ij_repeat
        self.flag_ij_compress = flag_ij_compress
        self.cat_option = cat_option  # flat, or multiple channel "multi_channel","multi_channel_zhang"
        self.flag_multiply_weight = flag_multiply_weight
        self.shulffle_key = shulffle_key
        self.add_self_image = add_self_image
        self.get_abs = get_abs



def main_for_representation_single_cell_type(out_dir, expr_file, pairs_for_predict_file, TF_divide_pos_file=None, geneName_map_file=None, TF_num=None,
                                             TF_pairs_num_lower_bound=0, TF_pairs_num_upper_bound=None,TF_order_random=False,
                                             flag_load_split_batch_pos=True,flag_load_from_h5=False,cellNum=None, add_self_image=True,
                                             top_or_random_or_all="top_cov",cat_option="multi_channel_zhang",get_abs=False):
    if out_dir.endswith("/"):
        pass
    else:
        out_dir=out_dir+"/"

    ec = RepresentationTest2(
        out_dir, x_method_version=11, plot_histogram=False,
        load_batch_split_pos=flag_load_split_batch_pos, start_batch_num=0, end_batch_num=TF_num, pair_in_batch_num=500, max_col=1,
        print_xdata=True,cellNum=cellNum)
    ec.setting_for_one_pair(flag_removeNoise=False, top_num=10, top_or_random_or_all=top_or_random_or_all, flag_ij_repeat=False,
                            flag_ij_compress=False, cat_option=cat_option, flag_multiply_weight=False,
                            shulffle_key=False,add_self_image=add_self_image, get_abs=get_abs)

    if flag_load_split_batch_pos:
        ec.load_split_batch_pos(TF_divide_pos_file)

    ec.get_gene_list(geneName_map_file)

    if flag_load_from_h5:
        ec.load_real_data(expr_file)
    else:
        ec.load_real_data_singlecelltype(expr_file)

    ec.get_gold_standard(pairs_for_predict_file)

    ec.file_generate_key_list = pairs_for_predict_file+"_for_generate"


    if ec.x_method_version == 11:
        if ec.top_or_random_or_all == "TF_hub":
            ec.get_hub_TF()
    ec.get_train_test(generate_multi=True,TF_pairs_num_lower_bound=TF_pairs_num_lower_bound,TF_pairs_num_upper_bound=TF_pairs_num_upper_bound,TF_order_random=TF_order_random)


def filter_low_expr_genes(gene_names, expr):
    cell_num=expr.shape[0]
    one_percent_cell_num=np.round(cell_num*0.01)
    print("one_percent_cell_num",one_percent_cell_num)
    tmp=np.where(expr>0)
    expr_pos=np.zeros(expr.shape)
    expr_pos[tmp]=1
    sum_pos_per_col=np.sum(expr_pos,axis=0)
    print(sum_pos_per_col.shape)
    print(sum_pos_per_col)
    select_genes_exprs_zero=np.where(sum_pos_per_col>one_percent_cell_num)
    select_genes_exprs_zero=select_genes_exprs_zero[0]

    print("len(select_genes_exprs_zero)",len(select_genes_exprs_zero))
    gene_names=gene_names[select_genes_exprs_zero]
    expr=expr[:,select_genes_exprs_zero]

    sum_expr_per_col=np.sum(expr,axis=0)
    select_genes_exprs_large=np.where(sum_expr_per_col>1*one_percent_cell_num)
    print("sum_expr_per_col>2*one_percent_cell_num",sum_expr_per_col>2*one_percent_cell_num)
    print("select_genes_exprs_large")
    print(len(select_genes_exprs_large[0]))
    print(select_genes_exprs_large)
    select_genes_exprs_large=select_genes_exprs_large[0]
    gene_names=gene_names[select_genes_exprs_large]
    expr=expr[:,select_genes_exprs_large]

    return gene_names, expr


def filter_nonCoding_genes(gene_names, expr, gene_annotation_file="gse140228_umi_counts_droplet_genes.tsv",
                           out_file="protein_coding_gene_names.txt"):
    filtered_index=[]
    filtered_gene_names=[]

    dict_name_to_biotype={}
    df=pd.read_table(gene_annotation_file)
    names_in_dict=df.iloc[:,1]
    biotype_in_dict=df.iloc[:,6]
    for i in range(0, len(names_in_dict)):
        print(names_in_dict[i], biotype_in_dict[i])
        dict_name_to_biotype[names_in_dict[i].lower()]=biotype_in_dict[i]

    for i in range(0,len(gene_names)):
        if dict_name_to_biotype.get(gene_names[i].lower())=='protein_coding':
            filtered_gene_names.append(gene_names[i])
            filtered_index.append(i)

    filtered_expr = expr[:, filtered_index]
    #output filter gene names to file
    filtered_gene_names=np.asarray(filtered_gene_names)
    np.savetxt(out_file,filtered_gene_names,fmt="%s")
    return filtered_gene_names, filtered_expr

def filter_low_var_genes(gene_names, expr):
    print("len(gene_names)", len(gene_names))
    vars = np.var(expr, axis=0)
    vars = np.asarray(vars, dtype=float)
    print("len(vars)", len(vars))
    select_genes_index = np.where(vars > 0)
    select_genes_index = select_genes_index[0]
    print("len(select_genes_index", len(select_genes_index))
    print("select_genes_index", select_genes_index)

    expr = expr[:, select_genes_index]
    gene_names = gene_names[select_genes_index]
    return gene_names, expr


def generate_train_pairs_by_TF_genes(gene_names, TF,out_file):
    out_train_pair_list=[]
    for i in range(0,len(TF)):
        for j in range(0,len(gene_names)):
            one_pair=str(TF[i])+'\t'+str(gene_names[j])+'\t'+'3'
            out_train_pair_list.append(one_pair)

    out_train_pair_list=np.asarray(out_train_pair_list)
    np.savetxt(out_file,out_train_pair_list,fmt='%s',delimiter='\n')



if __name__ == '__main__':


    flag_load_split_batch_pos = (args.flag_load_split_batch_pos=='True')
    flag_load_from_h5 = (args.flag_load_from_h5=='True')
    TF_order_random = (args.TF_order_random=='True')
    get_abs = (args.get_abs=='True')
    if args.TF_num=='None':
        TF_num = None
    else:
        TF_num = args.TF_num


    main_for_representation_single_cell_type(out_dir=args.out_dir, expr_file=args.expr_file, pairs_for_predict_file=args.pairs_for_predict_file, TF_divide_pos_file=args.TF_divide_pos_file, geneName_map_file=args.geneName_map_file, TF_num=TF_num,
                                             TF_order_random=TF_order_random,flag_load_split_batch_pos=flag_load_split_batch_pos,flag_load_from_h5=flag_load_from_h5,top_or_random_or_all=args.top_or_random,get_abs=get_abs)

    if False:
        main_for_representation_single_cell_type("bonemarrow_representation", 'bone_marrow_cell.h5', 'gold_standard_for_TFdivide',
                                                 TF_divide_pos_file='whole_gold_split_pos',
                                                 geneName_map_file="sc_gene_list.txt", TF_num=None,
                                                 flag_load_split_batch_pos=True,flag_load_from_h5=True,
                                                 add_self_image=True,cat_option="multi_channel_zhang",top_or_random_or_all="top_cov",get_abs=False)
    

    if False:
        main_for_representation_single_cell_type('hESC_representation_nobound_top_cov_noabs', 'hESC/ExpressionData.csv', "training_pairshESC.txt",
                                                 "training_pairshESC.txtTF_divide_pos.txt", "hESC_geneName_map.txt",TF_num=18,
                                                 TF_order_random=True, flag_load_split_batch_pos=True,add_self_image=True,
                                                 cat_option="multi_channel_zhang",top_or_random_or_all="top_cov",get_abs=False)
    


