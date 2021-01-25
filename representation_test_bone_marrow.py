
from __future__ import print_function


import pandas as pd
from numpy import *
import numpy as np
import json, re,os, sys
#from GENIE3 import *


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
        self.autoencoder = None
        self.encoder = None
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
        self.flag_autoencodder = None
        self.autoencoder_dim1 = None
        self.autoencoder_dim2 = None
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

        self.W = None
        self.H_top = None
        self.H = None
        self.gene_similarity_by_nmf = None
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

    def load_real_data_lung(self,filename):
        df = pd.read_table(filename,header='infer',index_col=0)
        print("load data shape",df.shape)
        print("df columns:",df.columns)
        print("df index", df.index)
        self.rpkm=df.T
        self.geneIDs=df.index
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

    def do_nmf(self,n_components=500):
        A=self.rpkm
        from scipy import sparse
        X=sparse.csr_matrix(A.values)

        from sklearn.decomposition import NMF
        model = NMF(n_components=n_components,init='nndsvd')
        self.W = model.fit_transform(X)
        self.H = model.components_
        W_df = pd.DataFrame(self.W)
        H_df = pd.DataFrame(self.H)
        W_df.to_csv(self.output_dir+"W_n_component_"+str(n_components)+".csv")
        H_df.to_csv(self.output_dir+"H_n_component_"+str(n_components)+".csv")

    def load_NMF_result(self,H_file,W_file):
        W = pd.read_csv(W_file,index_col=0)
        self.W = W.values
        print("W shape",self.W.shape)
        H = pd.read_csv(H_file,index_col=0)
        self.H = H.values
        print("H shape", self.H.shape)

        self.H_top = np.zeros(self.H.shape)
        H_top_index = np.argmax(self.H, axis=0)
        for i in range(0,len(H_top_index)):
            self.H_top[H_top_index[i],i] = self.H[H_top_index[i],i]

        #calculate similarity between H
        self.gene_similarity_by_nmf = np.corrcoef(self.H, rowvar=False)
        print("shape gene_similarity_by_nmf", self.gene_similarity_by_nmf.shape)


    def get_histogram_with_signature(self, geneA, signature):
        x_geneA = self.get_expr_by_networki_geneName(geneA)
        x_geneB = signature
        x_tf = log10(
            x_geneA + 10 ** -2)  # ## 43261 means the number of samples in the sc data, we also have one row that is sum of all cells, so the real size is 43262, that is why we use [0:43261]. For TF target prediction or other data, just remove "[0:43261]"
        x_gene = log10(x_geneB + 10 ** -2)  # For TF target prediction, remove "[0:43261]"

        H_T = histogram2d(x_tf, x_gene, bins=32)
        H = H_T[0].T
        HT = (log10(H / len(x_tf) + 10 ** -4) + 4) / 4
        return HT

    def get_related_signature_list(self,geneA, k=3):
        if k>self.H.shape[0]:
            print("error: the top num should not larger than NMF signature num")
        related_signature_list = []
        indexA = self.get_index_by_networki_geneName(geneA)
        H_column = self.H[:, indexA]
        # get the class of the gene, by get the index of the highest value in H_column.
        #np.asarray(H_column)
        H_column=H_column.ravel()

        the_order = np.argsort(-H_column)
        selected = the_order[0:k]
        for i in range(0, len(selected)):
            signature_indexi = selected[i]
            related_signature_list.append(self.W[:,signature_indexi])
        return related_signature_list

    def get_signature_in_W(self, geneA, geneB):
        histogram_list = []
        x = self.get_histogram_bins(geneA, geneA)
        if self.flag_autoencodder:
            x = self.get_value_from_autoencoder(x)
        histogram_list.append(x)

        x = self.get_histogram_bins(geneB, geneB)
        if self.flag_autoencodder:
            x = self.get_value_from_autoencoder(x)
        histogram_list.append(x)

        signature_list=self.get_related_signature_list(geneA,self.top_num)
        for signature in signature_list:
            x = self.get_histogram_with_signature(geneA, signature)
            histogram_list.append(x)

        signature_list_B=self.get_related_signature_list(geneB,self.top_num)
        for signature in signature_list_B:
            x = self.get_histogram_with_signature(geneB, signature)
            histogram_list.append(x)
        return histogram_list

    def get_related_genes_by_nmf(self,geneA,k=3):
        indexA = self.get_index_by_networki_geneName(geneA)
        H_column = self.H[:,indexA]
        # get the class of the gene, by get the index of the highest value in H_column.
        class_index = np.argmax(H_column)
        H_row=self.H_top[class_index,:]
        # get hub gene in the class, get top k genes which also includes in the class
        top_genes_in_class = []

        np.asarray(H_row)

        the_order = np.argsort(-H_row)
        selected = the_order[0:k]
        for i in range(0, len(selected)):
            genei = selected[i]
            top_genes_in_class.append(self.ID_to_name_map.get(str(self.geneIDs[genei])))
        return top_genes_in_class


    def get_related_pairs_by_nmf_similarity(self, geneA, geneB):
        histogram_list = []

        x = self.get_histogram_bins(geneA, geneA)
        if self.flag_autoencodder:
            x = self.get_value_from_autoencoder(x)
        histogram_list.append(x)

        x = self.get_histogram_bins(geneB, geneB)
        if self.flag_autoencodder:
            x = self.get_value_from_autoencoder(x)
        histogram_list.append(x)

        index = self.get_index_by_networki_geneName(geneA)
        cov_list_geneA = self.gene_similarity_by_nmf[index, :]
        cov_list_geneA = cov_list_geneA.ravel()
        the_order = np.argsort(-cov_list_geneA)
        select_index = the_order[0:self.top_num]
        for j in select_index:
            # if self.ID_to_name_map.get(self.geneIDs[j]) != geneA:
            x = self.get_histogram_bins(geneA, str(j))
            if self.flag_autoencodder:
                x = self.get_value_from_autoencoder(x)
            histogram_list.append(x)

        indexB = self.get_index_by_networki_geneName(geneB)
        cov_list_geneB = self.gene_similarity_by_nmf[indexB, :]
        cov_list_geneB = cov_list_geneB.ravel()
        the_order = np.argsort(-cov_list_geneB)
        select_index = the_order[0:self.top_num]
        for j in select_index:
            # if self.ID_to_name_map.get(self.geneIDs[j]) != geneB:
            x = self.get_histogram_bins(str(j), geneB)
            if self.flag_autoencodder:
                x = self.get_value_from_autoencoder(x)
            histogram_list.append(x)
        return histogram_list


    def get_related_pairs_by_nmf(self, geneA, geneB):
        histogram_list = []

        x = self.get_histogram_bins(geneA, geneA)
        if self.flag_autoencodder:
            x = self.get_value_from_autoencoder(x)
        histogram_list.append(x)

        x = self.get_histogram_bins(geneB, geneB)
        if self.flag_autoencodder:
            x = self.get_value_from_autoencoder(x)
        histogram_list.append(x)

        geneList=self.get_related_genes_by_nmf(geneA,self.top_num)
        for geneK in geneList:
            x = self.get_histogram_bins(geneA,geneK)
            histogram_list.append(x)

        geneList=self.get_related_genes_by_nmf(geneB,self.top_num)
        for geneK in geneList:
            x = self.get_histogram_bins(geneK,geneB)
            histogram_list.append(x)
        return histogram_list

    ###

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

    def load_autoencoder_model(self):
        if self.autoencoder_dim1 == 16:
            save_dir = "/Users/jiaxchen2/Desktop/realdata/autoencoder_real_half1_16_4/"
        elif self.autoencoder_dim1 == 8:
            save_dir = "/Users/jiaxchen2/Desktop/realdata/autoencoder_real_half1_8_4/"
        elif self.autoencoder_dim1 == 4:
            save_dir = "/Users/jiaxchen2/Desktop/realdata/autoencoder_real_half1_4_4/"
        autoencoder_model_path = save_dir + "keras_autoencoder.h5"
        encoder_model_path = save_dir + "keras_encoder.h5"
        from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
        from keras.models import Model
        from keras import regularizers
        from keras.callbacks import EarlyStopping, ModelCheckpoint
        import keras

        input_img = Input(shape=(32, 32, 1))
        if self.autoencoder_dim1 == 16:
            x = Conv2D(16, (3, 3), activation='relu', padding='same', activity_regularizer=regularizers.l1(10e-5))(
                input_img)  # 16*32*32
            x = MaxPooling2D((2, 2), padding='same')(x)  # 16*16*16
            x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)  # 8*16*16
            x = MaxPooling2D((2, 2), padding='same')(x)  # 8*8*8
            x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)  # 8*8*8
            encoded = MaxPooling2D((2, 2), padding='same')(x)  # 8*4*4
            # at this point the representation is (4, 4, 8) i.e. 128-dimensional
            x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)  # 8*4*4
            x = UpSampling2D((2, 2))(x)  # 8*8*8
            x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)  ##8*8*8
            x = UpSampling2D((2, 2))(x)  # 8*16*16
            x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)  # 16*16*16
            x = UpSampling2D((2, 2))(x)  # 16*32*32
            decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)  # 1*32*32
        elif self.autoencoder_dim1 == 8:

            x = Conv2D(8, (3, 3), activation='relu', padding='same', activity_regularizer=regularizers.l1(10e-5))(
                input_img)  # 16*32*32
            x = MaxPooling2D((2, 2), padding='same')(x)  # 16*16*16
            x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)  # 8*16*16
            x = MaxPooling2D((2, 2), padding='same')(x)  # 8*8*8
            x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)  # 8*8*8
            x = MaxPooling2D((2, 2), padding='same')(x)  # 8*4*4
            x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)  # 4*4*4
            encoded = MaxPooling2D((2, 2), padding='same')(x)  # 4*2*2

            # at this point the representation is (4, 4, 8) i.e. 128-dimensional to 4*4
            x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)  # 4*2*2
            x = UpSampling2D((2, 2))(x)  # 4*4*4
            x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)  # 8*4*4
            x = UpSampling2D((2, 2))(x)  # 8*8*8
            x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)  ##8*8*8
            x = UpSampling2D((2, 2))(x)  # 8*16*16
            x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)  # 16*16*16
            x = UpSampling2D((2, 2))(x)  # 16*32*32
            decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)  # 1*32*32
        elif self.autoencoder_dim1 == 4:
            x = Conv2D(8, (3, 3), activation='relu', padding='same', activity_regularizer=regularizers.l1(10e-5))(
                input_img)  # 16*32*32
            x = MaxPooling2D((2, 2), padding='same')(x)  # 16*16*16
            x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)  # 8*16*16
            x = MaxPooling2D((2, 2), padding='same')(x)  # 8*8*8
            x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)  # 8*8*8
            x = MaxPooling2D((2, 2), padding='same')(x)  # 8*4*4
            x = Conv2D(4, (3, 3), activation='relu', padding='same')(x)  # 4*4*4
            encoded = MaxPooling2D((2, 2), padding='same')(x)  # 4*2*2

            # at this point the representation is (4, 4, 8) i.e. 128-dimensional to 4*4
            x = Conv2D(4, (3, 3), activation='relu', padding='same')(encoded)  # 4*2*2
            x = UpSampling2D((2, 2))(x)  # 4*4*4
            x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)  # 8*4*4
            x = UpSampling2D((2, 2))(x)  # 8*8*8
            x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)  ##8*8*8
            x = UpSampling2D((2, 2))(x)  # 8*16*16
            x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)  # 16*16*16
            x = UpSampling2D((2, 2))(x)  # 16*32*32
            decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)  # 1*32*32
        #### end deep autoencoder ####

        # this model maps an input to its reconstruction
        autoencoder = Model(input_img, decoded)

        encoder = Model(input_img, encoded)

        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

        encoder.load_weights(encoder_model_path)
        autoencoder.load_weights(autoencoder_model_path)
        self.autoencoder = autoencoder
        self.encoder = encoder

    def get_value_from_autoencoder(self, x_test):
        xx = array(x_test)[:, :, newaxis]

        xx_data = []
        xx_data.append(xx)

        xx_array = np.array(xx_data)

        encoded_imgs = self.encoder.predict(xx_array)
        encoded_imgs = encoded_imgs.reshape(self.autoencoder_dim1,self.autoencoder_dim2)
        return encoded_imgs

    def get_all_related_pairs(self, geneA, geneB):
        histogram_list = []
        #if from simulation data, we need record network id, and get it by networkid
        networki = geneA.split(":")[0]
        for j in range(0, len(self.geneIDs)):
            if self.ID_to_name_map.get(self.geneIDs[j]) != geneB:
                x = self.get_histogram_bins(geneA, j)
                if self.flag_autoencodder:
                    x=self.get_value_from_autoencoder(x)
                histogram_list.append(x)
        for j in range(0, len(self.geneIDs)):
            if self.ID_to_name_map.get(self.geneIDs[j]) != geneA:
                x = self.get_histogram_bins(j, geneB)
                if self.flag_autoencodder:
                    x = self.get_value_from_autoencoder(x)
                histogram_list.append(x)
        return histogram_list

    def get_top_cov_pairs_backup(self, geneA, geneB):
        # get cov value first
        cov_list = []
        pair_list = []
        histogram_list = []
        networki = geneA.split(":")[0]
        for j in range(0, len(self.geneIDs)):
            if self.ID_to_name_map.get(self.geneIDs[j]) != geneB:
                cov_value = self.get_cov_for_genepair(geneA, str(j))
                cov_list.append(cov_value)
                pair_list.append(geneA + "," + str(j))
        for j in range(0, len(self.geneIDs)):
            if self.ID_to_name_map.get(self.geneIDs[j]) != geneA:
                cov_value = self.get_cov_for_genepair(str(j), geneB)
                cov_list.append(cov_value)
                pair_list.append(str(j) + "," + geneB)
        # order the cov_list
        #selected = np.argpartition(cov_list, -self.top_num)[-self.top_num:]
        np.asarray(cov_list)
        the_order = np.argsort(-cov_list)
        selected = the_order[0:self.top_num]
        for i in range(0,len(selected)):
            indexi=selected[i]
            tmp=pair_list[indexi].split(",")
            x=self.get_histogram_bins(tmp[0],tmp[1])
            if self.flag_autoencodder:
                x=self.get_value_from_autoencoder(x)
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
        if self.flag_autoencodder:
            x = self.get_value_from_autoencoder(x)
        if self.add_self_image:
            histogram_list.append(x)

        x = self.get_histogram_bins(geneB, geneB)
        if self.flag_autoencodder:
            x = self.get_value_from_autoencoder(x)
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
            if self.flag_autoencodder:
                x = self.get_value_from_autoencoder(x)
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
            if self.flag_autoencodder:
                x = self.get_value_from_autoencoder(x)
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
            if self.flag_autoencodder:
                x = self.get_value_from_autoencoder(x)
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
        if self.flag_removeNoise:
            # represent input data with go through autoencoder, call read expr table.
            pass

        # get the first i,j pair, compress or not.
        x = self.get_histogram_bins(geneA, geneB)
        if self.flag_autoencodder:
            if self.flag_ij_compress:
                x = self.get_value_from_autoencoder(x)

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
            elif self.top_or_random_or_all == "nmf_signature":
                histogram_list = self.get_signature_in_W(geneA, geneB)
            elif self.top_or_random_or_all == "nmf_genes":
                histogram_list = self.get_related_pairs_by_nmf(geneA, geneB)
            elif self.top_or_random_or_all == "nmf_similarity_genes":
                histogram_list = self.get_related_pairs_by_nmf_similarity(geneA,geneB)

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
                    if self.flag_autoencodder:
                        # consider the dimension...
                        multi_image = self.cat_not_compressed_ij(x, histogram_list)
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
                if self.flag_autoencodder:
                    dim2 = self.autoencoder_dim2
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
                    if self.flag_autoencodder:
                        dim2=self.autoencoder_dim2
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

    def multiply_weight(self, multi_image):
        if self.cat_option == "flat":
            #mulpiply a weight matrix element wise
            weight_matrix = np.ones(shape(multi_image))
            dim1 = 32
            dim2 = 32
            if self.flag_autoencodder:
                dim1 = self.autoencoder_dim1
                dim2 = self.autoencoder_dim2
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
        if self.flag_autoencodder:
            dim1 = self.autoencoder_dim1
            dim2 = self.autoencoder_dim2
        weight_matrix[0:dim1, 0:dim2] = 10

        x_ij=np.multiply(x_ij, weight_matrix)
        return x_ij


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

    def filter_and_normalize(self,out_dir,flag_filter=True,flag_normalize=True):
        expr = self.rpkm.iloc[:][:]
        expr = np.asarray(expr)
        #test=expr[0:43261,18000:20000]
        test=expr
        print(shape(self.rpkm))
        vars = np.var(test, axis=0)
        log_vars = np.log(vars)
        avgs = np.average(test,axis=0)
        log_avgs = np.log(avgs)
        nonzero_counts = np.count_nonzero(test,axis=0)
        log_non_zeros = np.log(nonzero_counts)
        print(log_vars)

        log_vars_remove_infinite = log_vars[np.isfinite(log_vars)]
        hist_log_vars, bin_log_edges_vars = np.histogram(log_vars_remove_infinite, bins=40)
        print("var log hist",hist_log_vars)
        print(bin_log_edges_vars)
        np.savetxt(out_dir+"hist_log_vars.txt",hist_log_vars)
        np.savetxt(out_dir+"bin_log_edges_vars.txt", bin_log_edges_vars)


        hist_vars, bin_edges_vars = np.histogram(vars, bins=40)
        print("var hist", hist_vars)
        print(bin_edges_vars)
        np.savetxt(out_dir + "hist_vars.txt", hist_vars)
        np.savetxt(out_dir + "bin_edges_vars.txt", bin_edges_vars)


        log_avgs_remove_infinite = log_avgs[np.isfinite(log_avgs)]
        hist_log_avgs, bin_log_edges_avgs = np.histogram(log_avgs_remove_infinite, bins=40)
        print("avg log hist", hist_log_avgs)
        print(bin_log_edges_avgs)
        np.savetxt(out_dir+"hist_log_avgs.txt", hist_log_avgs)
        np.savetxt(out_dir+"bin_log_edges_avgs.txt", bin_log_edges_avgs)

        hist_avgs, bin_edges_avgs = np.histogram(avgs, bins=40)
        print("avg hist", hist_avgs)
        print(bin_edges_avgs)
        np.savetxt(out_dir + "hist_avgs.txt", hist_avgs)
        np.savetxt(out_dir + "bin_edges_avgs.txt", bin_edges_avgs)


        log_non_zero_remove_infinite = log_non_zeros[np.isfinite(log_non_zeros)]
        hist_log_non_zeros, bin_log_edges_non_zeros = np.histogram(log_non_zero_remove_infinite, bins=40)
        print("non zero log hist", hist_log_non_zeros)
        print(bin_log_edges_non_zeros)
        np.savetxt(out_dir+"hist_log_non_zeros.txt", hist_log_non_zeros)
        np.savetxt(out_dir+"bin_log_edges_non_zeros.txt", bin_log_edges_non_zeros)

        hist_non_zeros, bin_edges_non_zeros = np.histogram(nonzero_counts, bins=40)
        print("non zero hist", hist_non_zeros)
        print(bin_edges_non_zeros)
        np.savetxt(out_dir + "hist_non_zeros.txt", hist_non_zeros)
        np.savetxt(out_dir + "bin_edges_non_zeros.txt", bin_edges_non_zeros)

        if flag_filter:
            filter_index = np.all([vars > 0.162, avgs > 0.7, nonzero_counts > 293.9], axis=0)

            filtered = np.where(filter_index)
            print(filtered)
            print("select gene num",shape(filtered))

            filtered_expr = test[:,filtered[0]]
            filtered_geneID = self.geneIDs[filtered[0]]
            #print(shape(filtered_expr))
            out_expr = np.transpose(filtered_expr)
            df = pd.DataFrame(out_expr, index=filtered_geneID)
            df.to_csv(out_dir+"filtered_expr.csv")
            if flag_normalize:
                normalized_df = self.quantileNormalize(df)
                normalized_df.to_csv(out_dir+"filtered_normalized_expr.csv")




    def convert_goldstandard_to_co_expression_label(self,print_gold_standard_and_corr=False):
        if self.geneIDs_TF is None:
            self.load_geneIDs_TF()

        TF_to_group = {}
        from collections import OrderedDict
        self.gold_standard = OrderedDict(self.gold_standard)
        gold_standard_keys = list(self.gold_standard.keys())
        print("gold_standard_keys",gold_standard_keys)
        for j in range(0,len(gold_standard_keys)):
            key = gold_standard_keys[j]
            separation = key.split(",")
            geneA_name, geneB_name = separation[0], separation[1]
            label = self.gold_standard.get(key)
            if label != 0:
                if geneA_name in self.geneIDs_TF:
                    if geneA_name in TF_to_group.keys():
                        group = TF_to_group.get(geneA_name)
                        group.append(geneB_name)
                        TF_to_group[geneA_name] = group
                    else:
                        group = []
                        group.append(geneA_name)
                        group.append(geneB_name)
                        TF_to_group[geneA_name]=group
                if geneB_name in self.geneIDs_TF:
                    if geneB_name in TF_to_group.keys():
                        group = TF_to_group.get(geneB_name)
                        group.append(geneA_name)
                        TF_to_group[geneB_name] = group
                    else:
                        group = []
                        group.append(geneB_name)
                        group.append(geneA_name)
                        TF_to_group[geneB_name]=group

        key_TFs = list(TF_to_group.keys())
        for i in range(0,len(key_TFs)):
            TFi = key_TFs[i]
            group = TF_to_group.get(TFi)
            for j in range(0,len(group)):
                for k in range(0, len(group)):
                    if k > j:
                        key = str(group[j])+","+str(group[k])
                        key2 = str(group[k])+","+str(group[j])
                        if key in self.gold_standard.keys():
                            self.gold_standard[key]=1
                            print("convert",group[j],str(group[k]),"to 1")
                        else:
                            print("not found ", group[j],str(group[k]))
                        if key2 in self.gold_standard.keys():
                            self.gold_standard[key2]=1
                            print("convert", str(group[k]), group[j], "to 1")
                        else:
                            print("not found ", str(group[k]), group[j])

        if print_gold_standard_and_corr:
            self.output_goldstandard_corr()

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



    def setting_for_one_pair(self, flag_removeNoise=False, flag_autoencodder=True, autoencoder_dim1=4,
                             autoencoder_dim2=4, top_num=1000, top_or_random_or_all="top_cov", flag_ij_repeat=True,
                             flag_ij_compress=False, cat_option="flat", flag_multiply_weight=True, shulffle_key=True, add_self_image=True, get_abs=False):
        self.flag_removeNoise = flag_removeNoise
        self.flag_autoencodder = flag_autoencodder
        self.autoencoder_dim1 = autoencoder_dim1
        self.autoencoder_dim2 = autoencoder_dim2
        self.top_num = top_num
        self.top_or_random_or_all = top_or_random_or_all #"top_cov", "random", "TF", "TF_hub", "TF_top_cov", "candidate_gene", "top_corr",  "nmf_signature", "nmf_genes", "nmf_similarity_genes"
        self.flag_ij_repeat = flag_ij_repeat
        self.flag_ij_compress = flag_ij_compress
        self.cat_option = cat_option  # flat, or multiple channel "multi_channel","multi_channel_zhang"
        self.flag_multiply_weight = flag_multiply_weight
        self.shulffle_key = shulffle_key
        self.add_self_image = add_self_image
        self.get_abs = get_abs


def main_for_real_data_single():
    ec = RepresentationTest2(
        '/Users/jiaxchen2/Desktop/realdata/boneMarrow2/TFdivide/',  x_method_version=11, plot_histogram=False,
        load_batch_split_pos=True,start_batch_num=0,end_batch_num=None,pair_in_batch_num=5000, max_col=1,print_xdata=True)
    ec.setting_for_one_pair(flag_removeNoise=False, flag_autoencodder=False, autoencoder_dim1=4,
                            autoencoder_dim2=4, top_num=10, top_or_random_or_all="top_cov", flag_ij_repeat=False,
                            flag_ij_compress=False, cat_option="multi_channel_zhang", flag_multiply_weight=False,
                            shulffle_key=False, add_self_image=True, get_abs= False)

    if ec.flag_autoencodder:
        ec.load_autoencoder_model()
    filename2="/Users/jiaxchen2/Desktop/HKBU_network/11-CNNC-master/data/sc_gene_list.txt"
    ec.get_gene_list(filename2)
    filename = "/Users/jiaxchen2/Desktop/HKBU_network/11-CNNC-master/data/expression/bone_marrow_cell.h5"
    ec.load_real_data(filename)

    #filename = "/Users/jiaxchen2/Desktop/realdata/boneMarrow2/magic_out_mat.csv"
    #ec.load_real_data_from_csv(filename)

    filename3 = "/Users/jiaxchen2/Desktop/realdata/boneMarrow/gold_standard_for_TFdivide"
    #filename3 = "/Users/jiaxchen2/Desktop/HKBU_network/11-CNNC-master/data/bone_marrow_gene_pairs_200.txt"
    #filename3 = "/Users/jiaxchen2/Desktop/realdata/boneMarrow2/gold_standard_for_corr0.1_pos_neg_c0rr0.07.txt"
    #filename3 = "/Users/jiaxchen2/Desktop/realdata/boneMarrow2/gold_standard_corr_boneMarrowImputed_corr0.6_neg0.5.txt"
    ec.get_gold_standard(filename3)
    #ec.load_geneIDs_TF(filename3)

    #filename4 = "/Users/jiaxchen2/Desktop/realdata/co-expression-WGCNA-boneMarrow/candidate_gene_0.75.txt"
    #ec.load_candidate_gene_list(filename4)

    #ec.output_goldstandard_corr(outfile="gold_standard_corr_boneMarrowImputed_corr0.6_neg0.5.txt",chipseq_filter_corr_cutoff=0.6)

    filename4 = "whole_gold_split_pos"
    ec.load_split_batch_pos(filename4)

    #ec.calculate_cov()

    #ec.convert_goldstandard_to_co_expression_label()#####!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    if ec.x_method_version==11:
        if ec.top_or_random_or_all== "TF_hub":
            ec.get_hub_TF()
    ec.get_train_test(generate_multi=True)

def tmp_get_genenum_cellnum():
    ec = RepresentationTest2(
        '/Users/jiaxchen2/Desktop/realdata/boneMarrow2/TFdivide/', x_method_version=11, plot_histogram=False,
        load_batch_split_pos=True, start_batch_num=0, end_batch_num=None, pair_in_batch_num=5000, max_col=1,
        print_xdata=True)
    ec.setting_for_one_pair(flag_removeNoise=False, flag_autoencodder=False, autoencoder_dim1=4,
                            autoencoder_dim2=4, top_num=10, top_or_random_or_all="top_cov", flag_ij_repeat=False,
                            flag_ij_compress=False, cat_option="multi_channel_zhang", flag_multiply_weight=False,
                            shulffle_key=False,add_self_image=True,get_abs=False)

    filename = "/Users/jiaxchen2/Desktop/HKBU_network/11-CNNC-master/data/expression/bone_marrow_cell.h5"
    print("bone_marrow_cell")
    ec.load_real_data(filename)

    filename = "/Users/jiaxchen2/Desktop/HKBU_network/11-CNNC-master/data/expression/dendritic_cell.h5"
    print("dendritic")
    ec.load_real_data(filename)

    filename = "/Users/jiaxchen2/Desktop/HKBU_network/11-CNNC-master/data/expression/mesc_cell.h5"
    print("mesc")
    ec.load_real_data(filename)



def main_for_filter_and_normalization():
    ec = RepresentationTest2(
        '/Users/jiaxchen2/Desktop/realdata/co-expression-WGCNA-boneMarrow/')  # 650,1300,2000

    filename2 = "/Users/jiaxchen2/Desktop/HKBU_network/11-CNNC-master/data/expression/sc_gene_list.txt"

    ec.get_gene_list(filename2)
    #filename = "/Users/jiaxchen2/Desktop/HKBU_network/11-CNNC-master/data/expression/rank_total_gene_rpkm.h5"
    filename = "/Users/jiaxchen2/Desktop/HKBU_network/11-CNNC-master/data/expression/bone_marrow_cell.h5"
    ec.load_real_data(filename)


    output_dir='/Users/jiaxchen2/Desktop/realdata/co-expression-WGCNA-boneMarrow/'
    ec.filter_and_normalize(output_dir)

def main_for_nmf():
    ec = RepresentationTest2(
        'NMF_20/', x_method_version=11, plot_histogram=False,
        load_batch_split_pos=True, start_batch_num=0, end_batch_num=None, pair_in_batch_num=5000, max_col=1,
        print_xdata=True)
    ec.setting_for_one_pair(flag_removeNoise=False, flag_autoencodder=False, autoencoder_dim1=4,
                            autoencoder_dim2=4, top_num=10, top_or_random_or_all="top_cov", flag_ij_repeat=False,
                            flag_ij_compress=False, cat_option="multi_channel_zhang", flag_multiply_weight=False,
                            shulffle_key=False,add_self_image=True, get_abs=False)


    #filename2 = "/Users/jiaxchen2/Desktop/HKBU_network/11-CNNC-master/data/sc_gene_list.txt"
    filename2 = "sc_gene_list.txt"
    ec.get_gene_list(filename2)
    #filename = "/Users/jiaxchen2/Desktop/HKBU_network/11-CNNC-master/data/expression/bone_marrow_cell.h5"
    filename = "bone_marrow_cell.h5"
    ec.load_real_data(filename)

    ec.do_nmf(n_components=20)

def main_for_representation_use_nmf(cat_option="multi_channel_zhang"):
    ec = RepresentationTest2(
        'test_for_function/', x_method_version=11, plot_histogram=True,
        load_batch_split_pos=True, start_batch_num=0, end_batch_num=2, pair_in_batch_num=100, max_col=1,
        print_xdata=True)
    ec.setting_for_one_pair(flag_removeNoise=False, flag_autoencodder=False, autoencoder_dim1=4,
                            autoencoder_dim2=4, top_num=10, top_or_random_or_all="nmf_signature", flag_ij_repeat=False,
                            flag_ij_compress=False, cat_option=cat_option, flag_multiply_weight=False,
                            shulffle_key=False)
    #filename2 = "/Users/jiaxchen2/Desktop/HKBU_network/11-CNNC-master/data/sc_gene_list.txt"
    filename2 = "sc_gene_list.txt"
    ec.get_gene_list(filename2)
    #filename = "/Users/jiaxchen2/Desktop/HKBU_network/11-CNNC-master/data/expression/bone_marrow_cell.h5"
    filename = "bone_marrow_cell.h5"
    ec.load_real_data(filename)

    #filename3 = "/Users/jiaxchen2/Desktop/realdata/boneMarrow/gold_standard_for_TFdivide"
    filename3 = "gold_standard_for_TFdivide"
    ec.get_gold_standard(filename3)

    #filename4 = "/Users/jiaxchen2/Desktop/realdata/boneMarrow/whole_gold_split_pos"
    filename4 = "whole_gold_split_pos"
    ec.load_split_batch_pos(filename4)

    if ec.x_method_version == 11:
        if ec.top_or_random_or_all == "TF_hub":
            ec.get_hub_TF()
        if ec.top_or_random_or_all == "nmf_signature" or ec.top_or_random_or_all == "nmf_genes" or ec.top_or_random_or_all == "nmf_similarity_genes":
            #H_file="/Users/jiaxchen2/Desktop/realdata/boneMarrow_nmf/H_n_component_50.csv"
            #W_file="/Users/jiaxchen2/Desktop/realdata/boneMarrow_nmf/W_n_component_50.csv"
            H_file = "NMF_20/H_n_component_20.csv"
            W_file = "NMF_20/W_n_component_20.csv"

            ec.load_NMF_result(H_file,W_file)
    ec.get_train_test(generate_multi=True)

def main_for_representation_use_nmf_liver(select_str='healthy_liver', pairs_for_predict_file="training_pairs_filter0.01_healthy.txt"):
    ec = RepresentationTest2(
        'hcc_GTEx_negLarger_corr0.01/', x_method_version=11, plot_histogram=False,
        load_batch_split_pos=True, start_batch_num=0, end_batch_num=24, pair_in_batch_num=2000, max_col=1,
        print_xdata=True)
    ec.setting_for_one_pair(flag_removeNoise=False, flag_autoencodder=False, autoencoder_dim1=4,
                            autoencoder_dim2=4, top_num=10, top_or_random_or_all="top_cov", flag_ij_repeat=False,
                            flag_ij_compress=False, cat_option="multi_channel_zhang", flag_multiply_weight=False,
                            shulffle_key=False)
    #filename2 = "/Users/jiaxchen2/Desktop/HKBU_network/11-CNNC-master/data/sc_gene_list.txt"
    filename2 = "../data_combined_liver/geneName_map.txt"
    ec.get_gene_list(filename2)
    #filename = "/Users/jiaxchen2/Desktop/HKBU_network/11-CNNC-master/data/expression/bone_marrow_cell.h5"
    filename = "../data_combined_liver/expr_sparse_matrix.npz"
    ec.load_real_data_sparse(filename,"../data_combined_liver/geneNames.txt","../data_combined_liver/combined_study_info.txt",select_str=select_str)

    #filename3 = "/Users/jiaxchen2/Desktop/realdata/boneMarrow/gold_standard_for_TFdivide"
    filename3 = pairs_for_predict_file
    ec.get_gold_standard(filename3)


    filename4 = "training_pairs_nofilter_macs_pvalue_e10_8_hcc.txtTF_divide_pos.txt"
    ec.load_split_batch_pos(filename4)

    if ec.x_method_version == 11:
        if ec.top_or_random_or_all == "TF_hub":
            ec.get_hub_TF()
        if ec.top_or_random_or_all == "nmf_signature" or ec.top_or_random_or_all == "nmf_genes" or ec.top_or_random_or_all == "nmf_similarity_genes":
            #H_file="/Users/jiaxchen2/Desktop/realdata/boneMarrow_nmf/H_n_component_50.csv"
            #W_file="/Users/jiaxchen2/Desktop/realdata/boneMarrow_nmf/W_n_component_50.csv"
            H_file = "H.csv"
            W_file = "W.csv"

            ec.load_NMF_result(H_file,W_file)
    ec.get_train_test(generate_multi=True)

def main_for_representation_use_nmf_lung(pairs_for_predict_file="training_pairs_filter0.01_healthy.txt"):
    ec = RepresentationTest2(
        'lung_macs_p8_byTF_negLarger_corr0.01/', x_method_version=11, plot_histogram=False,
        load_batch_split_pos=True, start_batch_num=0, end_batch_num=None, pair_in_batch_num=2000, max_col=1,
        print_xdata=True)
    ec.setting_for_one_pair(flag_removeNoise=False, flag_autoencodder=False, autoencoder_dim1=4,
                            autoencoder_dim2=4, top_num=10, top_or_random_or_all="top_cov", flag_ij_repeat=False,
                            flag_ij_compress=False, cat_option="multi_channel_zhang", flag_multiply_weight=False,
                            shulffle_key=False)

    filename4 = "training_pairs_macs_lung_negLarger_corr0.01.txtTF_divide_pos.txt"
    ec.load_split_batch_pos(filename4)


    #filename2 = "/Users/jiaxchen2/Desktop/HKBU_network/11-CNNC-master/data/sc_gene_list.txt"
    filename2 = "lung_geneName_map.txt"
    ec.get_gene_list(filename2)
    #filename = "/Users/jiaxchen2/Desktop/HKBU_network/11-CNNC-master/data/expression/bone_marrow_cell.h5"
    filename = "../lung_Allsamples_expr.txt"
    ec.load_real_data_lung(filename)

    #filename3 = "/Users/jiaxchen2/Desktop/realdata/boneMarrow/gold_standard_for_TFdivide"
    filename3 = pairs_for_predict_file
    ec.get_gold_standard(filename3)



    if ec.x_method_version == 11:
        if ec.top_or_random_or_all == "TF_hub":
            ec.get_hub_TF()
        if ec.top_or_random_or_all == "nmf_signature" or ec.top_or_random_or_all == "nmf_genes" or ec.top_or_random_or_all == "nmf_similarity_genes":
            #H_file="/Users/jiaxchen2/Desktop/realdata/boneMarrow_nmf/H_n_component_50.csv"
            #W_file="/Users/jiaxchen2/Desktop/realdata/boneMarrow_nmf/W_n_component_50.csv"
            H_file = "H.csv"
            W_file = "W.csv"

            ec.load_NMF_result(H_file,W_file)
    ec.get_train_test(generate_multi=True)



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
    ec.setting_for_one_pair(flag_removeNoise=False, flag_autoencodder=False, autoencoder_dim1=4,
                            autoencoder_dim2=4, top_num=10, top_or_random_or_all=top_or_random_or_all, flag_ij_repeat=False,
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
        if ec.top_or_random_or_all == "nmf_signature" or ec.top_or_random_or_all == "nmf_genes" or ec.top_or_random_or_all == "nmf_similarity_genes":
            #H_file="/Users/jiaxchen2/Desktop/realdata/boneMarrow_nmf/H_n_component_50.csv"
            #W_file="/Users/jiaxchen2/Desktop/realdata/boneMarrow_nmf/W_n_component_50.csv"
            H_file = "H.csv"
            W_file = "W.csv"

            ec.load_NMF_result(H_file,W_file)
    ec.get_train_test(generate_multi=True,TF_pairs_num_lower_bound=TF_pairs_num_lower_bound,TF_pairs_num_upper_bound=TF_pairs_num_upper_bound,TF_order_random=TF_order_random)



def load_lung_expr(expr_file,ID_to_name_mapfile,id_col,name_col):
    map_table = pd.read_csv(ID_to_name_mapfile, header=None)
    map_dict = {}
    for i in range(0, map_table.shape[0]):
        idi = map_table.iloc[i, id_col]
        namei = map_table.iloc[i, name_col]
        map_dict[idi] = namei
    print(map_dict)

    df = pd.read_table(expr_file, header='infer', index_col=0)
    expr = df.values
    print("expr.shape", expr.shape)

    geneNames_tmp = df.index
    geneNames_tmp = np.asarray(geneNames_tmp)
    geneNames_tmp = geneNames_tmp.ravel()

    print("geneNames", geneNames_tmp)

    geneNames = []
    geneIDs = []
    keep_rows = []
    for i in range(0, len(geneNames_tmp)):
        namei = map_dict.get(geneNames_tmp[i])
        if namei is not None:
            print("get genename:", geneNames_tmp[i], namei)
            geneNames.append(namei)
            geneIDs.append(geneNames_tmp[i])
            keep_rows.append(i)
        else:
            print("can not find self.geneNames_tmp[i]", geneNames_tmp[i])

    geneNames = np.asarray(geneNames)
    geneNames = geneNames.ravel()
    for i in range(0, len(geneNames)):
        geneNames[i] = geneNames[i].lower()

    geneIDs = np.asarray(geneIDs)
    geneIDs = geneIDs.ravel()
    expr = expr[keep_rows, :]
    print("keep expr shape", expr.shape)
    rpkm = pd.DataFrame(expr)
    rpkm=rpkm.T
    return [rpkm, geneNames]


def filter_cell(expr, data_name,cell_info_file, min_cells_in_each_type=1000):
    #for hcc, for lung
    import random
    if data_name=='hcc':

        select_cell_index=[]

        df = pd.read_table(cell_info_file)
        celltype_sub=df.iloc[:,5]
        unique_celltype=unique(celltype_sub)
        for i in range(0,len(unique_celltype)):
            tmp=np.where(celltype_sub==unique_celltype[i])
            print(unique_celltype[i])
            print(len(tmp[0]))
            cells_in_type = tmp[0]
            if len(tmp[0])>min_cells_in_each_type:
                print("shuffle")
                print(cells_in_type)
                random.shuffle(cells_in_type)
                cells_in_type=cells_in_type[0:min_cells_in_each_type]
                print(cells_in_type)
            cells_in_type=list(cells_in_type)
            select_cell_index=select_cell_index+cells_in_type
            print(len(select_cell_index))

        pass
    if data_name=='lung':
        pass

    expr=expr[select_cell_index,:]

    return expr

def select_expr_one_cell_type(expr, data_name,cell_info_file, cell_type_str="Lymphoid-B"):
    if data_name=='hcc':

        select_cell_index=[]

        df = pd.read_table(cell_info_file)
        celltype_sub=df.iloc[:,5]
        barcode = df.iloc[:,0]

        tmp = np.where(celltype_sub == cell_type_str)

        print(len(tmp[0]))
        cells_in_type = tmp[0]
        cells_in_type = list(cells_in_type)
        select_cell_index = select_cell_index + cells_in_type
        print(len(select_cell_index))

        expr = expr[select_cell_index, :]

        filtered_sample_info=df.iloc[select_cell_index,:]
        filtered_sample_info.to_csv(cell_type_str+"_filtered_sampleInfo.csv",sep=",")

        filtered_barcode=barcode[select_cell_index]


    return expr, filtered_barcode

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


def main_liver_data_one_cell_type(select_str='hcc'):
    ec = RepresentationTest2(
        'HCC_B_cell/', x_method_version=11, plot_histogram=False,
        load_batch_split_pos=True, start_batch_num=0, end_batch_num=None, pair_in_batch_num=2000, max_col=1,
        print_xdata=True)
    ec.setting_for_one_pair(flag_removeNoise=False, flag_autoencodder=False, autoencoder_dim1=4,
                            autoencoder_dim2=4, top_num=10, top_or_random_or_all="top_cov", flag_ij_repeat=False,
                            flag_ij_compress=False, cat_option="multi_channel_zhang", flag_multiply_weight=False,
                            shulffle_key=False)

    filename = "expr_sparse_matrix.npz"
    ec.load_real_data_sparse(filename, "geneNames.txt", "combined_study_info.txt", select_str=select_str)

    data = np.asarray(ec.rpkm)
    where_are_NaNs = isnan(data)
    data[where_are_NaNs] = 0
    print("data.shape", data.shape)
    gene_names = np.asarray(ec.geneIDs)
    print("gene_names", gene_names)

    [gene_names, data] = filter_nonCoding_genes(gene_names, data)

    [data, filtered_sampleID] = select_expr_one_cell_type(data, 'hcc', cell_info_file='gse140228_umi_counts_droplet_cellinfo.tsv',cell_type_str="Lymphoid-B")

    gene_names_as_index = np.asarray(gene_names, dtype=str)
    gene_names_as_index = list(gene_names_as_index)
    print("gene_names_as_index",gene_names_as_index)
    df_expr_table = pd.DataFrame(data)
    df_expr_table=df_expr_table.T
    df_expr_table.index = gene_names_as_index
    df_expr_table.columns = filtered_sampleID
    df_expr_table.to_csv("B_cell_proteincoding_expr_hcc.csv",sep=",",header=True, index_label=True)

    print("data.shape", data.shape)
    np.savetxt('filtered_gene_names_B_cell_hcc.txt',gene_names,fmt='%s')

    df_geneName_map = pd.DataFrame(gene_names, gene_names)
    df_geneName_map.to_csv('filtered_geneName_map_B_cell_hcc.txt', sep="\t", header=False)


    TFs = pd.read_csv("TF_ID_symbol_map.csv", header=None, index_col=0)
    regulators = list(TFs.iloc[:, 0])
    filtered_regulators = []
    for i in range(0, len(regulators)):
        regulators[i] = regulators[i].lower()
        if regulators[i] in gene_names:
            filtered_regulators.append(regulators[i])
    print("regulators", regulators)
    filtered_regulators=np.asarray(filtered_regulators)
    np.savetxt('filtered_regulators_B_cell_hcc.txt',filtered_regulators,fmt="%s")



def main_liver_fiilter_gene_test(select_str='hcc'):
    from arboreto.utils import load_tf_names
    from arboreto.algo import grnboost2
    from arboreto.algo import genie3
    from distributed import LocalCluster, Client

    local_cluster = LocalCluster(n_workers=10,
                                 threads_per_worker=1,
                                 memory_limit=8e9)
    custom_client = Client(local_cluster)

    ec = RepresentationTest2(
        'lung_nofilter/', x_method_version=11, plot_histogram=False,
        load_batch_split_pos=True, start_batch_num=0, end_batch_num=None, pair_in_batch_num=2000, max_col=1,
        print_xdata=True)
    ec.setting_for_one_pair(flag_removeNoise=False, flag_autoencodder=False, autoencoder_dim1=4,
                            autoencoder_dim2=4, top_num=10, top_or_random_or_all="top_cov", flag_ij_repeat=False,
                            flag_ij_compress=False, cat_option="multi_channel_zhang", flag_multiply_weight=False,
                            shulffle_key=False)

    filename = "expr_sparse_matrix.npz"
    ec.load_real_data_sparse(filename, "geneNames.txt", "combined_study_info.txt", select_str=select_str)

    data = np.asarray(ec.rpkm)
    where_are_NaNs = isnan(data)
    data[where_are_NaNs] = 0
    print("data.shape", data.shape)
    gene_names = np.asarray(ec.geneIDs)
    print("gene_names", gene_names)

    [gene_names, data] = filter_nonCoding_genes(gene_names, data)

    data = filter_cell(data, 'hcc', cell_info_file='gse140228_umi_counts_droplet_cellinfo.tsv',min_cells_in_each_type=2000)

    [gene_names, data] = filter_low_expr_genes(gene_names, data)

    [gene_names, data] = filter_low_var_genes(gene_names, data)

    print("data.shape", data.shape)
    np.savetxt('filtered_gene_names_hcc.txt',gene_names,fmt='%s')

    TFs = pd.read_csv("TF_ID_symbol_map.csv", header=None, index_col=0)
    regulators = list(TFs.iloc[:, 0])
    filtered_regulators = []
    for i in range(0, len(regulators)):
        regulators[i] = regulators[i].lower()
        if regulators[i] in gene_names:
            filtered_regulators.append(regulators[i])
    print("regulators", regulators)
    filtered_regulators=np.asarray(filtered_regulators)
    np.savetxt('filtered_regulators_hcc.txt',filtered_regulators,fmt="%s")

    generate_train_pairs_by_TF_genes(gene_names, filtered_regulators,"train_pairs_whole_net_hcc_GENIE3filter_globalCelltype.txt")

def generate_train_pairs_by_TF_genes(gene_names, TF,out_file):
    out_train_pair_list=[]
    for i in range(0,len(TF)):
        for j in range(0,len(gene_names)):
            one_pair=str(TF[i])+'\t'+str(gene_names[j])+'\t'+'3'
            out_train_pair_list.append(one_pair)

    out_train_pair_list=np.asarray(out_train_pair_list)
    np.savetxt(out_file,out_train_pair_list,fmt='%s',delimiter='\n')




if __name__ == '__main__':

    if False:
        main_for_representation_single_cell_type("bonemarrow_noSelfImage", 'bone_marrow_cell.h5', 'gold_standard_for_TFdivide',
                                                 TF_divide_pos_file='whole_gold_split_pos',
                                                 geneName_map_file="sc_gene_list.txt", TF_num=None,
                                                 flag_load_split_batch_pos=True,flag_load_from_h5=True,
                                                 add_self_image=False,cat_option="multi_channel_zhang",top_or_random_or_all="top_cov",get_abs=False)
    if False:
        main_for_representation_single_cell_type("bonemarrow_top_corr", 'bone_marrow_cell.h5', 'gold_standard_for_TFdivide',
                                                 TF_divide_pos_file='whole_gold_split_pos',
                                                 geneName_map_file="sc_gene_list.txt", TF_num=None,
                                                 flag_load_split_batch_pos=True,flag_load_from_h5=True,
                                                 add_self_image=True,cat_option="multi_channel_zhang",top_or_random_or_all="top_corr",get_abs=False)
    if False:
        main_for_representation_single_cell_type("bonemarrow_top_cov_abs", 'bone_marrow_cell.h5',
                                                 'gold_standard_for_TFdivide',
                                                 TF_divide_pos_file='whole_gold_split_pos',
                                                 geneName_map_file="sc_gene_list.txt", TF_num=None,
                                                 flag_load_split_batch_pos=True, flag_load_from_h5=True,
                                                 add_self_image=True, cat_option="multi_channel_zhang",
                                                 top_or_random_or_all="top_cov", get_abs=True)

    if False:
        main_for_representation_single_cell_type("bonemarrow_flat_top_cov", 'bone_marrow_cell.h5',
                                                 'gold_standard_for_TFdivide',
                                                 TF_divide_pos_file='whole_gold_split_pos',
                                                 geneName_map_file="sc_gene_list.txt", TF_num=None,
                                                 flag_load_split_batch_pos=True, flag_load_from_h5=True,
                                                 add_self_image=True, cat_option="flat",
                                                 top_or_random_or_all="top_cov",get_abs=False)
    ###############

    if False:
        main_for_representation_single_cell_type('HCC_Lymphonde', 'HCC Lymphnode_expr.csv',
                                                 "training_pairsB_cell_hcc.txt",
                                                 "training_pairsB_cell_hcc.txtTF_divide_pos.txt", "B_cell_hcc_geneName_map.txt",
                                                 TF_num=3,
                                                 TF_order_random=False, flag_load_split_batch_pos=True)
    if False:

        main_for_representation_single_cell_type('CC_Lymphonde', 'CC Lymphnode_expr.csv',
                                                 "training_pairsB_cell_hcc.txt",
                                                 "training_pairsB_cell_hcc.txtTF_divide_pos.txt",
                                                 "B_cell_hcc_geneName_map.txt",
                                                 TF_num=3,
                                                 TF_order_random=False, flag_load_split_batch_pos=True)
    if False:

        main_for_representation_single_cell_type('HCC_Blood', 'HCC Blood_expr.csv',
                                                 "training_pairsB_cell_hcc.txt",
                                                 "training_pairsB_cell_hcc.txtTF_divide_pos.txt",
                                                 "B_cell_hcc_geneName_map.txt",
                                                 TF_num=3,
                                                 TF_order_random=False, flag_load_split_batch_pos=True)
    if False:

        main_for_representation_single_cell_type('HCC_TumorCore', 'HCC TumorCore_expr.csv',
                                                 "training_pairsB_cell_hcc.txt",
                                                 "training_pairsB_cell_hcc.txtTF_divide_pos.txt",
                                                 "B_cell_hcc_geneName_map.txt",
                                                 TF_num=3,
                                                 TF_order_random=False, flag_load_split_batch_pos=True)
    if False:

        main_for_representation_single_cell_type('HCC_Normal', 'HCC Normal_expr.csv',
                                                 "training_pairsB_cell_hcc.txt",
                                                 "training_pairsB_cell_hcc.txtTF_divide_pos.txt",
                                                 "B_cell_hcc_geneName_map.txt",
                                                 TF_num=3,
                                                 TF_order_random=False, flag_load_split_batch_pos=True)


    #main_for_real_data_single()
    #main_for_filter_and_normalization()
    #main_for_nmf()
    #main_for_representation_use_nmf(cat_option='flat')
    #main_for_representation_use_nmf_lung("training_pairs_macs_lung_negLarger_corr0.01.txt")
    #main_for_representation_use_nmf_liver(select_str='hcc',pairs_for_predict_file='training_pairs_nofilter_macs_pvalue_e10_8_hcc.txt')

    #main_for_representation_single_cell_type('mDC_representation','mDC/ExpressionData.csv',"training_pairsmDC.txt",
     #                                        "training_pairsmDC.txtTF_divide_pos.txt", "mDC_geneName_map.txt")

    #main_for_representation_single_cell_type('mESC_representation_nobound', 'mESC/ExpressionData.csv', "training_pairsmESC.txt",
     #                                        "training_pairsmESC.txtTF_divide_pos.txt", "mESC_geneName_map.txt",TF_num=18,
      #                                       TF_order_random=True,flag_load_split_batch_pos=True)
    if False:
        main_for_representation_single_cell_type('hESC_representation_nobound_top_corr_noabs', 'hESC/ExpressionData.csv', "training_pairshESC.txt",
                                                 "training_pairshESC.txtTF_divide_pos.txt", "hESC_geneName_map.txt",TF_num=18,
                                                 TF_order_random=True, flag_load_split_batch_pos=True,add_self_image=True,
                                                 cat_option="multi_channel_zhang",top_or_random_or_all="top_corr",get_abs=False)
    if False:
        main_for_representation_single_cell_type('hESC_representation_nobound_top_cov_abs', 'hESC/ExpressionData.csv', "training_pairshESC.txt",
                                                 "training_pairshESC.txtTF_divide_pos.txt", "hESC_geneName_map.txt",TF_num=18,
                                                 TF_order_random=True, flag_load_split_batch_pos=True,add_self_image=True,
                                                 cat_option="multi_channel_zhang",top_or_random_or_all="top_cov",get_abs=True)
    if False:
        main_for_representation_single_cell_type('hESC_representation_nobound_top_cov_noabs', 'hESC/ExpressionData.csv', "training_pairshESC.txt",
                                                 "training_pairshESC.txtTF_divide_pos.txt", "hESC_geneName_map.txt",TF_num=18,
                                                 TF_order_random=True, flag_load_split_batch_pos=True,add_self_image=True,
                                                 cat_option="multi_channel_zhang",top_or_random_or_all="top_cov",get_abs=False)
    if False:
        main_for_representation_single_cell_type('hESC_representation_nobound_top_cov_noabs_noSelfImage', 'hESC/ExpressionData.csv',
                                                 "training_pairshESC.txt",
                                                 "training_pairshESC.txtTF_divide_pos.txt", "hESC_geneName_map.txt",
                                                 TF_num=18,
                                                 TF_order_random=True, flag_load_split_batch_pos=True,
                                                 add_self_image=False,
                                                 cat_option="multi_channel_zhang", top_or_random_or_all="top_cov",
                                                 get_abs=False)
    if True:
        main_for_representation_single_cell_type('mHSC_GM_representation_nobound_top_cov_noabs_noSelfImage', 'mHSC-GM/ExpressionData.csv',
                                                 "training_pairsmHSC_GM.txt",
                                                 "training_pairsmHSC_GM.txtTF_divide_pos.txt", "mHSC_GM_geneName_map.txt",
                                                 TF_num=18,
                                                 TF_order_random=True, flag_load_split_batch_pos=True,
                                                 add_self_image=False,
                                                 cat_option="multi_channel_zhang", top_or_random_or_all="top_cov",
                                                 get_abs=False)


    #main_for_representation_single_cell_type('mHSC_combined_representation_nobound', 'mHSC_combined/ExpressionData.csv', "training_pairsmHSC_combined.txt",
     #                                        "training_pairsmHSC_combined.txtTF_divide_pos.txt", "mHSC_combined_geneName_map.txt",
      #                                       TF_num=18,TF_order_random=True,flag_load_split_batch_pos=True)

    #main_for_representation_single_cell_type('mHSC_E_representation_nobound', 'mHSC-E/ExpressionData.csv', "training_pairsmHSC_E.txt",
     #                                        "training_pairsmHSC_E.txtTF_divide_pos.txt", "mHSC_E_geneName_map.txt",TF_num=18,
      #                                       TF_order_random=True, flag_load_split_batch_pos=True)

    #main_for_representation_single_cell_type('mHSC_GM_representation_nobound', 'mHSC-GM/ExpressionData.csv',
     #                                        "training_pairsmHSC_GM.txt",
      #                                       "training_pairsmHSC_GM.txtTF_divide_pos.txt", "mHSC_GM_geneName_map.txt",
       #                                      TF_num=18,
        #                                     TF_order_random=True, flag_load_split_batch_pos=True)
    if False:
        main_for_representation_single_cell_type('mHSC_L_representation_nobound', 'mHSC-L/ExpressionData.csv',
                                                 "training_pairsmHSC_L.txt",
                                                 "training_pairsmHSC_L.txtTF_divide_pos.txt", "mHSC_L_geneName_map.txt",
                                                 TF_num=18,
                                                 TF_order_random=True, flag_load_split_batch_pos=True)

    #main_for_representation_single_cell_type('B_health_representation', 'health_B.csv', "training_pairshealth_B.txt",
     #                                       "training_pairshealth_B.txtTF_divide_pos.txt", "health_B_geneName_map.txt",TF_num=None)

    #main_for_representation_single_cell_type('B_health_representation_random', 'health_B.csv', "training_pairshealth_B.txt",TF_divide_pos_file=None,geneName_map_file= "health_B_geneName_map.txt",TF_num=None,
     #                                        flag_load_split_batch_pos=False)

    #main_for_representation_single_cell_type('B_mild_representation', 'mild_B.csv',
     #                                        "pair_for_predict_B.txt", TF_divide_pos_file=None,
      #                                       geneName_map_file="health_B_geneName_map.txt", TF_num=None,
       #                                      flag_load_split_batch_pos=False)

    #main_for_representation_single_cell_type('B_severe_representation', 'severe_B.csv',
     #                                        "pair_for_predict_B.txt", TF_divide_pos_file=None,
      #                                       geneName_map_file="health_B_geneName_map.txt", TF_num=None,
       #                                      flag_load_split_batch_pos=False)


    #for dropout########################3
    if False:
        drop_out_rate_list = [0.7,0.8,0.9]
        dropoutCutoffs = 0.3

        for drop_out_rate in drop_out_rate_list:
            #for drop_out_rate in range(1, 10):
            #drop_out_rate = drop_out_rate / 10
            path = str(int(100 * dropoutCutoffs)) + '-' + str(str(drop_out_rate))
            exprFile=path +'_ExpressionData.csv'
            main_for_representation_single_cell_type(path, exprFile,'gold_standard_for_TFdivide',TF_divide_pos_file='whole_gold_split_pos',
                                                     geneName_map_file="sc_gene_list.txt", TF_num=None,
                                                     flag_load_split_batch_pos=True)
    #######for cell Num##################
    if False:
        main_for_representation_single_cell_type("bonemarrow_cellNum100", 'bone_marrow_cell.h5', 'gold_standard_for_TFdivide',
                                                 TF_divide_pos_file='whole_gold_split_pos',
                                                 geneName_map_file="sc_gene_list.txt", TF_num=None,
                                                 flag_load_split_batch_pos=True,flag_load_from_h5=True,cellNum=100)


    ##########################
    # main_for_representation_single_cell_type('hHep_representation', 'hHep/ExpressionData.csv', "training_pairshHep.txt",
    #                                        "training_pairshHep.txtTF_divide_pos.txt", "hHep_geneName_map.txt",
    #                                       TF_num=18, TF_pairs_num_lower_bound=100, TF_pairs_num_upper_bound=10000, TF_order_random=True)

    #main_for_representation_single_cell_type('CD4_health_representation', 'health_CD4.csv', "training_pairshealth_CD4.txt",
     #                                        "training_pairshealth_CD4.txtTF_divide_pos.txt", "health_CD4_geneName_map.txt",
      #                                       TF_num=None,flag_load_split_batch_pos=False)
    #main_for_representation_single_cell_type('CD4_mild_representation', 'mild_CD4_test.csv',"training_pairshealth_CD4.txt","training_pairshealth_CD4.txtTF_divide_pos.txt","health_CD4_geneName_map.txt",TF_num=None, flag_load_split_batch_pos=False)

    #main_for_representation_single_cell_type('CD4_severe_representation', 'severe_CD4_test.csv',"training_pairshealth_CD4.txt","training_pairshealth_CD4.txtTF_divide_pos.txt","health_CD4_geneName_map.txt",TF_num=None,flag_load_split_batch_pos=False)


    #main_GENIE3_liver('hcc')
    #main_GENIE3_lung()
    #main_GENIE3_bonemarrow()
    #main_GENIE3_mesc()
    #main_arboreto_GENIE3_lung()
    #main_arboreto_GENIE3_liver('hcc')
    #main_liver_fiilter_gene_test('hcc')
    #main_liver_data_one_cell_type('hcc')
    pass


