
import numpy as np
import pandas as pd
import scipy
import scipy.sparse


class ChipSeq_data_convert:
    def __init__(self):
        self.start_position_to_gene={}
        self.start_position=[]

        self.promoter_region_start=[]
        self.promoter_region_end=[]
        self.promoter_chrom=[]

        self.single_cell_exp_set=[]

        self.peak_set=[]#TF to peak position chr_start_end, split by ,
        self.positive_pair=[]#TF to gene, split by ,

        #########
        self.expr = None
        self.geneNames = None
        self.geneIDs = None #for lung data
        self.corr_matrix = None
        self.filter_by_corr = None
        self.flag_filter=True
        self.flag_remove_corr_NA=False
        self.flag_negative_also_large_corr=False
        self.adj_matrix = None

        self.TF_list = []
        self.rows_for_TF = []
        self.filtered_TF_rows = []
        self.filtered_candidate_genes_rows = []


    def function_read_gtf(self, filename):
        #get the gene's start site, and get the promoter region as 10kb upstream and 1kb downstream of start site
        #by chromosome position order
        import gzip

        with gzip.open(filename, 'r') as fin:
            for line in fin:
                split_line = line.decode('utf-8').split('\t')
                if len(split_line)==9:
                    if split_line[2]=='gene': # transcript!
                        tmp=split_line[8].split("gene_name \"")
                        tmp2=tmp[1].split("\";")
                        geneName=tmp2[0]

                        promoter_region_start=int(split_line[3])-10000
                        promoter_region_end=int(split_line[3])+10
                        self.promoter_region_start.append(promoter_region_start)
                        self.promoter_region_end.append(promoter_region_end)
                        self.promoter_chrom.append('chr'+split_line[0])

                        chr_start=str(split_line[0]+'_'+str(split_line[3]))
                        self.start_position_to_gene[chr_start] = geneName
                        self.start_position.append(chr_start)
        self.promoter_region_end = np.asarray(self.promoter_region_end)
        self.promoter_region_start = np.asarray(self.promoter_region_start)
        self.promoter_chrom = np.asarray(self.promoter_chrom)
        print(len(self.promoter_region_start))


    def read_meta_cluster(self, filename):
        #first get the certain experiment, record the start and end postion of the peak
        #by chromosome position order,
        import gzip

        with gzip.open(filename, 'r') as fin:
            for line in fin:
                split_line = line.decode('utf-8').split('\t')
                print(len(split_line))
                if len(split_line) == 13: #13
                    exp_set=split_line[8].split(";")
                    for exp in exp_set:
                        if exp in self.single_cell_exp_set:
                            #0 chr
                            #1 start
                            #2 end
                            #5 TF title
                            #8 exp
                            self.find_the_gene_by_peak_position(split_line[5],split_line[0],split_line[1],split_line[2],exp)


    def read_macs_peak(self, filename,pvlue_minus_log10_cutoff=5, TF_title_col=12, line_length=15):#1. pvalue <10^-5, 2. pvalue< 10^-8, 3. pvalue<10^-5, qvalue<10^-5
        #first get the certain experiment, record the start and end postion of the peak
        #by chromosome position order,
        import gzip

        with gzip.open(filename, 'r') as fin:
            for line in fin:
                split_line = line.decode('utf-8').split('\t')
                if len(split_line) == line_length: #13
                    exp_set=split_line[8].split(";")
                    for exp in exp_set:
                        if exp in self.single_cell_exp_set:
                            #0 chr
                            #1 start
                            #2 end
                            #12 TF title
                            #8 exp
                            #3 -log(pvalue) > (5=-log10(0.00001))
                            if float(split_line[3])>pvlue_minus_log10_cutoff:
                                self.find_the_gene_by_peak_position(split_line[TF_title_col],split_line[0],split_line[1],split_line[2],exp)
                            else:
                                pass


    def intersection(self,lst1, lst2):
        # Use of hybrid method
        temp = set(lst2)
        lst3 = [value for value in lst1 if value in temp]
        return lst3

    def find_the_gene_by_peak_position(self,TF_title, peak_chr, peak_start, peak_end,exp):
        #record the index, find by order
        #find by
        indexes1 = np.where(self.promoter_region_start < int(peak_start))
        indexes2 = np.where(self.promoter_region_end > int(peak_start))
        indexes3 = np.where(self.promoter_chrom == peak_chr)

        indexes4 = np.where(self.promoter_region_start < int(peak_end))
        indexes5 = np.where(self.promoter_region_end > int(peak_end))

        set1 = set(indexes1[0])
        set2 = set(indexes2[0])
        set3 = set(indexes3[0])
        set4 = set(indexes4[0])
        set5 = set(indexes5[0])

        candidate_target_set1 = set1&set2&set3
        candidate_target_set2 = set3&set4&set5
        candidate_target_set = candidate_target_set1|candidate_target_set2
        for i in candidate_target_set:
            pos=self.start_position[i]
            gene=self.start_position_to_gene.get(pos)

            self.peak_set.append(TF_title+','+exp+','+peak_chr+'_'+peak_start+'_'+peak_end+','+pos+","+gene)
            if TF_title+','+gene not in self.positive_pair:
                self.positive_pair.append(TF_title+','+gene)
        #print(self.positive_pair)
        #print(self.peak_set)

    def output_positive_pair_set(self,outfile_peak_set, outfile_positive_pairs):
        np.savetxt(outfile_peak_set, self.peak_set, delimiter='\n',fmt='%s')
        np.savetxt(outfile_positive_pairs, self.positive_pair, delimiter='\n',fmt='%s')

    ##################################################


    def read_sparse_expr_table(self, sparse_matrix_file, row_file, index1, column_file, index2):
        row_info=pd.read_table(row_file,header='infer')
        geneNames=row_info.iloc[:,index1]
        print(geneNames)

        col_info=pd.read_table(column_file,header='infer')
        cellIDs=col_info.iloc[:,index2]
        print(cellIDs)

        sparse_matrix=pd.read_table(sparse_matrix_file,header='infer',delimiter=' ')

        from scipy import sparse
        I = np.asarray(sparse_matrix.iloc[:, 0])-1
        J = np.asarray(sparse_matrix.iloc[:, 1])-1
        V = np.asarray(sparse_matrix.iloc[:, 2])
        expr = sparse.csr_matrix((V, (I, J)), shape=(len(geneNames), len(cellIDs)))
        #need normalize? need complete
        return [expr, geneNames]

    def read_expr_table(self,expr_file):
        from scipy import sparse
        df=pd.read_table(expr_file,header='infer',index_col=0)
        geneNames=df.index
        print(geneNames)
        expr=sparse.csr_matrix(df.values)
        # need normalize? need complete
        return [expr, geneNames]

    def concat_matrix(self,expr1,geneName1, expr2,geneName2):
        #get intersection of gene and concat
        from scipy.sparse import hstack
        set1 = set(geneName1)
        print("len(set1)",len(set1))
        set2 = set(geneName2)
        print("len(set2)",len(set2))
        geneNames=list(set1 & set2)


        geneNames=np.asarray(geneNames)
        geneNames=geneNames.ravel()

        geneName1=np.asarray(geneName1)
        geneName2=np.asarray(geneName2)
        index_list1=[]
        index_list2=[]
        print('geneNames',geneNames)
        print("len(geneNames)",len(geneNames))

        for i in range(0,len(geneNames)):
            index1=np.where(geneName1==geneNames[i])
            if len(index1[0])>1:
                m=expr1.tocsr()[index1[0], :]
                #expr1.tocsr()[index1[0][0], :] = m.mean(axis=0)
                expr1.tocsr()[index1[0][0], :] = m.sum(axis=0)
                index_list1.append(index1[0][0])
            else:
                index_list1.append(index1[0][0])

            index2=np.where(geneName2==geneNames[i])
            if len(index2[0])>1:
                m = expr2.tocsr()[index2[0], :]
                expr2.tocsr()[index2[0][0], :] = m.sum(axis=0)
                index_list2.append(index2[0][0])
            else:
                index_list2.append(index2[0][0])

        print("index_list1",index_list1)
        select_ind1=np.asarray(index_list1)
        print("select_ind1",select_ind1)
        select_ind1=select_ind1.ravel()
        print("select_ind1",select_ind1)
        select_ind2=np.asarray(index_list2)
        select_ind2=select_ind2.ravel()
        X1=expr1.tocsr()[select_ind1,:]
        X2=expr2.tocsr()[select_ind2,:]
        print("shape X1",X1.shape)
        print("shape X2",X2.shape)
        expr=hstack((X1,X2))
        print("shape expr",expr.shape)
        return [expr,geneNames]

    def read_edge_weight(self): # just in case, for filter or for generate whole ********
        pass

    def read_ID_symbol_map_table(self): #currently no need. use gene symbol is enough. **************
        pass

    def sparse_corrcoef(self,A, B=None):
        from scipy import sparse
        if B is not None:
            A = sparse.vstack((A, B), format='csr')

        A = A.astype(np.float64)
        n = A.shape[1]

        # Compute the covariance matrix
        rowsum = A.sum(1)
        centering = rowsum.dot(rowsum.T.conjugate()) / n
        C = (A.dot(A.T.conjugate()) - centering) / (n - 1)

        # The correlation coefficients are given by
        # C_{i,j} / sqrt(C_{i} * C_{j})
        d = np.diag(C)
        coeffs = C / np.sqrt(np.outer(d, d))

        return coeffs

    def calculate_gene_corr(self,expr):####*********************
        #?????????????????is it necessary???
        coef=self.sparse_corrcoef(expr)
        print("coef.shape",coef.shape)
        return coef

    def get_index_by_geneName(self,genei):
        index=np.where(self.geneNames==genei)
        if len(index[0]) > 0:
            return index[0]
        else:
            return None

    def get_index_by_geneID(self,genei):
        index=np.where(self.geneIDs==genei)
        if len(index[0]) > 0:
            return index[0]
        else:
            return None

    def get_exp_by_geneName(self,genei):
        index = np.where(self.geneNames == genei)
        from scipy import sparse
        if len(index[0]) > 0:
            row = index[0]
            if isinstance(self.expr,sparse.csr_matrix):
                genei_exp = np.asarray(self.expr.tocsr()[row, :].todense())
            else:
                genei_exp = self.expr[row, :]
            return genei_exp.ravel()
        else:
            return None

    def read_positive_pairs_and_filter(self,filename, reverse_corr_filter=False,col0_get_by_name=True, col1_get_by_name=True):
        positive_pairs=pd.read_csv(filename,header=None)
        #change the positive pairs to the index in self.geneNames.
        V = []
        I = []
        J = []
        print("positive_pairs",positive_pairs)
        print("self.geneNames", self.geneNames)
        print("self.geneIDs", self.geneIDs)
        for i in range(0,positive_pairs.shape[0]):
            if col0_get_by_name:
                indexA=self.get_index_by_geneName(positive_pairs.iloc[i,0])
            else:
                indexA = self.get_index_by_geneID(positive_pairs.iloc[i, 0])
            if col1_get_by_name:
                indexB=self.get_index_by_geneName(positive_pairs.iloc[i,1])
            else:
                indexB = self.get_index_by_geneID(positive_pairs.iloc[i, 1])

            if indexA is not None and indexB is not None:
                V.append(1)
                I.append(indexA)
                J.append(indexB)
                print("find gene name in positive pair", positive_pairs.iloc[i, 0], positive_pairs.iloc[i, 1])
            else:
                if col0_get_by_name:
                    print("col0 by name")
                    indexA = self.get_index_by_geneName(positive_pairs.iloc[i, 0].lower())

                else:
                    print("col0 by ID")
                    indexA = self.get_index_by_geneID(positive_pairs.iloc[i, 0].lower())
                if col1_get_by_name:
                    print("col1 by name")
                    indexB = self.get_index_by_geneName(positive_pairs.iloc[i, 1].lower())
                else:
                    print("col1 by ID")
                    indexB = self.get_index_by_geneID(positive_pairs.iloc[i, 1].lower())

                if indexA is not None and indexB is not None:
                    V.append(1)
                    I.append(indexA)
                    J.append(indexB)
                    print("find gene name in positive pair", positive_pairs.iloc[i, 0].lower(), positive_pairs.iloc[i, 1].lower())
                else:
                    print("can not find gene name in positive pair", positive_pairs.iloc[i,0].lower(), positive_pairs.iloc[i,1].lower())

        V = np.asarray(V)
        V = V.ravel()
        I = np.asarray(I)
        I = I.ravel()
        J = np.asarray(J)
        J = J.ravel()
        print("V",V)
        print("I",I)
        print("J",J)

        #make positive pairs label as 1
        if self.flag_negative_also_large_corr:
            V = np.ones((len(I),))

        self.adj_matrix= scipy.sparse.csr_matrix((V, (I, J)), shape=(len(self.geneNames), len(self.geneNames)))#all 0, if positive pairs set as 1, an sparse matrix
        self.adj_matrix=self.adj_matrix.toarray()

        print("corr_matrix",self.corr_matrix)
        print("adj_matrix",self.adj_matrix)

        if self.flag_filter:
            if self.flag_remove_corr_NA:
                self.adj_matrix = np.where(np.isnan(self.corr_matrix), 0,self.adj_matrix)
            if self.filter_by_corr:
                if self.flag_negative_also_large_corr:
                    self.adj_matrix = np.where(np.abs(self.corr_matrix) > self.corr_cutoff, self.adj_matrix, 2)

                else:

                    if reverse_corr_filter:
                        self.adj_matrix=np.where(np.abs(self.corr_matrix) > self.corr_cutoff, 0, self.adj_matrix)
                    else:
                        self.adj_matrix = np.where(np.abs(self.corr_matrix) > self.corr_cutoff, self.adj_matrix, 0)
        return self.adj_matrix

    def output_positive_pairs_according_to_adj_matrix(self,out_filename,lowerCase=False):
        if self.flag_negative_also_large_corr:
            result = np.where(self.adj_matrix == 1)
        else:
            result = np.where(self.adj_matrix!=0)
        print("result",result)
        rows=result[0]
        cols=result[1]
        print("rows",rows)
        geneA_names=self.geneNames[rows]
        geneB_names=self.geneNames[cols]
        print("geneA_names",geneA_names)
        print("geneB_names", geneB_names)
        if lowerCase:
            for i in range(0,len(geneA_names)):
                geneA_names[i]=geneA_names[i].lower()
                geneB_names[i]=geneB_names[i].lower()
        df=pd.DataFrame(geneA_names,geneB_names)
        df.to_csv(out_filename,sep="\t",header=False)

    def output_training_pairs_according_to_adj_matrix(self,out_filename,lowerCase=True, random_training_pairs=False):
        # one positive pair 1, one positive pair in other direction 2, one negative pair 0
        # since we have get genes, we can generate an adjacency matrix for TF to all genes, and sample pairs of TF to random genes
        # all genes in lower case.
        from random import shuffle
        TF_divide_pos=[]
        TF_divide_pos.append(0)
        #get_positive:
        if self.flag_negative_also_large_corr:
            result = np.where(self.adj_matrix == 1)
        else:
            result = np.where(self.adj_matrix != 0)
        positive_rows = result[0]
        positive_cols = result[1]
        print("positive_rows",positive_rows)
        print("len(positive_rows)",len(positive_rows))
        if random_training_pairs:
            indexs = np.arange(len(positive_rows))
            shuffle(indexs)
            positive_rows = positive_rows[indexs]
            positive_cols = positive_cols[indexs]

        negative_rows=[]
        negative_cols=[]
        negative_rows=np.asarray(negative_rows,dtype=int)
        negative_cols=np.asarray(negative_cols,dtype=int)

        reordered_positive_rows=[]
        reordered_positive_cols=[]

        TFs_rows=np.unique(positive_rows)
        print("TFs_rows",TFs_rows)
        print("len(TFs_rows)", len(TFs_rows))
        # get negative:
        for i in range(0,len(TFs_rows)):
            row_TFi=TFs_rows[i]
            tmp=np.where(positive_rows==row_TFi)
            num_pairs_TFi=len(tmp[0])


            # get the negative position for this TF


            tmp2 = np.where(self.adj_matrix[row_TFi] == 0)
            cols = tmp2[0]
            # random the negative position and get the equal negative pairs (same as positive pairs for this TF)
            indexs = np.arange(len(cols))
            shuffle(indexs)
            shuffled_negative_cols = cols[indexs]
            print("counts where =0,",len(cols))
            print("counts num_pairs_TFi,",num_pairs_TFi)

            if len(cols)<num_pairs_TFi:

                print("counts where =0 less than counts num_pairs_TFi!!!!!!!!!!")

                last_pos = TF_divide_pos[len(TF_divide_pos) - 1]
                TF_divide_pos.append(last_pos + len(cols) * 3)


                if self.flag_negative_also_large_corr:
                    tmp1 = np.where(self.adj_matrix[row_TFi] == 1)
                else:
                    tmp1 = np.where(self.adj_matrix[row_TFi] != 0)

                reordered_positive_rows = np.concatenate(
                    (reordered_positive_rows, np.full((len(cols),), TFs_rows[i])))

                positive_cols_in_TFi = tmp1[0]
                index2=np.arange(len(positive_cols_in_TFi))
                shuffle(index2)
                shuffled_positive_cols_in_TFi=positive_cols_in_TFi[index2]
                filtered_positive_cols_in_TFi=shuffled_positive_cols_in_TFi[0:len(cols)]
                reordered_positive_cols = np.concatenate((reordered_positive_cols, filtered_positive_cols_in_TFi))

            else:

                last_pos = TF_divide_pos[len(TF_divide_pos) - 1]
                TF_divide_pos.append(last_pos + num_pairs_TFi * 3)

                if self.flag_negative_also_large_corr:
                    tmp1 = np.where(self.adj_matrix[row_TFi] == 1)
                else:
                    tmp1 = np.where(self.adj_matrix[row_TFi] != 0)
                reordered_positive_rows = np.concatenate(
                    (reordered_positive_rows, np.full((num_pairs_TFi,), TFs_rows[i])))
                reordered_positive_cols = np.concatenate((reordered_positive_cols, tmp1[0]))

            selected_negative_cols=shuffled_negative_cols[0:num_pairs_TFi]

            negative_cols=np.concatenate((negative_cols,selected_negative_cols))

            rows = np.full((len(selected_negative_cols),), TFs_rows[i])
            negative_rows = np.concatenate((negative_rows, rows))
            print("TFi",i,"TF", self.geneNames[row_TFi], "len(add negative cols)", len(selected_negative_cols))
            print("len(combined negative pairs", len(negative_cols), len(negative_rows))

        if random_training_pairs:
            indexs = np.arange(len(negative_rows))
            shuffle(indexs)
            select_index=indexs[0:len(positive_rows)]
            negative_rows = negative_rows[select_index]
            negative_cols = negative_cols[select_index]

        reordered_positive_rows=np.asarray(reordered_positive_rows, dtype=int)
        reordered_positive_cols=np.asarray(reordered_positive_cols,dtype=int)
        geneA_names_positive = self.geneNames[reordered_positive_rows]
        geneB_names_positive = self.geneNames[reordered_positive_cols]
        geneA_names_negative = self.geneNames[negative_rows]
        geneB_names_negative = self.geneNames[negative_cols]
        #label_negative = np.zeros(len(geneA_names_negative, 1))
        print("len(geneA_names_negative)",len(geneA_names_negative))
        print("len(geneB_names_negative)", len(geneA_names_negative))
        print("len(geneA_names_positive)", len(geneA_names_positive))
        print("len(geneB_names_positive)", len(geneB_names_positive))

        out_string_list = []
        for i in range(0,len(geneA_names_positive)):
            if lowerCase:
                geneA_names_positive[i] = geneA_names_positive[i].lower()
                geneB_names_positive[i] = geneB_names_positive[i].lower()
                geneA_names_negative[i] = geneA_names_negative[i].lower()
                geneB_names_negative[i] = geneB_names_negative[i].lower()
            positive_pair=str(geneA_names_positive[i])+'\t'+str(geneB_names_positive[i])+'\t'+str(1)
            reverse_pair = str(geneB_names_positive[i]) + '\t' + str(geneA_names_positive[i]) + '\t' + str(2)
            negative_pair = str(geneA_names_negative[i]) + '\t' + str(geneB_names_negative[i]) + '\t' + str(0)
            out_string_list.append(positive_pair)
            out_string_list.append(reverse_pair)
            out_string_list.append(negative_pair)

        TF_divide_pos=np.asarray(TF_divide_pos)
        np.savetxt(out_filename+'TF_divide_pos.txt', TF_divide_pos, fmt="%d",delimiter='\n')
        out_string_list=np.asarray(out_string_list)
        np.savetxt(out_filename,out_string_list, fmt="%s", delimiter='\n')

    def output_geneName_to_geneName_map(self,out_filename,lowerCase=True):
        if lowerCase:
            for i in range(0,len(self.geneNames)):
                self.geneNames[i] = self.geneNames[i].lower()
        if self.geneIDs is None:
            df=pd.DataFrame(self.geneNames,self.geneNames)
            df.to_csv(out_filename,sep="\t",header=False)
        else:
            df = pd.DataFrame({'geneName':self.geneNames, 'geneID':self.geneIDs})
            df.to_csv(out_filename, sep="\t", header=False,index=False)

    def save_rpkm_and_geneNames(self,out_filename_rpkm,out_filename_geneNames):
        scipy.sparse.save_npz(out_filename_rpkm, self.expr)
        np.savetxt(out_filename_geneNames, self.geneNames, delimiter="\n", fmt="%s")




    def load_combined(self):
        import scipy.sparse
        sparse_matrix = scipy.sparse.load_npz('data_combined_liver/expr_sparse_matrix.npz')
        self.expr = sparse_matrix
        df = pd.read_table("geneNames.txt", header=None)
        tmp = df.values
        self.geneNames = np.asarray(tmp[:, 0])
        self.geneNames = self.geneNames.ravel()
        for i in range(0,len(self.geneNames)):
            self.geneNames[i] = self.geneNames[i].lower()


    def work_filter_positive_pair_server_combined(self):
        self.filter_by_corr=True
        self.flag_filter=True
        self.corr_cutoff=0.01
        if self.flag_filter:
            #self.load_initial_expr()
            self.load_combined()
            self.corr_matrix=self.calculate_gene_corr(self.expr)
        else:
            df = pd.read_table("geneNames.txt", header=None)
            tmp = df.values
            self.geneNames = np.asarray(tmp[:, 0])
            self.geneNames = self.geneNames.ravel()
        # read positive pair
        # filter positive pairs
        self.read_positive_pairs_and_filter("positive_pairs.csv")
        # output filtered positive pairs
        self.output_positive_pairs_according_to_adj_matrix("positive_pairs_filter_TFneg.txt")
        #output training pairs
        self.output_training_pairs_according_to_adj_matrix("training_pairs_filter0.01_random_TFneg.txt")

    def filter_cell_from_cell_info_file(self,filename,select_str):
        df=pd.read_table(filename,header=None,dtype=str)
        X=df.iloc[:,0]
        print('X',X)
        index=np.where(X==select_str)
        index=index[0]
        print('index',index)
        print('expr.shape',self.expr.shape)
        self.expr=self.expr.tocsr()[:,index]
        print('expr.shape', self.expr.shape)
        pass


    def get_TF_name_list(self,TF_file,index):
        df = pd.read_table(TF_file,header=None,dtype=str,delimiter=',')
        print("df",df)
        print("shape(df)",df.shape)
        TF_list=df.iloc[:,index]
        for i in range(0,len(TF_list)):
            if TF_list[i] in self.geneNames:
                self.TF_list.append(TF_list[i])
                row_TF=np.where(self.geneNames==TF_list[i])
                row_TF=row_TF[0]
                self.rows_for_TF.append(row_TF[0])

    def filter_TF_by_expr(self):
        genes_expr_sum=self.expr.tocsr().sum(axis=1)
        genes_expr_sum=np.asarray(genes_expr_sum)
        genes_expr_sum=genes_expr_sum.ravel()
        rows_larger_zero=np.where(genes_expr_sum>0)
        rows_larger_zero=rows_larger_zero[0]
        set1=set(self.rows_for_TF)
        set2=set(rows_larger_zero)
        filtered_TF_rows=set1&set2
        self.filtered_TF_rows=list(filtered_TF_rows)
        print('len(select_TF_rows',len(self.filtered_TF_rows))

    def filter_candidate_genes_by_expr(self):
        genes_expr_sum = self.expr.tocsr().sum(axis=1)
        genes_expr_sum = np.asarray(genes_expr_sum)
        genes_expr_sum = genes_expr_sum.ravel()
        rows_larger_zero = np.where(genes_expr_sum > 0)
        rows_larger_zero = rows_larger_zero[0]
        self.filtered_candidate_genes_rows = rows_larger_zero
        print('len(filtered_candidate_genes_rows',len(self.filtered_candidate_genes_rows))


    def get_pairs_for_for_TFs(self,flag_filter_by_corr=False,corr_cutoff=0.01):
        #adj with direction,
        #from self.filtered_TF_rows, self.filtered_candidate_genes_rows, generate adj
        self.adj_matrix=np.zeros((len(self.geneNames),len(self.geneNames)))
        #self.filtered_candidate_genes_rows=np.asarray(self.filtered_candidate_genes_rows)
        #self.filtered_TF_rows=np.asarray(self.filtered_TF_rows)
        #xx=self.filtered_TF_rows.reshape(len(self.filtered_TF_rows),1)
        #self.adj_matrix[xx, self.filtered_candidate_genes_rows] = 1
        self.adj_matrix[self.filtered_TF_rows,:]=1
        A=range(0,len(self.geneNames))
        set1=set(A)
        set2=set(self.filtered_candidate_genes_rows)
        filter_out_genes_cols =set1.difference(set2)
        filter_out_genes_cols =list(filter_out_genes_cols)
        self.adj_matrix[:,filter_out_genes_cols]=0
        np.fill_diagonal(self.adj_matrix, 0)
        if flag_filter_by_corr:
            self.corr_matrix = self.calculate_gene_corr(self.expr)
            # mask the adj if not satisfy cutoff
            self.adj_matrix = np.where(np.abs(self.corr_matrix) > corr_cutoff, self.adj_matrix, 0)

    def output_predict_pairs_according_to_adj_matrix(self,out_filename,lowerCase=False):
        result = np.where(self.adj_matrix!=0)
        print("result",result)
        rows=result[0]
        cols=result[1]
        print("rows",rows)
        geneA_names=self.geneNames[rows]
        geneB_names=self.geneNames[cols]
        values=np.full((geneA_names.shape[0],),3)
        print("geneA_names",geneA_names)
        print("geneB_names", geneB_names)
        if lowerCase:
            for i in range(0,len(geneA_names)):

                geneA_names[i]=geneA_names[i].lower()
                geneB_names[i]=geneB_names[i].lower()
                #values[i]=str(3)
        print("shape geneA_names",geneA_names.shape)
        print("shape geneB_names",geneB_names.shape)
        print("shape values",values.shape)
        df=pd.DataFrame({'geneA':geneA_names,'geneB':geneB_names,'value':values})
        df.to_csv(out_filename,sep="\t",header=False,index=False)

    def work_generate_gene_pairs_list_for_prediction(self,select_str='healthy_liver',corr_cutoff=0.2): #need complete
        #output only candidate edges
        #3. by nmf corr cutoff??
        #generate list according to format like training pairs with labels as 3(to be predict)
        self.load_combined()
        self.filter_cell_from_cell_info_file('combined_study_info.txt', select_str)
        self.get_TF_name_list("TF_ID_symbol_map.csv", 1)

        self.filter_TF_by_expr()
        self.filter_candidate_genes_by_expr()
        self.get_pairs_for_for_TFs(flag_filter_by_corr=True,corr_cutoff=corr_cutoff)
        self.output_predict_pairs_according_to_adj_matrix(select_str+str(corr_cutoff)+'_pairs_for_predict.txt',lowerCase=True)

    #####################################################################


    def get_geneIDs_from_geneName_for_GTEx_positive(self, geneNames, ID_to_name_mapfile,id_col,name_col, lower_flag=False):
        geneIDs = []
        map_table = pd.read_csv(ID_to_name_mapfile, header=None)
        map_ID_to_name = {}
        map_name_to_ID = {}
        for i in range(0, map_table.shape[0]):
            idi = map_table.iloc[i, id_col]
            namei = map_table.iloc[i, name_col]
            if lower_flag:
                idi=idi.lower()
                namei=namei.lower()
            map_ID_to_name[idi] = namei
            map_name_to_ID[namei] = idi

        print(map_name_to_ID)
        for i in range(0, len(geneNames)):
            geneNamesi=geneNames[i]
            geneIDs.append(map_name_to_ID.get(geneNamesi))
        geneIDs = np.asarray(geneIDs)
        geneIDs = geneIDs.ravel()
        return geneIDs


    def get_geneNames_and_geneIDs_lower(self):
        if self.geneNames is not None:
            for i in range(0, len(self.geneNames)):
                self.geneNames[i]=self.geneNames[i].lower()

        if self.geneIDs is not None:
            for i in range(0, len(self.geneIDs)):
                self.geneIDs[i]=self.geneIDs[i].lower()



    def load_single_cell_type_expr(self, expr_file):
        if expr_file.endswith('.txt'):
            df = pd.read_table(expr_file, header='infer', index_col=0)
        else:
            df = pd.read_csv(expr_file,header='infer',index_col=0)

        self.expr = df.values
        print("expr.shape", self.expr.shape)

        self.geneNames = df.index
        self.geneNames = np.asarray(self.geneNames)
        self.geneNames = self.geneNames.ravel()
        for i in range(0, len(self.geneNames)):
            self.geneNames[i] = self.geneNames[i].lower()


    def work_filter_positive_pair_single_cell_type(self, positive_pair_file, expr_file, label):
        self.filter_by_corr = False
        self.flag_remove_corr_NA = False
        self.flag_filter = False
        self.corr_cutoff = 0.1
        self.flag_negative_also_large_corr = False
        if self.flag_filter:
            self.load_single_cell_type_expr(expr_file)

            self.corr_matrix = np.corrcoef(self.expr)
            print("shape corr_matrix", self.corr_matrix.shape)
        else:
            self.load_single_cell_type_expr(expr_file)
        # read positive pair
        # filter positive pairs
        self.get_geneNames_and_geneIDs_lower()

        self.read_positive_pairs_and_filter(positive_pair_file, col0_get_by_name=True,
                                            col1_get_by_name=True)
        # output filtered positive pairs
        self.output_positive_pairs_according_to_adj_matrix("positive_pairs"+label+".txt")
        self.output_geneName_to_geneName_map(label+"_geneName_map.txt")
        # output training pairs
        self.output_training_pairs_according_to_adj_matrix("training_pairs"+label+".txt")


    def corr_only_nonzero(self,geneA_exp,geneB_exp):
        nonzero_indexA=np.where(geneA_exp!=0)
        nonzero_indexB=np.where(geneB_exp!=0)
        set1=set(nonzero_indexA[0])
        set2=set(nonzero_indexB[0])
        cells_for_corr=set1&set2
        if len(cells_for_corr)>0:
            print("cells for corr",cells_for_corr)
            cells_for_corr=list(cells_for_corr)
            vecA = geneA_exp[cells_for_corr]
            vecB = geneB_exp[cells_for_corr]
            corr = np.corrcoef(vecA,vecB)

            return corr[0,1]
        else:
            return 2

    def calculate_corr_for_pairs(self, train_pair_filename,out_file,flag_only_nonzero=True):
        #-2 for nan, # 2 for no overlap nonzero
        import math
        train_pairs = pd.read_table(train_pair_filename, header=None)
        print("train pairs", train_pairs)
        print("self.geneNames", self.geneNames)
        for i in range(0, train_pairs.shape[0]):
            geneA_exp = self.get_exp_by_geneName(train_pairs.iloc[i, 0])
            geneB_exp = self.get_exp_by_geneName(train_pairs.iloc[i, 1])

            if geneA_exp is not None and geneB_exp is not None:
                print("find gene name in positive pair", train_pairs.iloc[i, 0], train_pairs.iloc[i, 1])
            else:
                geneA_exp = self.get_exp_by_geneName(train_pairs.iloc[i, 0].lower())
                geneB_exp = self.get_exp_by_geneName(train_pairs.iloc[i, 1].lower())

            if flag_only_nonzero:
                corr=self.corr_only_nonzero(geneA_exp,geneB_exp)
            else:
                corr=np.corrcoef(geneA_exp,geneB_exp)
                corr=corr[0,1]
            if math.isnan(corr):
                corr=-2
            print(train_pairs.iloc[i, 0].lower(),train_pairs.iloc[i, 1].lower(),corr)
            train_pairs.iloc[i,2]=corr
        train_pairs.to_csv(out_file,sep="\t",header=False,index=False)


    def work_chipSeq_to_positive_pair(self,tissue='liver',species="Homo_sapiens"):
        self.initialize_exp_to_TF_set(tissue=tissue)
        if species=='Homo_sapiens':
            self.function_read_gtf("Homo_sapiens.GRCh38.99.gtf.gz")
            #self.read_meta_cluster("Homo_sapiens_meta_clusters.interval.gz")
            self.read_macs_peak("Homo_sapiens_macs2_peaks.interval.gz", 8)
        elif species=="Mus_musculus":
            self.function_read_gtf("Mus_musculus.GRCm38.100.gtf.gz")
            # self.read_meta_cluster("Homo_sapiens_meta_clusters.interval.gz")
            self.read_macs_peak("Mus_musculus_macs2_peaks.interval.gz", 8)
        elif species=="Drosophila_melanogaster":
            self.function_read_gtf("Drosophila_melanogaster.BDGP6.28.100.gtf.gz")
            # self.read_meta_cluster("Homo_sapiens_meta_clusters.interval.gz")
            self.read_macs_peak("Drosophila_melanogaster_macs2_peaks.interval.gz", 8,TF_title_col=11,line_length=14)
        self.output_positive_pair_set(tissue+"_macs_peak_set_pvalue_e10_8.csv", tissue+"_macs_positive_pairs__pvalue_e10_8.csv")




    def initialize_exp_to_TF_set(self, tissue='B_cell'):#for primary  CD4+ T-cells
        if tissue=='liver':
            self.single_cell_exp_set = ['EXP047717', 'EXP057994', 'EXP031438', 'EXP031441', 'EXP031443', 'EXP031445',
                                        'EXP031447',
                                        'EXP047494', 'EXP047495', 'EXP047496', 'EXP047716', 'EXP047717', 'EXP057994',
                                        'EXP057995',
                                        'EXP057996', 'EXP057997']
        elif tissue=='CD4':#human primary CD4+ T-cell
            self.single_cell_exp_set=['EXP054902','EXP054903','EXP054904','EXP054905','EXP054897','EXP054898','EXP054899','EXP054900','EXP049025']

        elif tissue=='lung':
            self.single_cell_exp_set = ['EXP048466','EXP048467','EXP048592','EXP048593','EXP049230','EXP049231',
                                        'EXP049232','EXP049233','EXP049234','EXP049235','EXP049236','EXP049237',
                                        'EXP049242','EXP049243','EXP049244','EXP049245','EXP049246','EXP049247','EXP054265']
        ###for Mus musculus
        elif tissue=='oligodendrocyte':
            self.single_cell_exp_set = ['EXP055948', 'EXP050010','EXP050011', 'EXP050012']

        elif tissue=='neural_stem_cells':
            self.single_cell_exp_set = ['EXP055947', 'EXP056081', 'EXP056080', 'EXP056080']

        elif tissue=='embryonic_cortex':
            self.single_cell_exp_set = ['EXP051224','EXP051220', 'EXP051225', 'EXP051221', 'EXP051226',
                                        'EXP051222', 'EXP051227', 'EXP051223', 'EXP054745', 'EXP054746']

        elif tissue=='CamkIIa-positive nuclei':
            self.single_cell_exp_set = ['EXP053537', 'EXP053535', 'EXP053538', 'EXP053536']

        elif tissue=='brain':
            self.single_cell_exp_set = ['EXP050013', 'EXP055948', 'EXP050010', 'EXP050011', 'EXP050012',
                                        'EXP059274', 'EXP059275', 'EXP055947']

        elif tissue=='Drosophila_eye_disks':
            self.single_cell_exp_set = ['EXP045482', 'EXP045785','EXP045786', 'EXP045811', 'EXP045812',
                                        'EXP045813', 'EXP045882', 'EXP045475',  'EXP045881']

        elif tissue=='Drosophila_ovary':
            self.single_cell_exp_set = ['EXP045465', 'EXP045466', 'EXP045463', 'EXP045464']

        elif tissue=='B_cell':
            self.single_cell_exp_set = ['EXP058120', 'EXP058121', 'EXP058126', 'EXP058127',
                                        'EXP000756', 'EXP000757', 'EXP000758', 'EXP000759',
                                        'EXP000760', 'EXP000761', 'EXP000762', 'EXP000763',
                                        'EXP000764', 'EXP000765', 'EXP000766', 'EXP000767',
                                        'EXP000768', 'EXP000769', 'EXP000770', 'EXP000771',
                                        'EXP000772', 'EXP000773', 'EXP000774', 'EXP000775']



def main_single_cell_type_chipseq_to_positive_pair():
    tcs = ChipSeq_data_convert()
    tcs.work_chipSeq_to_positive_pair(tissue='B_cell')
    

def main_single_cell_type_filter_positive_pair():
    tcs = ChipSeq_data_convert()
    tcs.work_filter_positive_pair_single_cell_type(positive_pair_file="B_cell_macs_positive_pairs__pvalue_e10_8.csv",
                                                   expr_file="health_B.csv", label="health_B")

if __name__ == '__main__':
    main_single_cell_type_chipseq_to_positive_pair()
    main_single_cell_type_filter_positive_pair()
