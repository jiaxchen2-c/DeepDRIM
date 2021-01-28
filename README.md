# DeepDRIM


DeepDRIM is develop to consider the neighbor images and is improved from a previous model CNNC https://github.com/xiaoyeye/CNNC.


Dependency:   python 3, packages:


## Data


We test DeepDRIM at the following eight cell type.
bone marrow-derived macrophages, 
dendritic cells,
mESC(1): IB10 mouse, embryonic stem cells,
hESC: human embryonic stem cells,
mESC(2): 5G6GR mouse embryonic stem cells,
mHSC(E): mouse hematopoietic stem cell lines of erythroid lineage,
mHSC(GM): mouse hematopoietic stem cell lines of granulocyte-macrophage lineage,
mHSC(L): mouse hematopoietic stem cell lines of lymphoid lineage.


Benchmark and processed gene expression profiles for bone marrow-derived macrophages, dendritic cells, mESC(1) are availabel from https://github.com/xiaoyeye/CNNC. 
Benchmark and processed gene expression profiles for hESC, mESC(2), mHSC(E), mHSC(GM), mHSC(L) are availabel at https://doi.org/10.5281/zenodo.3378975.

The benchmark provide the pairs with positive labels. We randomly select same number of pairs with negative labels as the positive labels, and generate the training pair file. 

To study B cells in COVID-19, we generate gold standard of B cells based on ChIP-seq experiment from GTRD (https://gtrd.biouml.org) database ChIP-seq peaks (MACS2) files (Homo_sapiens_macs2_peaks.interval.gz) and corresponding gtf file (Homo_sapiens.GRCh38.99.gtf.gz). Then we search corresponding experiment ID by keyword in the GTRD website (http://gtrd20-06.biouml.org/bioumlweb/#).

 
  

Readme for data folder....!!!!!!!!!!!!!!!



## TASK 1, evaluate DeepDRIM in eight cell line



### STEP 1: Generate input for DeepDRIM

Code: generate_input_realdata.py

input: Gene expression profile and the benchmark, etc.

parameters:

out_dir: Indicate the path for output.
expr_file: The file of the gene expression profile. Can be h5 or csv file, the format please refer the example data.
pairs_for_predict_file: The file of the training gene pairs and their labels.
geneName_map_file: The file to map the name of gene in expr_file to the pairs_for_predict_file
flag_load_from_h5: Is the expr_file is a h5 file. True or False.
flag_load_split_batch_pos: Is there a file that indicate the position in pairs_for_predict_file to divide pairs into different TFs.
TF_divide_pos_file: File that indicate the position in pairs_for_predict_file to divide pairs into different TFs.
TF_num: To generate representation for this number of TFs. Should be a integer that equal or samller than the number of TFs in the pairs_for_predict_file.
TF_order_random: If the TF_num samller than the number of TFs in the pairs_for_predict_file, we need to indicate TF_order_random, if TF_order_random=True, then the code will generate representation for randomly selected TF_num TFs.

command for each cell type:

python3 generate_input_realdata.py -out_dir code_test -expr_file bone_marrow_cell.h5 -pairs_for_predict_file gold_standard_for_TFdivide -geneName_map_file sc_gene_list.txt -flag_load_from_h5 True -flag_load_split_batch_pos True -TF_divide_pos_file whole_gold_split_pos -TF_num 13

python3 generate_input_realdata.py -out_dir code_test -expr_file mesc_cell.h5 -pairs_for_predict_file gold_standard_mesc_whole.txt -geneName_map_file mesc_sc_gene_list.txt -flag_load_from_h5 True -flag_load_split_batch_pos True -TF_divide_pos_file mesc_divideTF_pos.txt -TF_num 38

python3 generate_input_realdata.py -out_dir code_test -expr_file dendritic_cell.h5 -pairs_for_predict_file gold_standard_dendritic_whole.txt -geneName_map_file sc_gene_list.txt -flag_load_from_h5 True -flag_load_split_batch_pos True -TF_divide_pos_file dendritic_divideTF_pos -TF_num 16


python3 generate_input_realdata.py -out_dir code_test -expr_file hESC/ExpressionData.csv -pairs_for_predict_file training_pairshESC.txt -geneName_map_file hESC_geneName_map.txt -flag_load_from_h5 False -flag_load_split_batch_pos True -TF_divide_pos_file training_pairshESC.txtTF_divide_pos.txt -TF_num 18 -TF_order_random True

python3 generate_input_realdata.py -out_dir code_test -expr_file mESC/ExpressionData.csv -pairs_for_predict_file training_pairsmESC.txt -geneName_map_file mESC_geneName_map.txt -flag_load_from_h5 False -flag_load_split_batch_pos True -TF_divide_pos_file training_pairsmESC.txtTF_divide_pos.txt -TF_num 18 -TF_order_random True

python3 generate_input_realdata.py -out_dir code_test -expr_file mHSC-E/ExpressionData.csv -pairs_for_predict_file training_pairsmHSC_E.txt -geneName_map_file mHSC_E_geneName_map.txt -flag_load_from_h5 False -flag_load_split_batch_pos True -TF_divide_pos_file training_pairsmHSC_E.txtTF_divide_pos.txt -TF_num 18 -TF_order_random True

python3 generate_input_realdata.py -out_dir code_test -expr_file mHSC-GM/ExpressionData.csv -pairs_for_predict_file training_pairsmHSC_GM.txt -geneName_map_file mHSC_GM_geneName_map.txt -flag_load_from_h5 False -flag_load_split_batch_pos True -TF_divide_pos_file training_pairsmHSC_GM.txtTF_divide_pos.txt -TF_num 18 -TF_order_random True

python3 generate_input_realdata.py -out_dir code_test -expr_file mHSC-L/ExpressionData.csv -pairs_for_predict_file training_pairsmHSC_L.txt -geneName_map_file mHSC_L_geneName_map.txt -flag_load_from_h5 False -flag_load_split_batch_pos True -TF_divide_pos_file training_pairsmHSC_L.txtTF_divide_pos.txt -TF_num 18 -TF_order_random True


example output:


x file: the representation of genes' expression file, use as the input of the model.
y file: the label for the corresponding pairs.
z file: indicate the gene name for each pair.
version0: The x file only include the primary image of the gene pair, can be used as input for model CNNC.
version11: The x file include the primary images and neighbor images for each gene pair, can be used as input for model DeepDRIM.




### STEP 2: TF-aware three-fold Cross-validation for DeepDRIM

code: DeepDRIM.py

input: the output of the STEP 1.

parameters:

num_batches: Since in STEP 1, we divide training pairs by TFs, and representation for one TF is included in one batch. Here the num_batches should be the number of TF or the number of x file (in version11 folder generated in the last step).
data_path: The path that includes x file, y file and z file, which is generated in the last step.
output_dir: Indicate the path for output.
cross_validation_fold_divide_file: A file that indicate how to divide the x file into three-fold. See example data XXXX. The file include three line, each line list the ID of the x files for the folder (split by ',').

to_predict: True or False. Default is False, then the code will do cross-validation evaluation. If set to True, we need to indicate weight_path for a trained model and the code will do prediction based on the trained model.
weight_path: The path for a trained model.



command example:

python3 DeepDRIM.py -num_batches 13 -data_path boneMarrow/version11/ -output_dir boneMarrow -cross_validation_fold_divide_file cross_validation_fold_divide.txt


python3 DeepDRIM.py -num_batches 18 -data_path mHSC_L_representation_nobound/version11/ -output_dir mHSC_L_test -cross_validation_fold_divide_file cross_validation_fold_divide2.txt

python3 DeepDRIM.py -to_predict True -num_batches 18 -data_path  mHSC_L_representation_nobound/version11/ -output_dir predict_test/ -weight_path keras_cnn_trained_model_shallow.h5



## TASK 2, construct GRN in specific cell type use DeepDRIM


### STEP 1: Generate ChIP-seq gold standard for specific cell type. 

Example: B cell
search corresponding experiment ID by keyword in (http://gtrd20-06.biouml.org/bioumlweb/#)
Experiment ID: 'EXP058120', 'EXP058121', 'EXP058126', 'EXP058127',
                                        'EXP000756', 'EXP000757', 'EXP000758', 'EXP000759',
                                        'EXP000760', 'EXP000761', 'EXP000762', 'EXP000763',
                                        'EXP000764', 'EXP000765', 'EXP000766', 'EXP000767',
                                        'EXP000768', 'EXP000769', 'EXP000770', 'EXP000771',
                                        'EXP000772', 'EXP000773', 'EXP000774', 'EXP000775


Prepare data:
Homo_sapiens.GRCh38.99.gtf.gz
Homo_sapiens_macs2_peaks.interval.gz
experiment ID

1. Run:
GTRD_chipSeq_data_convert.py -> main_single_cell_type_chipseq_to_positive_pair

input:
Homo_sapiens.GRCh38.99.gtf.gz
Homo_sapiens_macs2_peaks.interval.gz

parameters: 
tissue = 'B_cell'

in the ChipSeq_data_convert.initialize_exp_to_TF_set() for corrsponding tissue or cell type,
to set (for B cell):
ChipSeq_data_convert.single_cell_exp_set = ['EXP058120', 'EXP058121', 'EXP058126', 'EXP058127',
                                        'EXP000756', 'EXP000757', 'EXP000758', 'EXP000759',
                                        'EXP000760', 'EXP000761', 'EXP000762', 'EXP000763',
                                        'EXP000764', 'EXP000765', 'EXP000766', 'EXP000767',
                                        'EXP000768', 'EXP000769', 'EXP000770', 'EXP000771',
                                        'EXP000772', 'EXP000773', 'EXP000774', 'EXP000775']

cut off of pvalue is set to 1E-8.


output: 'B_cell_macs_positive_pairs__pvalue_e10_8'  

2. Run:
GTRD_chipSeq_data_convert.py -> main_single_cell_type_filter_positive_pair

Input: 'B_cell_macs_positive_pairs__pvalue_e10_8' 

health_B.csv (expression profile file for filter gene) 

label=health_B

output: 

positive_pairshealth_B_cell.txt
health_B_geneName_map.txt
training_pairshealth_B.txt


Then run STEP1-2 in TASK 1 to train the model.


### STEP 4: Predict use trained DeepDRIM

Input: Trained model from TASK1 STEP2, Representation for other pairs generated by TASK1 STEP1. 

(Example model, trained by healthy B cell scRNA-seq data, XXX !!!!!!!!!!!!)


example command:
 python3 DeepDRIM.py -to_predict True -num_batches 18 -data_path  mHSC_L_representation_nobound/version11/ -output_dir predict_test/ -weight_path keras_cnn_trained_model_shallow.h5 !!!!!!!!!!!!


## TASK 3, The effectiveness of neighbor images, test by simulation data


### STEP 1: Generate Random Network

Run: 

R -f simulation_indirect_demo.R

STEP 2: Generate Representation for the simulated networks

code: generate_input_simulation.py

Indicate the folder that includes all simulated networks as input_dir in the generate_input_simulation.py->main()

run:
python generate_input_simulation.py

### STEP 2: Run CNNC with the representation.



