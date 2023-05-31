#############################
##########new_set############
#############################
NAMES=./new_dimers_names.txt
NATIVEDIR=../data/new_dimers/PDB
DOCKQ=/proj/berzelius-2021-29/users/x_patbr/DockQ/DockQ.py

#model_1
#reycle 10
#for run in {1..5}
#do
#       MODELDIR='/proj/berzelius-2021-29/users/x_patbr/results/af2/new_dimers/af_std_and_hhblits_msas/model_1/recycle_10/run_'$run'/rewritten_pdbs'
#       for i in {1..8}
#       do
#       NAME=$(sed -n $i'p' $NAMES)
#       echo $NAME
#       python3 $DOCKQ $MODELDIR/$NAME'.pdb' $NATIVEDIR/$NAME'.pdb' >> 'dockqstats_newdimers_af2_af2andhhblitsmsa_model_1_rec10_run'$run'.txt'
#       done
#done





#############################
########Benchmark 4##########
#############################

NAMES=./bench4_names.txt
NATIVEDIR=../data/dockground/PDB/rewritten_complex
DOCKQ=/proj/berzelius-2021-29/users/x_patbr/DockQ/DockQ.py
#/proj/snic2019-35-62/users/x_patbr/DockQ/DockQ.py
#/proj/berzelius-2021-29/users/x_patbr/DockQ/DockQ.py

#########HHblits n2############
#model_1_ptm
#reycle 10

MODELDIR=/proj/snic2019-35-62/users/x_patbr/results/af2/hhblits_n2_merged/model_1_ptm/recycle_10/rewritten_pdbs
#for i in {1..219}
#do
 #NAME=$(sed -n $i'p' $NAMES)
 #echo $NAME
 #python3 $DOCKQ $MODELDIR/$NAME'_u1-'$NAME'_u2.pdb' $NATIVEDIR/$NAME'_bc.pdb' >> dockqstats_bench4_af2_hhblitsn2_model_1_ptm_rec10.txt
#done

#model_1_ptm
#ensemble 8
# MODELDIR=/proj/berzelius-2021-29/users/x_patbr/results/af2/hhblits_msa/model_1_ptm/ensemble_8/rewritten_pdbs
# for i in {1..219}
# do
#   NAME=$(sed -n $i'p' $NAMES)
#   echo $NAME
#   python3 $DOCKQ $MODELDIR/$NAME'_u1-'$NAME'_u2.pdb' $NATIVEDIR/$NAME'_bc.pdb' >> dockqstats_bench4_af2_hhblitsn2_model_1_ptm_ens8.txt
# done


#model_1
#reycle 10
#MODELDIR=/proj/snic2019-35-62/users/x_patbr/results/af2/hhblits_n2_merged/model_1/recycle_10/rewritten_pdbs
#for i in {1..219}
#do
  #NAME=$(sed -n $i'p' $NAMES)
  #echo $NAME
  #python3 $DOCKQ $MODELDIR/$NAME'_u1-'$NAME'_u2.pdb' $NATIVEDIR/$NAME'_bc.pdb' >> dockqstats_bench4_af2_hhblitsn2_model_1_rec10.txt
#done

#model_1
#ensemble 8
MODELDIR=/proj/snic2019-35-62/users/x_patbr/results/af2/hhblits_n2_merged/model_1/ensemble_8/rewritten_pdbs
#for i in {1..219}
#do
  #NAME=$(sed -n $i'p' $NAMES)
  #echo $NAME
  #python3 $DOCKQ $MODELDIR/$NAME'_u1-'$NAME'_u2.pdb' $NATIVEDIR/$NAME'_bc.pdb' >> dockqstats_bench4_af2_hhblitsn2_model_1_ens8.txt
#done

#RF
MODELDIR=/proj/berzelius-2021-29/users/x_arnel/distpred-hhblits
#for i in {1..219}
#do
  #NAME=$(sed -n $i'p' $NAMES)
  #echo $NAME
  #python3 $DOCKQ $MODELDIR/$NAME'_u1_'$NAME'_u2.pdb' $NATIVEDIR/$NAME'_bc.pdb' >> dockqstats_bench4_RF.txt
#done

############AF std msa###########
#model_1_ptm
#reycle 10
# MODELDIR=/proj/berzelius-2021-29/users/x_patbr/results/af2/af2_std_msa/model_1_ptm/recycle_10/rewritten_pdbs
# for i in {1..219}
# do
#   NAME=$(sed -n $i'p' $NAMES)
#   echo $NAME
#   python3 $DOCKQ $MODELDIR/$NAME'_u1-'$NAME'_u2.pdb' $NATIVEDIR/$NAME'_bc.pdb' >> dockqstats_bench4_af2_af2stdmsa_model_1_ptm_rec10.txt
# done
#
# #model_1_ptm
# #ensemble 8
# MODELDIR=/proj/berzelius-2021-29/users/x_patbr/results/af2/af2_std_msa/model_1_ptm/ensemble_8/rewritten_pdbs
# for i in {1..219}
# do
#   NAME=$(sed -n $i'p' $NAMES)
#   echo $NAME
#   python3 $DOCKQ $MODELDIR/$NAME'_u1-'$NAME'_u2.pdb' $NATIVEDIR/$NAME'_bc.pdb' >> dockqstats_bench4_af2_af2stdmsa_model_1_ptm_ens8.txt
# done
#
#
# #model_1
# #reycle 10
# MODELDIR=/proj/berzelius-2021-29/users/x_patbr/results/af2/af2_std_msa/model_1/recycle_10/rewritten_pdbs/
# for i in {1..219}
# do
#   NAME=$(sed -n $i'p' $NAMES)
#   echo $NAME
#   python3 $DOCKQ $MODELDIR/$NAME'_u1-'$NAME'_u2.pdb' $NATIVEDIR/$NAME'_bc.pdb' >> dockqstats_bench4_af2_af2stdmsa_model_1_rec10.txt
# done
#
# #model_1
# #ensemble 8
# MODELDIR=/proj/berzelius-2021-29/users/x_patbr/results/af2/af2_std_msa/model_1/ensemble_8/rewritten_pdbs/
# for i in {1..219}
# do
#   NAME=$(sed -n $i'p' $NAMES)
#   echo $NAME
#   python3 $DOCKQ $MODELDIR/$NAME'_u1-'$NAME'_u2.pdb' $NATIVEDIR/$NAME'_bc.pdb' >> dockqstats_bench4_af2_af2stdmsa_model_1_ens8.txt
# done
#

#model_2
#ensemble 8
MODELDIR=/proj/berzelius-2021-29/users/x_patbr/results/af2/af2_std_msa/model_2/ensemble_8/rewritten_pdbs/
for i in {1..219}
do
   NAME=$(sed -n $i'p' $NAMES)
   echo $NAME
   python3 $DOCKQ $MODELDIR/$NAME'_u1-'$NAME'_u2.pdb' $NATIVEDIR/$NAME'_bc.pdb' >> dockqstats_bench4_af2_af2stdmsa_model_2_ens8.txt
done


#
# ##########AF std and HHblits n2 msas############
# #model_1_ptm
# #reycle 10
# MODELDIR=/proj/berzelius-2021-29/users/x_patbr/results/af2/af_std_and_hhblits_msas/model_1_ptm/recycle_10/rewritten_pdbs/
# for i in {1..219}
# do
#   NAME=$(sed -n $i'p' $NAMES)
#   echo $NAME
#   python3 $DOCKQ $MODELDIR/$NAME'_u1-'$NAME'_u2.pdb' $NATIVEDIR/$NAME'_bc.pdb' >> dockqstats_bench4_af2_af2andhhblitsmsa_model_1_ptm_rec10.txt
# done
#
# #model_1_ptm
# #ensemble_8
# MODELDIR=/proj/berzelius-2021-29/users/x_patbr/results/af2/af_std_and_hhblits_msas/model_1_ptm/ensemble_8/rewritten_pdbs/
# for i in {1..219}
# do
#   NAME=$(sed -n $i'p' $NAMES)
#   echo $NAME
#   python3 $DOCKQ $MODELDIR/$NAME'_u1-'$NAME'_u2.pdb' $NATIVEDIR/$NAME'_bc.pdb' >> dockqstats_bench4_af2_af2andhhblitsmsa_model_1_ptm_ens8.txt
# done
#
#
# #model_1
# #reycle 10
# for run in {1..5}
# do
#  	MODELDIR='/proj/berzelius-2021-29/users/x_patbr/results/af2/af_std_and_hhblits_msas/model_1/recycle_10/run_'$run'/rewritten_pdbs'
#  	for i in {1..219}
#  	do
#   	NAME=$(sed -n $i'p' $NAMES)
#   	echo $NAME
#   	python3 $DOCKQ $MODELDIR/$NAME'_u1-'$NAME'_u2.pdb' $NATIVEDIR/$NAME'_bc.pdb' >> 'dockqstats_bench4_af2_af2andhhblitsmsa_model_1_rec10_run'$run'.txt'
#  	done
# done
#
#
# #model_1
# #ensemble_8
# MODELDIR=/proj/berzelius-2021-29/users/x_patbr/results/af2/af_std_and_hhblits_msas/model_1/ensemble_8/rewritten_pdbs/
# for i in {1..219}
# do
#   NAME=$(sed -n $i'p' $NAMES)
#   echo $NAME
#   python3 $DOCKQ $MODELDIR/$NAME'_u1-'$NAME'_u2.pdb' $NATIVEDIR/$NAME'_bc.pdb' >> dockqstats_bench4_af2_af2andhhblitsmsa_model_1_ens8.txt
# done

################################
############Marks###############
NAMES1=./marks_id1.txt
NAMES2=./marks_id2.txt
NATIVEDIR=/proj/berzelius-2021-29/users/x_patbr/data/PDB/marks/complex
DOCKQ=/proj/berzelius-2021-29/users/x_patbr/DockQ/DockQ.py

#AF
#hhblits MSA, model_1, 10 recycles
MODELDIR=/proj/berzelius-2021-29/users/x_patbr/results/af2/marks/hhblits_msa/model_1/recycle_10/rewritten_pdbs
#for i in {1..1504}
#do
  #ID1=$(sed -n $i'p' $NAMES1)
  #ID2=$(sed -n $i'p' $NAMES2)
  #echo $i,$ID1,$ID2
  #python3 $DOCKQ $MODELDIR/$ID1'-'$ID2'.pdb' $NATIVEDIR/$ID1'_'$ID2'.pdb' >> dockqstats_marks_af2_hhblitsn2_model_1_rec10.txt
#done

#AF std MSA, model_1, 10 recycles
MODELDIR=/proj/berzelius-2021-29/users/x_patbr/results/af2/marks/af2_std_msa/model_1/recycle_10/rewritten_pdbs
#for i in {1..1504}
#do
  #ID1=$(sed -n $i'p' $NAMES1)
  #ID2=$(sed -n $i'p' $NAMES2)
  #echo $i,$ID1,$ID2
  #python3 $DOCKQ $MODELDIR/$ID1'-'$ID2'.pdb' $NATIVEDIR/$ID1'_'$ID2'.pdb' >> dockqstats_marks_af2_af2stdmsa_model_1_rec10.txt
#done

#hhblits MSA: paired+fused, model_1, 10 recycles
MODELDIR=/proj/berzelius-2021-29/users/x_patbr/results/af2/marks/fused_and_hhbllits_msas/model_1/recycle_10
for ri in {2..5}
do
	for i in {1..1504}
	do
  	ID1=$(sed -n $i'p' $NAMES1)
  	ID2=$(sed -n $i'p' $NAMES2)
  	echo $ri,$i,$ID1,$ID2
  	python3 $DOCKQ $MODELDIR/run_$ri/rewritten_pdbs/$ID1'-'$ID2'.pdb' $NATIVEDIR/$ID1'_'$ID2'.pdb' >> dockqstats_marks_af2_pairedandfused_model_1_rec10_run$ri.txt
	done
done

#AF and hhblits MSA, model_1, 10 recycles
MODELDIR=/proj/berzelius-2021-29/users/x_patbr/results/af2/marks/af_std_and_hhblits_msas/model_1/recycle_10
#for run in {2..5}
#do
#for i in {1..1504}
#do
#  ID1=$(sed -n $i'p' $NAMES1)
#  ID2=$(sed -n $i'p' $NAMES2)
#  echo $i,$ID1,$ID2
#  python3 $DOCKQ $MODELDIR/run_$run/rewritten_pdbs/$ID1'-'$ID2'.pdb' $NATIVEDIR/$ID1'_'$ID2'.pdb' >> dockqstats_marks_af2_af2andhhblitsmsa_model_1_rec10_run$run.txt
#done
#done

#RF
MODELDIR=/proj/berzelius-2021-29/users/x_patbr/results/rf/marks
#for i in {1..1504}
#do
  #ID1=$(sed -n $i'p' $NAMES1)
  #ID2=$(sed -n $i'p' $NAMES2)
  #echo $i,$ID1,$ID2
  #MODEL=$MODELDIR/$ID1'_'$ID2'_top*.pdb'
  #python3 $DOCKQ $MODEL $NATIVEDIR/$ID1'_'$ID2'.pdb' >> dockqstats_marks_RF.txt
#done
