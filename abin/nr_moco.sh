#!/bin/bash

input_filename=$1
ref_filename=$2
res_prefix=$3

# niftyreg binary path
niftyreg_bin=`which reg_f3d`
if [[ -z $niftyreg_bin ]];
    echo "NiftyReg binaries not found, are you sure it is installed in a directory in your PATH variable?"
fi
afni_bin=`which 3dcalc`
if [[ -z $afni_bin ]];
    echo "AFNI binaries not found, are you sure it is installed in a directory in your PATH variable?"
fi

# grab first volume
3dcalc -a ${input_filename}[0] -expr 'a' -prefix ref.nii
# grab nr volumes 
nt=`3dinfo -nt $input_filename`
for ((i=1; i<$nt; i++)); 
do
    # grab i-th volume
    3dcalc -a ${input_filename}[$i] -expr 'a' -prefix mov.nii
    # niftyreg call
    $niftyreg_bin -ref ref.nii -flo mov.nii -be 0.005 -le 0.05 -cpp ${res_prefix}_CPP$i -res tmp_$i.nii
    rm mov.nii
done
# concat back to 3d+t
3dTcat ref.nii tmp_*.nii -prefix ${res_prefix}.nii

# cleanup
rm ref.nii tmp_*.nii



